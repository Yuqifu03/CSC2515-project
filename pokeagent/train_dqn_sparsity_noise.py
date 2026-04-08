# Main entry point for target Q-Network logic with or without LLM intrinsic rewards.
import os
import argparse
import csv
import json
import time
import gymnasium as gym
import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim

from env_wrappers import SparseMountainCarWrapper
from utils.llm_teacher import LLMTeacher

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", type=str, choices=["baseline", "llm-direct", "llm-code"], default="baseline")
  parser.add_argument("--llm-backend", type=str, choices=["remote", "local"], default="remote")
  parser.add_argument("--llm-model", type=str, default="claude-sonnet-4-5-20250929")
  parser.add_argument("--sparsity-level", type=int, default=0)
  parser.add_argument("--noise-type", type=str, choices=["none", "gaussian", "logical"], default="none")
  parser.add_argument("--noise-std", type=float, default=0.1)
  parser.add_argument("--llm-freq", type=int, default=100)
  parser.add_argument("--batch-interval", type=int, default=20, help="Steps between batch LLM queries in llm-direct mode.")
  parser.add_argument("--episodes", type=int, default=2000)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--discount-factor", type=float, default=0.99)
  parser.add_argument("--epsilon-start", type=float, default=1.0)
  parser.add_argument("--output-csv", type=str, default="results/run.csv")
  parser.add_argument("--heatmap-episodes", type=int, default=500, help="Number of episodes to record visited states for heatmap.")
  return parser.parse_args()

class DQN(nn.Module):
  def __init__(self, state_dim, action_dim):
    super(DQN, self).__init__()
    self.fc1 = nn.Linear(state_dim, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, action_dim)

    # Slightly increase the variance of network initialization
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.5)
        nn.init.constant_(m.bias, 0.0)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    return self.fc3(x)

class ReplayBuffer:
  def __init__(self, capacity):
    self.memory = deque(maxlen=capacity)

  def push(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def sample(self, batch_size):
    batch = random.sample(self.memory, batch_size)
    state, action, reward, next_state, done = zip(*batch)
    return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

  def __len__(self):
    return len(self.memory)

def main():
  args = parse_args()

  env = gym.make("MountainCar-v0")
  env = SparseMountainCarWrapper(env, sparsity_level=args.sparsity_level)

  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.n

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")

  policy_net = DQN(state_dim, action_dim).to(device)
  target_net = DQN(state_dim, action_dim).to(device)
  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval()

  optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
  criterion = nn.MSELoss()

  REPLAY_MEMORY_SIZE = 10000
  MIN_REPLAY_MEMORY_SIZE = 1000
  BATCH_SIZE = 64
  TARGET_UPDATE_FREQ = 500
  memory = ReplayBuffer(REPLAY_MEMORY_SIZE)

  epsilon = args.epsilon_start
  MIN_EPSILON = 0.03
  END_EPSILON_DECAYING = 500
  epsilon_decay_value = (epsilon - MIN_EPSILON) / END_EPSILON_DECAYING

  # ====== Output Setup ======
  out_dir = os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else '.'
  os.makedirs(out_dir, exist_ok=True)

  # Episode-level CSV (enriched)
  f_csv = open(args.output_csv, 'w', newline='')
  csv_writer = csv.writer(f_csv)
  csv_writer.writerow(["episode", "total_reward", "max_position", "intrinsic_reward",
                        "steps", "success", "api_calls"])

  # LLM debug log (.jsonl) — only for LLM modes
  llm_log_path = args.output_csv.replace(".csv", "_llm_debug.jsonl")
  f_llm_log = None
  if args.mode in ["llm-direct", "llm-code"]:
    f_llm_log = open(llm_log_path, 'w')

  # Step-level: visited states for heatmap (first N episodes only)
  visited_positions = []
  visited_velocities = []
  HEATMAP_EP_LIMIT = args.heatmap_episodes

  teacher = None
  if args.mode in ["llm-direct", "llm-code"]:
    teacher = LLMTeacher(backend=args.llm_backend, model_name=args.llm_model)

  # Enforce minimum batch interval of 20 in llm-direct mode
  if args.mode == "llm-direct" and args.batch_interval < 20:
    print(f"[WARN] batch_interval={args.batch_interval} is too small, enforcing minimum of 20")
    args.batch_interval = 20

  current_reward_code = None
  intrinsic_reward_func = None

  # Spatial hashing proxy for caching
  discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / [20, 20]
  def get_hash_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    discrete_state = np.round(discrete_state).astype(int)
    discrete_state[0] = min(19, max(0, discrete_state[0]))
    discrete_state[1] = min(19, max(0, discrete_state[1]))
    return tuple(discrete_state)

  llm_reward_cache = {}
  recent_trajectory = deque(maxlen=10)
  recent_max_positions = []

  # Batch interval scoring buffer for llm-direct
  segment_buffer = []

  # Global API call counter
  total_api_calls = 0

  def compile_reward_func(code_str):
    if not code_str:
      return None
    local_scope = {}
    try:
      exec(code_str, {}, local_scope)
      if 'intrinsic_reward' in local_scope:
        return local_scope['intrinsic_reward']
      else:
        return None
    except Exception as e:
      return None

  def log_llm_event(episode, step, event_type, state_info, prompt_snippet, response_snippet, reward_value):
    """Write a structured debug entry to the LLM .jsonl log."""
    nonlocal f_llm_log
    if f_llm_log is None:
      return
    entry = {
      "ts": time.strftime("%H:%M:%S"),
      "episode": episode,
      "step": step,
      "type": event_type,
      "state": state_info,
      "prompt": prompt_snippet[:500] if prompt_snippet else None,
      "response": response_snippet[:500] if response_snippet else None,
      "reward": reward_value
    }
    f_llm_log.write(json.dumps(entry) + "\n")
    f_llm_log.flush()

  if args.llm_freq == 0 and args.mode != "baseline":
    summary = "Begin of training. Agent has no progress."
    if args.mode == "llm-code":
      current_reward_code = teacher.get_reward_function(summary)
      print(f"LLM generated new code:\n{current_reward_code}")
      intrinsic_reward_func = compile_reward_func(current_reward_code)
      total_api_calls += 1
      log_llm_event(0, 0, "code_gen", None, summary, current_reward_code, None)

  # Warm-up phase
  print("Starting warm-up... Filling experience pool with random actions")
  warmup_state, _ = env.reset()
  for _ in range(2000):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    r_intrinsic = 0.0

    total_reward = reward + r_intrinsic
    memory.push(warmup_state, action, total_reward, next_state, terminated)

    if done:
      warmup_state, _ = env.reset()
    else:
      warmup_state = next_state
  print("Warm-up complete!")

  total_steps = 0
  for episode in range(args.episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    episode_intrinsic_reward = 0
    episode_max_position = -1.2
    episode_steps = 0
    episode_api_calls = 0
    recent_trajectory.clear()

    # Update llm-code iteratively
    if args.mode == "llm-code" and args.llm_freq > 0 and episode % args.llm_freq == 0 and episode > 0:
      avg_max_pos = np.mean(recent_max_positions) if len(recent_max_positions) > 0 else -1.2
      summary = f"Over the last {args.llm_freq} episodes, the agent's average maximum position reached was {avg_max_pos:.3f}. Goal is {env.unwrapped.goal_position}."
      print(f"Querying LLM at episode {episode} for new evaluation code...")
      current_reward_code = teacher.get_reward_function(summary, prev_code=current_reward_code)
      print(f"LLM generated new code:\n{current_reward_code}")
      intrinsic_reward_func = compile_reward_func(current_reward_code)
      recent_max_positions = []
      total_api_calls += 1
      episode_api_calls += 1
      log_llm_event(episode, 0, "code_gen",
                     {"avg_max_pos": float(avg_max_pos)},
                     summary, current_reward_code, None)

    while not done:
      total_steps += 1
      episode_steps += 1
      recent_trajectory.append(state.tolist())

      if np.random.random() > epsilon:
        with torch.no_grad():
          state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
          q_values = policy_net(state_tensor)
          action = q_values.argmax().item()
      else:
        action = np.random.randint(0, action_dim)

      new_state, r_env, terminated, truncated, _ = env.step(action)
      done = terminated or truncated
      episode_max_position = max(episode_max_position, new_state[0])

      # Step-level instrumentation: record visited states for heatmap
      if episode < HEATMAP_EP_LIMIT:
        visited_positions.append(float(new_state[0]))
        visited_velocities.append(float(new_state[1]))

      r_intrinsic = 0.0

      # Batch interval scoring for llm-direct
      if args.mode == "llm-direct":
        hash_s = get_hash_state(state)
        if hash_s in llm_reward_cache:
          r_intrinsic = llm_reward_cache[hash_s]
        else:
          # Accumulate into segment buffer for batch scoring
          segment_buffer.append((hash_s, action))
          # Flush the batch when it reaches the interval size
          if len(segment_buffer) >= args.batch_interval:
            # Filter out steps already in cache
            uncached = [(s, a) for s, a in segment_buffer if s not in llm_reward_cache]
            if uncached and len(llm_reward_cache) < 400:
              print(f"Batch query: {len(uncached)} uncached / {len(segment_buffer)} total steps")
              batch_rewards = teacher.get_batch_rewards(uncached)
              for (bs, _ba), br in zip(uncached, batch_rewards):
                llm_reward_cache[bs] = br
              total_api_calls += 1
              episode_api_calls += 1
              log_llm_event(episode, episode_steps, "batch_direct",
                            {"cache_size": len(llm_reward_cache), "batch_size": len(uncached)},
                            f"{len(uncached)} uncached steps",
                            str([llm_reward_cache.get(s) for s, _ in uncached[:5]]) + "...",
                            None)
            segment_buffer.clear()
            # Now try to get the reward for the current step from cache
            if hash_s in llm_reward_cache:
              r_intrinsic = llm_reward_cache[hash_s]
      elif args.mode == "llm-code" and intrinsic_reward_func:
        try:
          r_intrinsic = intrinsic_reward_func(state, action)
        except:
          r_intrinsic = 0.0
      # Reward Scaling and sensitivity bounds
      if r_intrinsic != 0:
        r_intrinsic = np.clip(r_intrinsic, -1.0, 1.0) * 0.1

      if args.noise_type == "gaussian" and r_intrinsic != 0:
        r_intrinsic += np.random.normal(0, args.noise_std)
      elif args.noise_type == "logical" and r_intrinsic != 0:
        # noise condition: 20% random inversion per DQN-plan instructions
        if np.random.random() < 0.2:
          r_intrinsic = -r_intrinsic

      total_reward = r_env + r_intrinsic
      episode_reward += r_env
      episode_intrinsic_reward += r_intrinsic

      memory.push(state, action, total_reward, new_state, terminated)
      state = new_state

      # Replay and Train
      if len(memory) > MIN_REPLAY_MEMORY_SIZE:
        states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)

        current_q_values = policy_net(states_t).gather(1, actions_t)
        with torch.no_grad():
          max_next_q_values = target_net(next_states_t).max(1)[0].unsqueeze(1)

        target_q_values = rewards_t + (1 - dones_t) * args.discount_factor * max_next_q_values

        loss = criterion(current_q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      if total_steps % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # ====== End of Episode Bookkeeping ======
    recent_max_positions.append(episode_max_position)

    # Determine success: terminated AND position >= goal
    success = terminated and (state[0] >= env.unwrapped.goal_position)

    # Flush remaining segment buffer at end of episode
    if args.mode == "llm-direct" and segment_buffer:
      uncached = [(s, a) for s, a in segment_buffer if s not in llm_reward_cache]
      if uncached and len(llm_reward_cache) < 400:
        print(f"End-of-episode flush: {len(uncached)} uncached / {len(segment_buffer)} total steps")
        batch_rewards = teacher.get_batch_rewards(uncached)
        for (bs, _ba), br in zip(uncached, batch_rewards):
          llm_reward_cache[bs] = br
        total_api_calls += 1
        episode_api_calls += 1
      segment_buffer.clear()

    if epsilon > MIN_EPSILON:
      epsilon -= epsilon_decay_value
      epsilon = max(MIN_EPSILON, epsilon)

    # Write enriched episode-level CSV row
    csv_writer.writerow([episode, episode_reward, episode_max_position,
                         episode_intrinsic_reward, episode_steps,
                         int(success), episode_api_calls])

    if episode >= 100 and episode % 100 == 0:
      cache_info = f", Cache: {len(llm_reward_cache)}/400" if args.mode == "llm-direct" else ""
      api_info = f", API calls: {total_api_calls}" if args.mode != "baseline" else ""
      print(f"Episode: {episode}, Epsilon: {epsilon:.2f}, Reward: {episode_reward:.1f}, "
            f"Intrinsic: {episode_intrinsic_reward:.1f}, Max Pos: {episode_max_position:.2f}, "
            f"Steps: {episode_steps}, Success: {success}{cache_info}{api_info}")

  f_csv.close()
  if f_llm_log:
    f_llm_log.close()

  # ====== Post-Training Visualizations ======
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  os.makedirs('results', exist_ok=True)

  # --- 1. State-Space Visit Frequency Heatmap (hist2d) ---
  if len(visited_positions) > 0:
    print(f"Generating visit heatmap from {len(visited_positions)} recorded states...")
    fig, ax = plt.subplots(figsize=(10, 6))
    h = ax.hist2d(visited_positions, visited_velocities, bins=[50, 50],
                  range=[[-1.2, 0.6], [-0.07, 0.07]], cmap='magma')
    fig.colorbar(h[3], ax=ax, label='Visit Frequency')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title(f'State-Space Visit Heatmap ({args.mode}, first {HEATMAP_EP_LIMIT} episodes)')
    fig.savefig(f"results/visit_heatmap_{args.mode}_ep{args.episodes}.png", dpi=150)
    plt.close(fig)
    print("Visit heatmap saved.")
  else:
    print("No visited states recorded, skipping visit heatmap.")

  # --- 2. Q-Value Heatmap ---
  print("Generating Q-Value Heatmap...")
  pos_grid = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 50)
  vel_grid = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 50)
  q_values_matrix = np.zeros((50, 50))

  for i, p in enumerate(pos_grid):
    for j, v in enumerate(vel_grid):
      s_tensor = torch.FloatTensor([[p, v]]).to(device)
      with torch.no_grad():
        q_values_matrix[j, i] = policy_net(s_tensor).max(1)[0].item()

  fig, ax = plt.subplots(figsize=(10, 6))
  im = ax.imshow(q_values_matrix, extent=[pos_grid[0], pos_grid[-1], vel_grid[0], vel_grid[-1]],
                 origin='lower', aspect='auto', cmap='viridis')
  fig.colorbar(im, ax=ax, label='Max Q-Value')
  ax.set_xlabel('Position')
  ax.set_ylabel('Velocity')
  ax.set_title(f'DQN Q-Value Coverage Heatmap ({args.mode})')
  fig.savefig(f"results/qvalue_heatmap_{args.mode}_ep{args.episodes}.png", dpi=150)
  plt.close(fig)
  print("Q-Value heatmap saved.")

  print(f"\nTraining complete. Total API calls: {total_api_calls}")
  print(f"CSV: {args.output_csv}")
  if f_llm_log:
    print(f"LLM debug log: {llm_log_path}")

  env.close()

if __name__ == "__main__":
  main()
