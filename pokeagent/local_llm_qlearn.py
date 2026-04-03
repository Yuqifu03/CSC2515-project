import os
import hydra
import wandb
import numpy as np
import gym
import logging
import re
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import pandas as pd

# Import original utility functions and the Reward Shaping class
from utils.setup import setup_wandb, set_seeds
from utils.reward import ShapedReward

log = logging.getLogger(__name__)

def clean_llm_code(code_string):
    """Strips markdown backticks if the LLM includes them."""
    pattern = r"```python\n(.*?)\n```"
    match = re.search(pattern, code_string, re.DOTALL)
    if match:
        return match.group(1)
    return code_string.replace("```python", "").replace("```", "").strip()

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # --- 1. Global Settings ---
    os.environ["OPENAI_API_BASE"] = cfg.get("llm_base_url", "http://localhost:11434/v1")
    os.environ["OPENAI_API_KEY"] = "ollama" 
    os.environ["LLM_MODEL"] = "deepseek-coder-v2:16b"
    
    # Study parameters
    test_frequencies = [50, 75, 100, 125, 150, 175, 200] 
    # num_seeds = 3  
    seeds=[42, 100, 2026, 3030, 4040, 5050, 6060, 7070, 8080, 9999]                        
    episodes_per_run = 600
    
    # Data containers for the final local plot
    final_rewards_history = [] 
    llm_costs_history = []    

    print(f"🚀 Starting Multi-Run Study. Episodes per run: {episodes_per_run}")

    # --- 2. Multi-Level Experiment Loop ---
    for freq in test_frequencies:
        seed_results = []
        total_calls_for_this_freq = 0
        
        print(f"\n>> Testing Feedback Frequency: {freq}")
        
        for current_seed in seeds:
            # Skip if already exists in CSV
            if os.path.exists("experiment_results.csv"):
                existing_df = pd.read_csv("experiment_results.csv")
                if ((existing_df['frequency'] == freq) & (existing_df['seed'] == current_seed)).any():
                    print(f"   ⏩ Skipping freq {freq} seed {current_seed} (already finished)")
                    continue
            # Create a unique name for this specific run
            run_name = f"freq_{freq}_seed_{current_seed}"
            print(f"   Starting WandB Run: {run_name}")
            
            # Initialize a fresh WandB run for every seed/freq combination
            wandb.init(
                project="reward-shaping-rl", 
                name=run_name,
                config={
                    "frequency": freq,
                    "seed": current_seed,
                    "model": os.environ["LLM_MODEL"],
                    **OmegaConf.to_container(cfg, resolve=True)
                },
                reinit=True # Allow multiple runs in one script
            )
            
            # Reset seeds and environment
            set_seeds(current_seed)
            env = gym.make("MountainCar-v0", max_episode_steps=600)
            
            # Initialize Q-Table and logic fresh for each seed
            DISCRETE_OBSERVATION_SPACE_SIZE = [20] * len(env.observation_space.high)
            discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBSERVATION_SPACE_SIZE
            q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBSERVATION_SPACE_SIZE + [env.action_space.n]))
            
            epsilon = 0.5
            LEARNING_RATE = cfg.lr
            DISCOUNT = 0.95
            END_EPSILON_DECAYING = int(episodes_per_run)
            epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - 1)
            MIN_EPSILON = 0.01
            
            sr = ShapedReward()
            current_reward_func = sr.generate_default_func()
            run_rewards = []
            run_llm_calls = 0

            # --- Training Loop (Episodes) ---
            for ep in range(episodes_per_run):
                obs = env.reset()
                state_raw = obs[0] if isinstance(obs, tuple) else obs
                
                def get_discrete_state(state):
                    ds = (state - env.observation_space.low) / discrete_os_win_size
                    clipped = np.clip(ds, 0, np.array(DISCRETE_OBSERVATION_SPACE_SIZE) - 1)
                    return tuple(clipped.astype(np.int32))

                discrete_state = get_discrete_state(state_raw)
                done = False
                sum_rewards = 0
                episode_length = 0
                traj = []

                while not done and episode_length < 600:
                    if np.random.random() > epsilon:
                        action = np.argmax(q_table[discrete_state])
                    else:
                        action = np.random.randint(0, env.action_space.n)

                    step_result = env.step(action)
                    
                    if len(step_result) == 5:
                        new_state, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        new_state, reward, done, info = step_result

                    if new_state[0] >= 0.5:
                        done = True

                    # new_state, reward, done, info, _ = env.step(action)
                    new_discrete_state = get_discrete_state(new_state)
                    traj.append((new_state, action, reward))

                    try:
                        shaped_val = current_reward_func(new_state[0], new_state[1], action)
                    except Exception:
                        import traceback
                        sr.last_error = traceback.format_exc() 
                        shaped_val = 0
                    # if ep % 50 == 0:
                    #     print(f"Step {ep} | Pos: {float(np.mean(obs[0])):.2f} | Shaped Reward: {float(np.mean(shaped_val)):.4f}")
                    if not done:
                        max_future_q = np.max(q_table[new_discrete_state])
                        current_q = q_table[discrete_state + (action,)]
                        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                            reward + shaped_val + DISCOUNT * max_future_q
                        )
                        q_table[discrete_state + (action,)] = new_q
                    elif new_state[0] >= env.goal_position:
                        q_table[discrete_state + (action,)] = reward + shaped_val

                    discrete_state = new_discrete_state
                    sum_rewards += reward
                    episode_length += 1

                run_rewards.append(sum_rewards)
                
                # Log EVERY episode to WandB to see the learning curve
                wandb.log({
                    "episode_reward": sum_rewards,
                    "episode_length": episode_length,
                    "epsilon": epsilon,
                    "episode": ep
                })

                # LLM Feedback Logic
                if ep % freq == 0 and ep != 0:
                    run_llm_calls += 1
                    try:
                        new_func = sr.generate_reward_func(traj)
                        if callable(new_func):
                            current_reward_func = new_func
                    except Exception:
                        pass

                if END_EPSILON_DECAYING >= ep >= 1:
                    epsilon = max(MIN_EPSILON, epsilon - epsilon_decay_value)

            # Record final results for this specific run
            last_50_avg = np.mean(run_rewards[-50:])
            seed_results.append(last_50_avg)
            total_calls_for_this_freq = run_llm_calls 
            
            # Finalize this WandB run
            wandb.finish()
            env.close()

            csv_file = "experiment_results.csv"
            result_data = {
                "frequency": freq,
                "seed": current_seed,
                "avg_reward_last_50": last_50_avg,
                "llm_calls": total_calls_for_this_freq,
                "model": os.environ.get("LLM_MODEL", "unknown")
            }
            
            df = pd.DataFrame([result_data])
            # If file doesn't exist, write with header; else append without header
            df.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))
            print(f"   ✅ Result saved to {csv_file}")

        # Average seeds for this frequency for the local summary plot
        avg_reward_for_freq = np.mean(seed_results)
        final_rewards_history.append(avg_reward_for_freq)
        llm_costs_history.append(total_calls_for_this_freq)

    # --- 3. Final Summary Plot (Local Save) ---
    print("\n Generating final plot from experiment_results.csv...")
    all_data = pd.read_csv("experiment_results.csv")
    
    summary_df = all_data.groupby('frequency').agg({
            'avg_reward_last_50': 'mean',
            'llm_calls': 'mean' 
        }).sort_values('frequency').reset_index()

    test_frequencies = summary_df['frequency'].tolist()
    final_rewards_history = summary_df['avg_reward_last_50'].tolist()
    llm_costs_history = summary_df['llm_calls'].tolist()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_r = 'tab:blue'
    ax1.set_xlabel('Feedback Frequency (Episodes)')
    ax1.set_ylabel('Avg Reward (Last 50 Eps)', color=color_r)
    ax1.plot(test_frequencies, final_rewards_history, marker='o', color=color_r, linewidth=2, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color_r)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color_c = 'tab:red'
    ax2.set_ylabel('Total LLM Calls (Cost)', color=color_c)
    ax2.plot(test_frequencies, llm_costs_history, marker='s', linestyle='--', color=color_c, linewidth=2, label='LLM Cost')
    ax2.tick_params(axis='y', labelcolor=color_c)

    plt.title('Final Study: Reward Performance vs. LLM Cost')
    fig.tight_layout()
    plt.savefig('final_summary_plot.png')
    
    print("\n✨ All experiments finished.")
    print(f"Final results plotted and saved locally as 'final_summary_plot.png'.")
    plt.show()
    # plt.savefig('final_summary_plot.png')

if __name__ == "__main__":
    main()