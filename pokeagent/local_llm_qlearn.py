import os
import hydra
import wandb
import numpy as np
import gym
import logging
import re
from omegaconf import DictConfig, OmegaConf

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
    # --- 1. Environment & Local LLM Configuration ---
    # Set environment variables for the LLM client
    os.environ["OPENAI_API_BASE"] = cfg.get("llm_base_url", "http://localhost:11434/v1")
    os.environ["OPENAI_API_KEY"] = "ollama" 
    os.environ["LLM_MODEL"] = "deepseek-coder-v2:16b"
    
    print(f"🚀 Starting Local Training with {os.environ['LLM_MODEL']}")
    set_seeds(cfg.seed)
    setup_wandb(OmegaConf.to_container(cfg, resolve=True))

    # --- 2. Initialize Environment & Q-Table ---
    env = gym.make("MountainCar-v0")
    
    # Define discretization parameters
    DISCRETE_OBSERVATION_SPACE_SIZE = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBSERVATION_SPACE_SIZE

    LEARNING_RATE = cfg.lr
    DISCOUNT = 0.95
    EPISODES = cfg.train_iterations
    FEEDBACK_FREQUENCY = cfg.get("feedback_frequency", 100)

    # Epsilon Decay Setup
    epsilon = 0.5
    START_EPSILON_DECAYING = 1
    # Decay over 80% of total episodes to allow for more exploration
    END_EPSILON_DECAYING = int(EPISODES * 0.8)
    epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
    MIN_EPSILON = 0.01

    # Initialize Q-Table
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBSERVATION_SPACE_SIZE + [env.action_space.n]))

    def get_discrete_state(state):
        """Converts continuous state to discrete grid indices with clipping."""
        discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        # Clip to ensure indices are within [0, 19]
        clipped_state = np.clip(discrete_state, 0, np.array(DISCRETE_OBSERVATION_SPACE_SIZE) - 1)
        return tuple(clipped_state.astype(np.int32))

    # Initialize Reward Shaper
    sr = ShapedReward()
    current_reward_func = sr.generate_default_func()

    # --- 3. Training Loop ---
    for ep in range(EPISODES):
        # Reset environment and handle different gym API versions
        obs = env.reset()
        state_raw = obs[0] if isinstance(obs, tuple) else obs
        discrete_state = get_discrete_state(state_raw)
        
        done = False
        episode_length = 0
        sum_rewards = 0
        traj = []

        while not done and episode_length < 600:
            # Action selection (Epsilon-Greedy)
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Step the environment
            new_state, reward, done, info, _ = env.step(action)
            new_discrete_state = get_discrete_state(new_state)
            
            # Record trajectory for LLM feedback
            traj.append((new_state, action, reward))

            # Apply Shaped Reward using CONTINUOUS values (new_state[0], new_state[1])
            # This is critical so the LLM physics logic works correctly.
            shaped_val = current_reward_func(new_state[0], new_state[1], action)

            # Q-Learning Update
            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]
                
                # Formula: Q = (1-a)Q + a(r + shaped_r + discount * max_next_Q)
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                    reward + shaped_val + DISCOUNT * max_future_q
                )
                q_table[discrete_state + (action,)] = new_q
                
            elif new_state[0] >= env.goal_position:
                # Terminal state: Goal reached
                q_table[discrete_state + (action,)] = 0
                log.info(f"Goal reached at episode {ep}")

            discrete_state = new_discrete_state
            episode_length += 1
            sum_rewards += reward

        # --- 4. LLM Feedback Logic ---
        if ep % FEEDBACK_FREQUENCY == 0 and ep != 0:
            log.info(f"Episode {ep}: Calling {os.environ['LLM_MODEL']} for feedback...")
            try:
                # Generate new reward function based on recent trajectory
                new_func = sr.generate_reward_func(traj)
                if callable(new_func):
                    current_reward_func = new_func
                    log.info("Reward function updated successfully.")
            except Exception as e:
                log.warning(f"LLM Logic Error: {e}")

        # Epsilon Decay logic with lower bound
        if END_EPSILON_DECAYING >= ep >= START_EPSILON_DECAYING:
            epsilon = max(MIN_EPSILON, epsilon - epsilon_decay_value)

        # Log metrics to WandB
        wandb.log({
            "reward": sum_rewards, 
            "length": episode_length, 
            "epsilon": epsilon,
            "episode": ep
        })
        
        if ep % 100 == 0:
            print(f"Ep: {ep} | Reward: {sum_rewards:.1f} | Eps: {epsilon:.2f} | Len: {episode_length}")

    env.close()
    print("Training finished.")

if __name__ == "__main__":
    main()