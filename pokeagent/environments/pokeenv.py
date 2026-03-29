import asyncio

import numpy as np
from gym.spaces import Box, Space
import logging
from gym.utils.env_checker import check_env

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data.gen_data import GenData
from poke_env.player import (
    Gen8EnvSinglePlayer,
    RandomPlayer,
)

from pokeagent.agents.pokegym import PokeGen8Gym
from pokeagent.models.dqn import DQNAgent
from pokeagent.utils.reward import ShapedReward

def train_m1(env: PokeGen8Gym, agent: DQNAgent, episodes:int, sr:ShapedReward=None, device=None, save_dir=None):
    """
    Training method 1: Sequential learning of reward function. Code is a bit jank but it gets the job done for now.
    """
    META_STEPS = 5
    shaped_reward = 0
    shaped_reward_func = sr.generate_default_func()
    
    for meta_step in range(META_STEPS):
        for ep in range(episodes):
            print('-=-=-=-=- NEW EP:', ep)
            state, info = env.reset()
            s, battle = state
            steps = 0
            average_loss = 0
            while True:
                
                # agent step and learn
                action = agent.action(s) # [agent.action(state)]
                new_state, reward, terminated, truncated, info = env.step(action)
                new_s, new_battle = new_state[0], new_state[1]
                done = terminated or truncated
                shaped_reward = shaped_reward_func(battle, new_battle)

                agent.cache(s, action, reward + shaped_reward, new_s, done)
                q, loss = agent.optimize()
                # logger.log_step(reward, loss, q)
                
                # state = new_state
                s = new_s
                battle = new_battle
                
                if done:
                    print('done!', done)
                    break
                
                if loss is not None and loss > 0:
                    average_loss += loss
                    steps += 1
                    
            # log episode info
            # logger.log_episode()
            if ep > 0 and ep % 500 == 0:
                evaluate(agent, 20)

            if (steps > 0):
                average_loss = average_loss / steps 
            else:
                average_loss = -1
                
            logging.info('average_loss', average_loss)
        agent.save_all()
        won, total_games = evaluate(agent, 20)
        shaped_reward_func = sr.generate_reward_func(won / total_games)
        agent = DQNAgent(embedding_size=env.input_size, 
                num_actions=env.action_space.n,
                device=device,
                evaluate=False,
                lr=0.001,
                save_dir=save_dir,
                warmup=100,
                name="iterate_{meta_step}")
        sr.save()
    env.close()
    sr.save()

def train_m2(env: PokeGen8Gym, agent: DQNAgent, episodes:int, sr:ShapedReward=None, device=None, save_dir=None):
    """
    Training method 2: Tree-based. Takes a while because I'm not using threading or async training...
    """
    META_STEPS = 5
    NUM_LEAVES = 5
    shaped_reward = 0
    shaped_reward_func = sr.generate_default_func()
    
    for meta_step in range(META_STEPS):
        MAX_REWARDS = []
        for k in range(NUM_LEAVES):
            for ep in range(episodes):
                print('-=-=-=-=- NEW EP:', ep)
                state, info = env.reset()
                s, battle = state
                steps = 0
                average_loss = 0
                while True:
                    
                    # agent step and learn
                    action = agent.action(s) # [agent.action(state)]
                    new_state, reward, terminated, truncated, info = env.step(action)
                    new_s, new_battle = new_state[0], new_state[1]
                    done = terminated or truncated
                    shaped_reward = shaped_reward_func(battle, new_battle)

                    agent.cache(s, action, reward + shaped_reward, new_s, done)
                    q, loss = agent.optimize()
                    # logger.log_step(reward, loss, q)
                    
                    # state = new_state
                    s = new_s
                    battle = new_battle
                    
                    if done:
                        print('done!', done)
                        break
                    
                    if loss is not None and loss > 0:
                        average_loss += loss
                        steps += 1
                        
                # log episode info
                # logger.log_episode()
                if ep > 0 and ep % 500 == 0:
                    evaluate(agent, 20)

                if (steps > 0):
                    average_loss = average_loss / steps 
                else:
                    average_loss = -1
                    
                logging.info('average_loss', average_loss)
            agent.save_all()
            won, total_games = evaluate(agent, 20)
            shaped_reward_func = sr.generate_reward_func(won / total_games)
            agent = DQNAgent(embedding_size=env.input_size, 
                    num_actions=env.action_space.n,
                    device=device,
                    evaluate=False,
                    lr=0.001,
                    save_dir=save_dir,
                    warmup=100,
                    name="iterate_{meta_step}_{k}")
            MAX_REWARDS.append(won / total_games)
        sr.save()
    env.close()
    sr.save()

def train_m3(env: PokeGen8Gym, agent: DQNAgent, episodes:int, sr:ShapedReward=None, device=None, save_dir=None):
    """
    Training method 3 
    """
    shaped_reward = 0
    shaped_reward_func = sr.generate_default_func() # sr.generate_reward_func([])
    sr.save()
    
    for ep in range(episodes):
        print('-=-=-=-=- NEW EP:', ep)
        state, info = env.reset()
        s, battle = state
        steps = 0
        average_loss = 0
        while True:
            
            # agent step and learn
            action = agent.action(s) # [agent.action(state)]
            new_state, reward, terminated, truncated, info = env.step(action)
            new_s, new_battle = new_state[0], new_state[1]
            done = terminated or truncated
            shaped_reward = shaped_reward_func(battle, new_battle)

            agent.cache(s, action, reward + shaped_reward, new_s, done)
            q, loss = agent.optimize()
            # logger.log_step(reward, loss, q)
            
            # state = new_state
            s = new_s
            battle = new_battle
            
            if done:
                print('done!', done)
                break
            
            if loss is not None and loss > 0:
                average_loss += loss
                steps += 1
                
        # log episode info
        # logger.log_episode()
        if ep > 0 and ep % 500 == 0:
            won, total_games = evaluate(agent, 20)
            # shaped_reward_func = sr.generate_reward_func(won / total_games)
            sr.save()

        if (steps > 0):
            average_loss = average_loss / steps 
        else:
            average_loss = -1
            
        logging.info('average_loss', average_loss)
        
    env.close()
    agent.save_all()
    sr.save()

def evaluate(agent: DQNAgent, episodes:int):
    eval_env = PokeGen8Gym(set_team=True, opponent="random") # change later

    for ep in range(episodes):
        state, info = eval_env.reset()
        s, battle = state
        while True:
            
            # agent step and learn
            action = agent.action(s) # [agent.action(state)]
            new_state, reward, terminated, truncated, info = eval_env.step(action)
            new_s, new_battle = new_state[0], new_state[1]
            done = terminated or truncated
            
            # state = new_state
            s = new_s
            
            if done:
                logging.info(f'eval step {ep}/{episodes}')
                print('done!', done)
                break
                
    logging.info(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.close()
    return eval_env.n_won_battles, eval_env.n_finished_battles

def evalw(agent: DQNAgent, eval_env, episodes:int):

    for ep in range(episodes):
        state, info = eval_env.reset()
        s, battle = state
        while True:
            
            # agent step and learn
            action = agent.action(s) # [agent.action(state)]
            new_state, reward, terminated, truncated, info = eval_env.step(action)
            new_s, new_battle = new_state[0], new_state[1]
            done = terminated or truncated
            
            # state = new_state
            s = new_s
            
            if done:
                print('done!', done, reward)
                break
                
    logging.info(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    return eval_env.n_won_battles, eval_env.n_finished_battles

async def main():
    # First test the environment to ensure the class is consistent
    # with the OpenAI API
    test_env = PokeGen8Gym(set_team=True, opponent="random")
    # check_env(test_env)
    test_env.close()

    # Create one environment for training and one for evaluation
    # opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = PokeGen8Gym(set_team=True, opponent="random")
    # opponent = RandomPlayer(battle_format="gen8randombattle")
    # eval_env = SimpleRLPlayer(
        # battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    # )

    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape
    
    n_steps = 10
    done = False
    while not done:
        # Random action
        action = train_env.action_space.sample()
        (obs, battle), reward, done, info, what = train_env.step(action)
        print(battle, obs, reward, done)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
