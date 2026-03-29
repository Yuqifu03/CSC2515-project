# import utils.query_llm as query_llm
# from typing import List
# import os
# import logging

# class ShapedReward(object):
#     """
#     Class for generating shaped reward function.
#     """

#     def __init__(self, save_dir:str):
#         self.GOAL_PROMPT = """
#         You are an agent playing Pokemon Showdown, the popular online battle simulator that allows you to build your own team of Pokemon and battle against other players in real-time.
#         [GOAL] Your goal is to defeat your opponent's Pokemon by making them faint using your six Pokemon. You are using the following team: 
#         {team}
#         """
#         self.PARAM_PROMPT = """
#         [PARAMS] Both inputs to your reward function are an AbstractBattle, which has the following format:
#         You can access your opponenent's team with next_battle.opponent_team, which is a list of Pokemon objects.
#         You can access your current team with next_battle.team, which is a list of Pokemon. You can loop over your Pokemon's stats in the following way. Use these stats however you want.
#         for mon in battle.team.values():
#             active = float(mon.active)
#             fainted = float(mon.fainted)
#             health = mon.current_hp
#             type_1=battle.opponent_active_pokemon.type_1,
#             type_2=battle.opponent_active_pokemon.type_2,
#         for move in next_battle.available_moves
#             move.base_power # powerful moves may be better!


#         For example, to reward defeating a Pokemon, you could write:
#         def reward(prev_battle_state, next_battle_state):
#             prev_num_fainted = len([mon for mon in prev_battle_state.opponent_team.values() if mon.fainted])
#             next_num_fainted = len([mon for mon in next_battle_state.opponent_team.values() if mon.fainted])
#             return next_num_fainted - prev_num_fainted

#         As another example, to reward attacking your opponent, you could write:
#         def reward(prev_battle_state, next_battle_state):
#             prev_health = len([mon.current_hp for mon in prev_battle_state.opponent_team.values()])
#             curr_health = len([mon.current_hp for mon in next_battle_state.opponent_team.values()])
#             return curr_health - prev_health / 1000.0 # normalizing factor
#         """
#         self.TRAJECTORY = """Your previous reward function was {code}. It gave you a reward of {reward}. Improve upon this by creating another unique function!
#         !"""
#         self.PROMPT = """When I ask you a question, only respond with the code which is the answer. 
#         No padding words before or after. Code a function in Python called "reward()" for the Pokemon Showdown API, 
#         where the arguments are the previous state prev_battle_state and the next state next_battle_state. You are designing the reward for the transition
#         that was just taken. Do not provide any code other than the function definition for "def reward(prev_battle_state, next_battle_state):". Make sure it returns a reward.
#         The reward function will be dependent on all the arguments in a non-trivial way. Use as much knowledge about Pokemon Showdown that you have
#         to design this function.

#         {goal}

#         {trajectory}

#         {param}

#         """

#         self.log_of_responses = []
#         self.valid_code_history = ['none']
#         self.team = self.extract_team()
#         self.save_dir = save_dir

#     def extract_team(self):
#         with open('data/team1.txt') as f:
#             team = f.read()
#         return team
    
#     def build_prompt(self, reward: float, failed : bool):
#         prompt = self.PROMPT.format(
#             goal=self.GOAL_PROMPT.format(team=self.team),
#             param=self.PARAM_PROMPT,
#             trajectory=self.TRAJECTORY.format(code=self.valid_code_history[-1], reward=reward),
#         )

#         if failed:
#             prefix = "Make sure to define a function def reward() in Python that exactly follows the arguments specified and returns one reward value.\n"
#             prompt = prefix + prompt
#         return prompt

#     def generate_default_func(self):
#         """
#         Default reward function.
#         """
#         code = """
# def reward(prev_battle_state, next_battle_state):
#     prev_fainted = [mon for mon in prev_battle_state.opponent_team.values() if mon.fainted]
#     next_fainted = [mon for mon in next_battle_state.opponent_team.values() if mon.fainted]
    
#     prev_total_hp = sum([mon.current_hp for mon in prev_battle_state.opponent_team.values()])
#     next_total_hp = sum([mon.current_hp for mon in next_battle_state.opponent_team.values()])
    
#     reward_defeat = len(next_fainted) - len(prev_fainted)
#     reward_damage = (prev_total_hp - next_total_hp) / 1000.0
    
#     return reward_defeat + reward_damage
#         """
#         exec(code, globals())
#         return reward

#     def generate_reward_func(self, reward:float):
#         """
#         Current format hard-coded for MountainCar or Showdown to work and produce less errors.
#         """
#         # while True:
#         failed = False
#         # trajectory = trajectory[-50:]
#         code = None
#         for i in range(2):
#             try:
#                 prompt = self.build_prompt(reward, failed=failed)
#                 print("[P]:", prompt)
#                 cost, code = query_llm.query_gpt(prompt)
#                 self.log_of_responses.append({"prompt": prompt, "code": code})
#                 logging.info('code', code)
#                 exec(code, globals())
#             except:
#                 self.valid_code_history.append('failed: ' + code)
#                 if failed:
#                     print("Error: failed again!")
#                     print(code)
#                 else:
#                     print("Error in trying to define function!")
#                 failed = True
#                 continue

#             # try:
#             #     _ = reward(0, 0, 0)
#             # except:
#             #     print("Reward arguments are wrong!")
#             #     print(reward.__dict__)
#             self.valid_code_history.append(code)
#             return code
#         return self.generate_default_func()

#     def dump(self):
#         print(self.log_of_responses)

#     def save(self):
#         with open(os.path.join(self.save_dir, 'reward_code.txt'), 'w') as f:
#             f.write(str(self.valid_code_history))


# if __name__ == "__main__":
#     sr = ShapedReward()
#     reward = sr.generate_reward_func([0,0,0])
#     sr.dump()
#     print(reward(1, 2, 3))

import utils.query_llm as query_llm
from typing import List
import os
import logging
import re

class ShapedReward(object):
    """
    Class for generating shaped reward functions for MountainCar-v0.
    """

    def __init__(self, save_dir: str = "./logs"):
        # --- MOUNTAINCAR SPECIFIC PROMPTS ---
        self.GOAL_PROMPT = """
        [GOAL] You are an AI agent helping a car drive up a hill to reach a yellow flag. 
        The environment is MountainCar-v0. The car is at the bottom of a valley and must build momentum 
        by driving back and forth to reach the goal at position 0.5.

        [HINT]: To reach 0.5, the car MUST build momentum by swinging. Encourage high velocity (abs(vel)) and reward the agent when the Total Energy (Height + Speed) increases. DO NOT penalize velocity.
        """
        
        self.PARAM_PROMPT = """
        [PARAMS] pos (position), vel (velocity), action.
        [PHYSICS] The car needs Mechanical Energy to climb. 
        Energy = Potential Energy + Kinetic Energy.
        Potential Energy is roughly: (pos + 0.5)^2
        Kinetic Energy is: 0.5 * vel^2

        [INSTRUCTION] Please create a reward function that provides a total value between -1.0 and +2.0. 
        Make sure the reward for "building energy" (swinging) is significantly larger than the -1 penalty per step. 
        Example: return (height_reward + velocity_reward) * scale
        """

        self.TRAJECTORY = """Your previous reward function was:
        {code}
        It resulted in a total episode reward of {reward}. 
        Improve this function to help the car reach the goal faster!
        """

        self.PROMPT = """Only respond with the Python code for the function. 
        No conversational text. No markdown backticks. 
        Define a function: def reward(pos, vel, action):
        
        {goal}
        {trajectory}
        {param}
        """

        self.log_of_responses = []
        self.valid_code_history = ['None']
        self.save_dir = save_dir

    def build_prompt(self, reward: float, failed: bool):
        prompt = self.PROMPT.format(
            goal=self.GOAL_PROMPT,
            param=self.PARAM_PROMPT,
            trajectory=self.TRAJECTORY.format(
                code=self.valid_code_history[-1], 
                reward=reward
            ),
        )
        if failed:
            prompt = "The previous code failed to execute. Ensure valid Python syntax.\n" + prompt
        return prompt

    def generate_default_func(self):
        """Default fallback reward function for MountainCar."""
        def reward(pos, vel, action):
            # Basic potential-based reward: encourage moving away from the center
            return abs(vel) * 1.0
        return reward

    def generate_reward_func(self, trajectory_data):
        current_score = sum([t[2] for t in trajectory_data]) if isinstance(trajectory_data, list) else 0.0
        
        failed = False
        code = None
        model_name = os.getenv("LLM_MODEL", "llama3.1")

        for i in range(2):
            try:
                prompt = self.build_prompt(current_score, failed=failed)
                
                print(f"\n[LLM Request] Sending prompt to {model_name}...") 
                
                cost, code = query_llm.query_gpt(prompt, model=model_name)
                code = self._clean_code(code)
                
                print("\n" + "="*30)
                print(f"NEW REWARD CODE GENERATED BY {model_name}:")
                print(code)
                print("="*30 + "\n")
                
                self.log_of_responses.append({"prompt": prompt, "code": code})

                ldict = {}
                exec(code, globals(), ldict)
                if 'reward' in ldict:
                    self.valid_code_history.append(code)
                    return ldict['reward']
                
            except Exception as e:
                print(f"❌ LLM Code Execution Error: {e}")
                self.valid_code_history.append(f"failed: {code}")
                failed = True
                continue

        return self.generate_default_func()

    def _clean_code(self, code_str):
        """Removes markdown formatting often added by LLMs."""
        if "```" in code_str:
            code_str = re.sub(r"```python\n|```", "", code_str)
        return code_str.strip()

    def dump(self):
        print(self.log_of_responses)