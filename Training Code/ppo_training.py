import os
import json
import numpy as np

import rlgym
from rlgym.utiles.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import VelocityReward, EventReward
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.action_parsers import ContinuousAction

import gym
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# Paths
extracted_replays_folder = "./extracted_data"  # Folder where your extracted replays are
output_model_path = "./rocket_league_ppo_model"  # Path to save the trained PPO model

# Define the custom Rocket League Gym environment
class RocketLeagueEnv(gym.Env):
    def __init__(self, replay_files, time_step=0.1):
        super(RocketLeagueEnv, self).__init__()
        self.replay_files = replay_files
        self.time_step = time_step
        self.current_replay = None
        self.current_frame_idx = 0
        self.max_steps = 1000  # Max steps per episode (to prevent infinite loops)
        
        # Action space: Assuming the bot can make 5 actions (e.g., throttle, steer, jump, etc.)
        # Adjust this as needed based on your action space design.
        self.action_space = spaces.Discrete(5)

        # Observation space: Assuming state includes car and ball positions, velocities, etc.
        # The size of the observation space can change depending on the exact features you want to use.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

    def reset(self):
        # Randomly choose a replay file to start with
        self.current_replay = self._load_random_replay()
        self.current_frame_idx = 0
        return self._get_state()

    def step(self, action):
        self.current_frame_idx += 1
        
        if self.current_frame_idx >= len(self.current_replay):
            done = True
            reward = 0  # You can define your own reward structure
            info = {}
            state = self.reset()  # Reset the environment to start a new episode
        else:
            state = self._get_state()
            done = False
            reward = self._calculate_reward(action)  # Define your reward structure here
            info = {}

        return state, reward, done, info

    def _load_random_replay(self):
        """ Load a random replay file from the extracted replays folder. """
        replay_file = np.random.choice(self.replay_files)
        with open(replay_file, "r") as f:
            return json.load(f)

    def _get_state(self):
        """ Get the state from the current frame (car and ball data). """
        frame = self.current_replay[self.current_frame_idx]
        car_state = frame["cars"][0]  # Assuming there is only one car, adjust if multiple cars
        ball_state = frame["ball"]

        car_pos = np.array(car_state["pos"])
        car_vel = np.array(car_state["vel"])
        ball_pos = np.array(ball_state["pos"])
        ball_vel = np.array(ball_state["vel"])

        # Flattening the state (example: position + velocity for car and ball)
        state = np.concatenate([car_pos, car_vel, ball_pos, ball_vel])

        return state

    def _calculate_reward(self, action):
        """ Define how the reward should be calculated based on the action taken. """
        # Placeholder: reward function should be based on bot's action and the environment state
        # For now, a simple reward: +1 for each step (this should be adjusted based on the task)
        return 1


# Load all the extracted replay files
replay_files = [os.path.join(extracted_replays_folder, f) for f in os.listdir(extracted_replays_folder) if f.endswith(".json")]

# Create the Rocket League environment
env = RocketLeagueEnv(replay_files)

# Wrap the environment in a DummyVecEnv for parallelism (if needed)
env = DummyVecEnv([lambda: env])

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the PPO model
model.learn(total_timesteps=100000)  # Adjust the total_timesteps based on your needs

# Save the trained model
model.save(output_model_path)
print(f"Model saved to {output_model_path}")

# Testing the trained model (optional)
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
    