import os
import random
import numpy as np
import torch

import rlgym
import rlgym_tools
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

from rlgym.api import RLGym
from rlgym.api import RewardFunction
from rlgym.api import DoneCondition
from rlgym.api import AgentID
from rlgym.api import StateMutator

from rlgym.rocket_league.common_values import BALL_RADIUS
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.action_parsers import LookupTableAction as DiscreteAction
from rlgym.rocket_league.action_parsers import LookupTableAction as DefaultAction
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator

from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.reward_functions import GoalReward, CombinedReward
from rlgym.rocket_league.sim import RocketSimEngine



# ===== Custom Reward Function =====
class CustomGroundAirReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.last_ball_touch_height = 0

    def reset(self, agents: list[AgentID], initial_state: GameState, shared_info: dict):
        self.last_ball_touch_height = initial_state.ball.position[2]

    def get_rewards(self, agents: list[AgentID], state: GameState, is_terminated: dict, is_truncated: dict, shared_info: dict) -> dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car_pos = state.cars[agent].position
            car_vel = state.cars[agent].velocity
            ball_pos = state.ball.position
            ball_vel = state.ball.velocity

            # Reward proximity to ball
            distance = np.linalg.norm(car_pos - ball_pos)
            rewards[agent] = 1 / (distance + 1e-6) * 0.05  # small incentive

            # Reward moving towards ball
            to_ball_dir = (ball_pos - car_pos)
            to_ball_dir /= np.linalg.norm(to_ball_dir) + 1e-6
            forward_dir = state.cars[agent].forward()
            vel_towards_ball = np.dot(forward_dir, to_ball_dir) * np.linalg.norm(car_vel)
            rewards[agent] += vel_towards_ball * 0.0005

            # Aerial incentive
            if ball_pos[2] > 300:  # Ball is off the ground
                rewards[agent] += 0.02  # Encourage ball control in air

            # Penalize being idle
            if np.linalg.norm(car_vel) < 100:
                rewards[agent] -= 0.01

            # Bonus for touching ball
            if state.cars[agent].has_touch:
                rewards[agent] += 0.5
                # Reward aerial touches more
                if ball_pos[2] > 300:
                    rewards[agent] += 1.0

        return rewards

# ===== Done Condition =====
class CustomGoalScoredCondition(DoneCondition):
    def reset(self, initial_state: GameState, shared_info: dict):
        pass

    def is_done(self, state: GameState, shared_info: dict) -> bool:
        # Check if the ball is in the goal
        ball_pos = state.ball.position
        if abs(ball_pos[0]) > 5120 and abs(ball_pos[1]) < 5120:  # Goal scoring range
            return True
        return False

# ===== Ball Initialization =====
class BallInitializationMutator(StateMutator):
    def apply(self, state, shared_info):
        # Initialize ball
        state.ball.position = np.array([0, 0, 93.15])
        state.ball.linear_velocity = np.zeros(3)
        state.ball.angular_velocity = np.zeros(3)

        # Initialize cars
        for car in state.cars.values():
            car.physics.position = np.array([-1000, 0, 17])
            car.physics.linear_velocity = np.zeros(3)
            car.physics.angular_velocity = np.zeros(3)
            car.has_jumped = True
            car.has_double_jumped = True

    def reset(self, state, shared_info):
        self.apply(state, shared_info)
        return state

# ===== Environment Setup =====
class RLGymGymWrapper(gym.Env):
    def __init__(self, rlgym_env):
        super().__init__()
        self.rlgym_env = rlgym_env
        self.num_agents = 2  # 2 if it's 1v1
        self.observation_space = self.rlgym_env.observation_space
        self.action_space = self.rlgym_env.action_space

        # Initialize observation space dynamically
        sample_obs, _ = self.rlgym_env.reset()
        self.observation_space = self._convert_obs_space(sample_obs)

        # Force correct action space manually
        self.action_space = spaces.Discrete(90)

    def _process_string_obs(self, obs):
        """
        Convert string-based observation (like 'blue-0') into a numeric value.
        Adjust this method according to your observation structure.
        """
        if obs.startswith('blue-'):
            # Example: Extract '0' from 'blue-0' and convert to float
            return float(obs.split('-')[1])
        elif obs.startswith('orange-'):
            # Example: Extract '0' from 'orange-0' and convert to float
            return float(obs.split('-')[1])
        else:
            # Return a default value if unable to parse the string (you can adjust this as needed)
            return 0.0

    def _process_obs(self, obs):
        """
        This method processes each observation individually (whether numeric or string).
        It's used in both reset and step methods.
        """
        processed_obs = []
        for o in obs:
            if isinstance(o, str):
                processed_obs.append(self._process_string_obs(o))
            else:
                processed_obs.append(o)
        return np.array(processed_obs, dtype=np.float32)

    def _convert_obs_space(self, sample_obs):
        # Print to inspect sample_obs and understand its structure
        #print("Sample observation before conversion:", sample_obs)

        processed_obs = []
        for obs in sample_obs:
            if isinstance(obs, str):
                # Process string observation, e.g., 'blue-0' -> numeric conversion
                processed_obs.append(self._process_string_obs(obs))
            else:
                # Directly append numeric observations
                processed_obs.append(obs)

        processed_obs = np.array(processed_obs, dtype=np.float32)

        # Define observation space based on processed observations
        low = -np.inf * np.ones_like(processed_obs)
        high = np.inf * np.ones_like(processed_obs)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Get the initial observation and info from the rlgym environment
        obs, info = self.rlgym_env.reset()

        # Process observation before returning
        obs = self._process_obs(obs)
        return obs, info

    def step(self, action):
        # Ensure action is numpy array and shape matches number of agents
        if np.isscalar(action):
            action = np.array([action])
        elif isinstance(action, np.ndarray):
            if action.shape == (1,):
                action = action.squeeze(0)
            elif len(action.shape) == 2 and action.shape[0] == 1:
                action = action.squeeze(0)
            if np.isscalar(action):
                action = np.array([action])

        num_agents = len(self.rlgym_env.state.cars)
        if len(action) != num_agents:
            raise ValueError(f"Action length {len(action)} does not match number of agents {num_agents}")

        agent_ids = list(self.rlgym_env.state.cars.keys())
        actions_dict = {agent_id: action[i] for i, agent_id in enumerate(agent_ids)}

        obs, reward, terminated, truncated, info = self.rlgym_env.step(actions_dict)
        done = terminated or truncated
        return obs, reward, done, info

def make_self_play_env():
    # Set team size using FixedTeamSizeMutator (1v1)
    state_mutator = state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        BallInitializationMutator(),
        KickoffMutator()  
    )

    # Use RocketSim as the transition engine
    transition_engine = RocketSimEngine()

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=DefaultObs(),
        action_parser=DefaultAction(),
        reward_fn=CustomGroundAirReward(),
        termination_cond=GoalCondition(),
        truncation_cond=NoTouchTimeoutCondition(timeout_seconds=30),
        transition_engine=transition_engine
    )

    return RLGymGymWrapper(rlgym_env)

# ===== TRAINING SETUP =====
def main():
    env = DummyVecEnv([make_self_play_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=4096,
        batch_size=1024,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        learning_rate=2.5e-4,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    total_timesteps = 20_000_000  # ~6-8 hours if you have good hardware
    save_every = 2_000_000  # Save every 2 million steps
    save_path = "saved_models"

    os.makedirs(save_path, exist_ok=True)

    for i in range(1, total_timesteps // save_every + 1):
        print(f"Training iteration {i}")
        model.learn(total_timesteps=save_every, reset_num_timesteps=False)
        model.save(os.path.join(save_path, f"ppo_selfplay_v{i}"))

    model.save(os.path.join(save_path, "final_model"))

if __name__ == "__main__":
    main()


    