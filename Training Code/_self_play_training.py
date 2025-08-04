import os
import numpy as np
import torch
from torch.nn import Tanh

import rlgym
import rlgym_tools
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, SubprocVecEnv

from rlgym.api import RLGym, RewardFunction, DoneCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.action_parsers import LookupTableAction as DefaultAction
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.reward_functions import GoalReward, CombinedReward
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence
from rlgym.rocket_league.sim import RocketSimEngine


class CustomReward(RewardFunction):
    def __init__(self):
        self.reward = CombinedReward(
            [
                GoalReward(),
            ]
        )

    def reset(self, initial_state):
        self.reward.reset(initial_state)

    def get_reward(self, state, previous_action, previous_state):
        return self.reward.get_reward(state, previous_action, previous_state)


terminal_conditions = [
    GoalCondition(),
    NoTouchTimeoutCondition(1800),
]

state_mutators = MutatorSequence(
    [
        FixedTeamSizeMutator(1),
        KickoffMutator(),
    ]
)


def create_env():
    return RLGym(
        reward_function=CustomReward(),
        terminal_conditions=terminal_conditions,
        obs_builder=DefaultObs(),
        state_mutators=state_mutators,
        action_parser=DefaultAction(),
        spawn_opponents=True,
        team_size=1,
        tick_skip=8,
        sim=RocketSimEngine()
    )


def make_env():
    def _init():
        return create_env()
    return _init


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    num_envs = 8
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    policy_kwargs = dict(
        activation_fn=Tanh,
        net_arch=[512, 512, dict(pi=[256, 256, 256], vf=[256, 256, 256])]
    )

    model_path = "models/rocket_league_ppo.zip"

    if os.path.exists(model_path):
        print("Loading existing model...")
        model = PPO.load(model_path, env=env, device="auto", custom_objects={"n_envs": env.num_envs})
    else:
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            n_epochs=10,
            policy_kwargs=policy_kwargs,
            learning_rate=5e-5,
            ent_coef=0.01,
            vf_coef=1.0,
            gamma=0.99,
            verbose=3,
            batch_size=2048,
            n_steps=256,
            tensorboard_log="logs",
            device="auto"
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=250_000 // num_envs,
        save_path="./models/",
        name_prefix="rocket_league_ppo"
    )

    try:
        total_timesteps = model.num_timesteps
        target_timesteps = total_timesteps + 50_000_000

        while model.num_timesteps < target_timesteps:
            model.learn(250_000, reset_num_timesteps=False, callback=checkpoint_callback)
            model.save(model_path)
            print(f"Saved model at step {model.num_timesteps}")

    except KeyboardInterrupt:
        print("Training interrupted, saving model...")
        model.save(model_path)
        print("Model saved, exiting.")
