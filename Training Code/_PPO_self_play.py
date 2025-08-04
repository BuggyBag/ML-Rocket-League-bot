import os
import sys
import glob
import random
import numpy as np
import torch
import traceback

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
#from rlgym.rocket_league.action_parsers import LookupTableAction as DiscreteAction
from rlgym.rocket_league.action_parsers import LookupTableAction as DefaultAction
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator

from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.reward_functions import GoalReward, CombinedReward
from rlgym.rocket_league.sim import RocketSimEngine

from rlgym_compat.game_state import GameState, PlayerData, PhysicsObject

sys.path.append(r"C:\Users\lexiv\AppData\Local\RLBotGUIX\RLBotPackDeletable\RLBotPack-master\RLBotPack\Necto\Nexto")
from bot import Nexto
from nexto_obs import encode_gamestate



def make_lookup_table():
    actions = []
    # Ground
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
    # Aerial
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                        actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
    return np.array(actions)

def encode_gamestate(state):
    """
    Encodes gamestate to exactly 454 floats.
    Works with your Car dataclass structure (physics nested).
    """
    obs = []

    # Ball info (9 floats)
    obs.extend(state.ball.position)  # 3
    obs.extend(state.ball.linear_velocity)  # 3
    obs.extend(state.ball.angular_velocity)  # 3

    # For 10 cars, each with 48 features (480 total)
    car_ids = sorted(state.cars.keys())
    for i in range(10):
        if i < len(car_ids):
            car = state.cars[car_ids[i]]

            # Correct physics-based access
            obs.extend(car.physics.position)  # 3 floats (x, y, z)
            obs.extend(car.physics.linear_velocity)  # 3 floats (x, y, z)
            obs.extend(car.physics.angular_velocity)  # 3

            # Other car state info
            obs.append(float(car.boost_amount))  # 1
            obs.append(float(car.on_ground))  # 1
            obs.append(float(car.has_flip))  # 1
            obs.append(float(car.is_demoed))  # 1

            # These vectors might not exist, so fill with zeros
            obs.extend([0.0, 0.0, 0.0])  # forward_vector
            obs.extend([0.0, 0.0, 0.0])  # up_vector
            obs.extend([0.0, 0.0, 0.0])  # right_vector

            # Team info
            obs.append(float(car.team_num))  # 1

            # These stats likely don't exist on your car object, so we zero-pad
            obs.extend([0.0] * 6)  # goals, saves, shots, demolishes, assists, etc.

        else:
            # Pad if less than 10 cars
            obs.extend([0.0]*48)

    time_per_tick = 1 / 120  # assuming 120hz

    # Misc game info (3 floats)
    obs.append(float(state.tick_count * time_per_tick))  # where `time_per_tick` is the time duration per tick
    obs.append(float(state.ball.position[2]))  # Ball's height (z-axis)
    obs.append(float(state.tick_count))  # 1

    obs = np.array(obs, dtype=np.float32)

    # Ensure correct length
    if obs.shape[0] != 454:
        raise ValueError(f"encode_gamestate output wrong length: got {obs.shape[0]}, expected 454")

    return obs

# ===== ENSURE NEXTO COMPATIBILITY =====
class CompatGameState:
    def __init__(self, ball, players, blue_score, orange_score, boost_pads, inverted_ball, inverted_boost_pads):
        self.ball = ball
        self.players = players
        self.blue_score = blue_score
        self.orange_score = orange_score
        self.boost_pads = boost_pads
        self.inverted_ball = inverted_ball
        self.inverted_boost_pads = inverted_boost_pads
        
        # Add game_info with seconds_elapsed
        self.game_info = type('GameInfo', (), {'seconds_elapsed': 0.0})

class CompatGameStateArrayWrapper:
    """
    Wraps a CompatGameState and converts all tuples in PhysicsObject to np.ndarray,
    so encode_gamestate (which expects .tolist()) works.
    """
    def __init__(self, compat_state):
        # Copy all attributes
        for k, v in compat_state.__dict__.items():
            setattr(self, k, v)
        # Convert ball and inverted_ball
        for attr in ['ball', 'inverted_ball']:
            obj = getattr(self, attr, None)
            if obj is not None:
                for field in ['position', 'linear_velocity', 'angular_velocity']:
                    val = getattr(obj, field, None)
                    if isinstance(val, tuple):
                        setattr(obj, field, np.array(val, dtype=np.float32))
        # Convert players' car_data and inverted_car_data if present
        if hasattr(self, "players"):
            for player in self.players:
                for car_attr in ['car_data', 'inverted_car_data']:
                    car_obj = getattr(player, car_attr, None)
                    if car_obj is not None:
                        for field in ['position', 'linear_velocity', 'angular_velocity']:
                            val = getattr(car_obj, field, None)
                            if isinstance(val, tuple):
                                setattr(car_obj, field, np.array(val, dtype=np.float32))

def safe_to_tuple(arr):
    """
    Ensures the output is always a tuple of length 3.
    Used for passing to PhysicsObject, which expects tuples.
    """
    if arr is None:
        return (0.0, 0.0, 0.0)
    if isinstance(arr, np.ndarray):
        return tuple(arr.tolist())
    return tuple(arr)

class DictToObj:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = DictToObj(v)
            setattr(self, k, v)

def convert_rlgym_to_compat(rlgym_state):
    """
    Converts an rlgym state to a CompatGameState with tuples for PhysicsObject.
    """
    if isinstance(rlgym_state, dict):
        rlgym_state = DictToObj(rlgym_state)

    # Ball
    ball = PhysicsObject(
        position=safe_to_tuple(rlgym_state.ball.position),
        linear_velocity=safe_to_tuple(rlgym_state.ball.linear_velocity),
        angular_velocity=safe_to_tuple(rlgym_state.ball.angular_velocity)
    )

    # Inverted ball (mirror X and Y)
    inverted_ball = PhysicsObject(
        position=safe_to_tuple([-ball.position[0], -ball.position[1], ball.position[2]]),
        linear_velocity=safe_to_tuple([-ball.linear_velocity[0], -ball.linear_velocity[1], ball.linear_velocity[2]]),
        angular_velocity=safe_to_tuple(ball.angular_velocity)
    )

    # Players
    players = []
    for car_id, car in rlgym_state.cars.items():
        player = PlayerData()
        player.team_num = car.team_num

        # Main car data
        player.car_data = PhysicsObject(
            position=safe_to_tuple(car.physics.position),
            linear_velocity=safe_to_tuple(car.physics.linear_velocity),
            angular_velocity=safe_to_tuple(car.physics.angular_velocity)
        )

        # Inverted car data (mirror X and Y)
        player.inverted_car_data = PhysicsObject(
            position=safe_to_tuple([-player.car_data.position[0], -player.car_data.position[1], player.car_data.position[2]]),
            linear_velocity=safe_to_tuple([-player.car_data.linear_velocity[0], -player.car_data.linear_velocity[1], player.car_data.linear_velocity[2]]),
            angular_velocity=safe_to_tuple(player.car_data.angular_velocity)
    )

    player.is_demoed = bool(car.is_demoed)
    player.on_ground = bool(car.on_ground)
    player.ball_touched = bool(car.ball_touches)
    player.has_flip = bool(car.has_flip)
    player.boost_amount = float(car.boost_amount)
    players.append(player)

    # Scores
    blue_score = getattr(rlgym_state, "blue_score", 0)
    orange_score = getattr(rlgym_state, "orange_score", 0)

    # Boost pads (active states as bool array)
    if hasattr(rlgym_state, "boost_pads") and rlgym_state.boost_pads is not None:
        try:
            boost_pads = np.array([pad.is_active for pad in rlgym_state.boost_pads], dtype=bool)
        except Exception:
            boost_pads = np.array(rlgym_state.boost_pads, dtype=bool)
    else:
        boost_pads = np.zeros(34, dtype=bool)  # Default for standard map

    inverted_boost_pads = boost_pads.copy()

    return CompatGameState(
        ball=ball,
        players=players,
        blue_score=blue_score,
        orange_score=orange_score,
        boost_pads=boost_pads,
        inverted_ball=inverted_ball,
        inverted_boost_pads=inverted_boost_pads
    )

# ===== NEXTO init MODEL =====
class NextoOpponent:
    def __init__(self, action_space, name="Nexto", team=1, index=1):
        self.action_space = action_space
        self.bot = Nexto(name, team, index) # If NextoBot needs parameters, add them here
        self.lookup_table = make_lookup_table()
        self.lookup_dict = {tuple(row): idx for idx, row in enumerate(self.lookup_table)}

    def act(self, rlgym_state):
        try:
            compat_state = convert_rlgym_to_compat(rlgym_state)
            # Wrap to ensure arrays for encode_gamestate
            compat_state_array = CompatGameStateArrayWrapper(compat_state)
            obs = encode_gamestate(compat_state_array)
            output, _ = self.bot.get_output(compat_state_array)
            output = np.array(output).squeeze().astype(np.int32)
            output_tuple = tuple(output)
            action_index = self.lookup_dict.get(output_tuple, None)
            if action_index is None:
                print("Action vector not found in lookup dict, using random action")
                return self.action_space.sample()
            action_index = int(action_index)  # ensure scalar int
            return action_index
        except Exception as e:
            print(f"Opponent policy error: {e}, using random action (Nexto())")
            traceback.print_exc()
            return self.action_space.sample()


# ===== FREEZE MODEL =====
class FrozenPolicyOpponent:
    def __init__(self, model_paths, device="cpu"):
        self.model_paths = model_paths
        self.models = [PPO.load(path, device=device) for path in model_paths]
        self.action_space = spaces.Discrete(len(make_lookup_table()))

    def act(self, rlgym_state):
        try:
            compat_state = convert_rlgym_to_compat(rlgym_state)
            obs = encode_gamestate(compat_state)
            action = self.bot.get_output(np.array([obs]))
        
            # Defensive: ensure action is scalar integer
            if isinstance(action, np.ndarray):
                if action.size == 1:
                    action = action.item()
                elif action.ndim == 1 and action.shape[0] == 1:
                    action = action[0]
                else:
                    # If action is multi-dimensional, flatten safely
                    action = int(action.flatten()[0])
            elif isinstance(action, list):
                action = int(action[0])
            else:
                # If action is already scalar, just convert to int
                action = int(action)
        
            action_index = int(action )  # ensure scalar int
            return action_index
        except Exception as e:
            #print(f"Opponent policy error: {e}, using random action (from Frozen())")
            return self.action_space.sample()

def get_latest_model_paths(save_dir, pattern="ppo_selfplay_v*.zip", n=1):
    """Return a list of the n latest PPO self-play model paths in the directory."""
    model_files = glob.glob(os.path.join(save_dir, pattern))
    if not model_files:
        return []
    # Sort by version number in filename (assumes filenames like ppo_selfplay_v1.zip)
    model_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_v')[-1]))
    return model_files[-n:]

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
            car = state.cars[agent]
            car_pos = car.physics.position
            car_vel = car.physics.linear_velocity
            ball_pos = state.ball.position
            ball_vel = state.ball.linear_velocity

            # Reward proximity to ball
            distance = np.linalg.norm(car_pos - ball_pos)
            rewards[agent] = 1 / (distance + 1e-6) * 0.05  # small incentive

            # Reward moving towards ball
            to_ball_dir = (ball_pos - car_pos)
            to_ball_dir /= np.linalg.norm(to_ball_dir) + 1e-6
            forward_dir = car.physics.forward
            vel_towards_ball = np.dot(forward_dir, to_ball_dir) * np.linalg.norm(car_vel)
            rewards[agent] += vel_towards_ball * 0.0005

            # Aerial incentive
            if ball_pos[2] > 300:  # Ball is off the ground
                rewards[agent] += 0.02  # Encourage ball control in air

            # Penalize being idle
            if np.linalg.norm(car_vel) < 100:
                rewards[agent] -= 0.01

            # Bonus for touching ball
            if state.cars[agent].ball_touches:
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
    def __init__(self, rlgym_env, opponent_policy=None):
        super().__init__()
        self.rlgym_env = rlgym_env
        self.opponent_policy = opponent_policy
        self.num_agents = 1

        # Set observation space to 454 floats
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(454,), dtype=np.float32
        )

        # Action space (Discrete 90, as before)
        self.action_space = spaces.Discrete(90)

        print(f"Observation space shape: {self.observation_space.shape}")
        print(f"Observation space: {self.observation_space}")

    def reset(self, seed=None, options=None):
        # Reset RLGym environment
        self.rlgym_env.reset()

        # Get observation encoded as 454 floats
        obs = self.encode_gamestate(self.rlgym_env.state)

        return obs, {}

    def step(self, action): 
        # Convert action to int
        if isinstance(action, np.ndarray):
            action = int(np.squeeze(action).item())
        else:
            action = int(action)

        agent_ids = list(self.rlgym_env.state.cars.keys())
        actions_dict = {}

        # Agent's action
        actions_dict[agent_ids[0]] = np.array([action])

        # Opponent's action
        if self.opponent_policy is not None:
            try:
                opponent_action = self.opponent_policy.act(self.rlgym_env.state)
                if isinstance(opponent_action, np.ndarray):
                    opponent_action = int(opponent_action.flatten()[0])
                else:
                    opponent_action = int(opponent_action)
                actions_dict[agent_ids[1]] = np.array([opponent_action])
            except Exception as e:
                print(f"Opponent policy error: {e}, using random action")
                actions_dict[agent_ids[1]] = np.array([self.action_space.sample()])
        else:
            actions_dict[agent_ids[1]] = np.array([self.action_space.sample()])

        result = self.rlgym_env.step(actions_dict)

        if len(result) == 5:
            _, reward_dict, terminated, truncated, info = result
        elif len(result) == 4:
            _, reward_dict, done, info = result
            terminated = done
            truncated = False
        else:
            raise ValueError(f"Unexpected step result length: {len(result)}")

        reward = float(reward_dict[agent_ids[0]])

        obs = self.encode_gamestate(self.rlgym_env.state)

        return obs, reward, terminated, truncated, info

    def encode_gamestate(self, state):
        """
        Encodes gamestate to exactly 454 floats.
        Works with your Car dataclass structure (physics nested).
        """
        obs = []

        # Ball info
        obs.extend(state.ball.position)
        obs.extend(state.ball.linear_velocity) 
        obs.extend(state.ball.angular_velocity)

        # For 10 cars, each with 48 features (480 total)
        car_ids = sorted(state.cars.keys())
        for i in range(10):
            if i < len(car_ids):
                car = state.cars[car_ids[i]]

                # Correct physics-based access
                obs.extend(car.physics.position)  # 3 floats (x, y, z)
                obs.extend(car.physics.linear_velocity)  # 3 floats (x, y, z)
                obs.extend(car.physics.angular_velocity)  # 3 floats (x, y, z)

                # Other car state info
                obs.append(float(car.boost_amount))
                obs.append(float(car.on_ground))
                obs.append(float(car.has_flip)) 
                obs.append(float(car.is_demoed))  

                # These vectors might not exist, so fill with zeros
                obs.extend([0.0, 0.0, 0.0])  # forward_vector
                obs.extend([0.0, 0.0, 0.0])  # up_vector
                obs.extend([0.0, 0.0, 0.0])  # right_vector

                # Team info
                obs.append(float(car.team_num))

                # These stats likely don't exist on your car object, so we zero-pad
                obs.extend([0.0] * 6)  # goals, saves, shots, demolishes, assists, etc.

            else:
                # Pad if less than 10 cars
                obs.extend([0.0]*48)

        time_per_tick = 1 / 120  # assuming 120hz

        # Misc game info
        obs.append(float(state.tick_count * time_per_tick))  # where `time_per_tick` is the time duration per tick
        obs.append(float(state.ball.position[2]))  # Ball's height (z-axis)
        obs.append(float(state.tick_count)) 

        obs = np.array(obs, dtype=np.float32)

        # Ensure correct length
        if obs.shape[0] != 454:
            raise ValueError(f"encode_gamestate output wrong length: got {obs.shape[0]}, expected 454")

        return obs

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

    save_path = "saved_models"
    # Get the latest model for opponent pool
    latest_model_paths = get_latest_model_paths(save_path, n=1)  # n=1 for only the latest, or n=3 for a pool

    # If there is at least one saved model, use it as the opponent
    if latest_model_paths:
        opponent_policy = FrozenPolicyOpponent(latest_model_paths, device="cpu")
    else:
        opponent_policy = NextoOpponent(action_space=spaces.Discrete(90), name="Nexto", team=1, index=1)  # Use Nexto as initial opponent

    return RLGymGymWrapper(rlgym_env, opponent_policy=opponent_policy)

# ===== TRAINING SETUP =====
def main():
    env = DummyVecEnv([make_self_play_env])
    print(f"Using observation space: {env.observation_space}")

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

    total_timesteps = 50_000_000
    save_every = 2_000_000  # Save every 2 million steps
    save_path = r"C:\Users\lexiv\AppData\Local\RLBotGUIX\MyBots\KaosX9\src\saved_models"

    os.makedirs(save_path, exist_ok=True)

    for i in range(1, total_timesteps // save_every + 1):
        print(f"Training iteration {i}")
        model.learn(total_timesteps=save_every, reset_num_timesteps=False)
        model.save(os.path.join(save_path, f"ppo_selfplay_v{i}"))

    model.save(os.path.join(save_path, "kaosx9_bot"))

# --------------------------------------------------------------------- #
# --------------------------------------------------------------------- #
# --------------------------------------------------------------------- #

def dummy_main():
    env = DummyVecEnv([make_self_play_env])
    print(f"Using observation space: {env.observation_space}")

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

    total_timesteps = 500_000
    save_every = 100_000
    save_path = r"C:\Users\lexiv\AppData\Local\RLBotGUIX\MyBots\KaosX9\src\saved_models"

    os.makedirs(save_path, exist_ok=True)

    for i in range(1, total_timesteps // save_every + 1):
        print(f"Training iteration {i}")
        model.learn(total_timesteps=save_every, reset_num_timesteps=False)
        model.save(os.path.join(save_path, f"ppo_selfplay_v{i}"))

    model.save(os.path.join(save_path, "kaosx9_bot"))

if __name__ == "__main__":
    dummy_main()