from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3

import json
import os
import sys
import glob
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from stable_baselines3 import PPO
import numpy as np

sys.path.append(r"C:\Users\lexiv\AppData\Local\RLBotGUIX\RLBotPackDeletable\RLBotPack-master\RLBotPack\Necto\Nexto")
#from nexto_obs import encode_gamestate
from rlgym_compat.game_state import PhysicsObject, PlayerData  



KICKOFF_CONTROLS = (
    11 * 4 * [SimpleControllerState(throttle=1, boost=True)]
    + 4 * 4 * [SimpleControllerState(throttle=1, boost=True, steer=-1)]
    + 2 * 4 * [SimpleControllerState(throttle=1, jump=True, boost=True)]
    + 1 * 4 * [SimpleControllerState(throttle=1, boost=True)]
    + 1 * 4 * [SimpleControllerState(throttle=1, yaw=0.8, pitch=-0.7, jump=True, boost=True)]
    + 13 * 4 * [SimpleControllerState(throttle=1, pitch=1, boost=True)]
    + 10 * 4 * [SimpleControllerState(throttle=1, roll=1, pitch=0.5)]
)

KICKOFF_NUMPY = np.array([
    [scs.throttle, scs.steer, scs.pitch, scs.yaw, scs.roll, scs.jump, scs.boost, scs.handbrake]
    for scs in KICKOFF_CONTROLS
])

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

class CarPhysics:
    def __init__(self, position, linear_velocity, angular_velocity, boost_amount, on_ground, has_flip, is_demoed, team_num):
        self.physics = type('Physics', (), {})()
        self.physics.position = position
        self.physics.linear_velocity = linear_velocity
        self.physics.angular_velocity = angular_velocity
        self.boost_amount = boost_amount
        self.on_ground = on_ground
        self.has_flip = has_flip
        self.is_demoed = is_demoed
        self.team_num = team_num

class BallPhysics:
    def __init__(self, position, linear_velocity, angular_velocity):
        self.position = position
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity

import numpy as np

class PhysicsData:
    def __init__(self, position, linear_velocity, angular_velocity):
        self.position = np.array(position)
        self.linear_velocity = np.array(linear_velocity)
        self.angular_velocity = np.array(angular_velocity)

class CarData:
    def __init__(self, position, rotation, linear_velocity, angular_velocity):
        self.position = np.array(position)
        self.rotation = np.array(rotation)  # rotation as (pitch, yaw, roll)
        self.linear_velocity = np.array(linear_velocity)
        self.angular_velocity = np.array(angular_velocity)

    def rotation_mtx(self):
        # Implement your rotation matrix conversion here
        pitch, yaw, roll = self.rotation
        # This is a placeholder, replace with real matrix conversion
        return np.identity(3)  # 3x3 identity matrix as dummy rotation

class PlayerData:
    def __init__(self, car_id, team_num, car_data, inverted_car_data,
                 is_demoed, on_ground, ball_touched, has_flip, boost_amount):
        self.car_id = car_id
        self.team_num = team_num
        self.car_data = car_data
        self.inverted_car_data = inverted_car_data
        self.is_demoed = is_demoed
        self.on_ground = on_ground
        self.ball_touched = ball_touched
        self.has_flip = has_flip
        self.boost_amount = boost_amount

class RLState:
    def __init__(self, blue_score, orange_score, boost_pads,
                 ball, inverted_ball, cars, tick_count):
        self.blue_score = blue_score
        self.orange_score = orange_score
        self.boost_pads = np.array(boost_pads)
        self.ball = ball
        self.inverted_ball = inverted_ball
        self.cars = cars
        self.tick_count = tick_count

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

class GameState:
    def __init__(self):
        self.players = []
        self.ball = None
        self.boost_pads = []
        self.blue_score = 0
        self.orange_score = 0
        self.tick_count = 0
        
    def decode(self, packet, ticks_elapsed):
        """Update state from packet"""
        self.tick_count += ticks_elapsed
        
        # Update ball
        ball_physics = PhysicsData(
            position=(
                packet.game_ball.physics.location.x,
                packet.game_ball.physics.location.y,
                packet.game_ball.physics.location.z
            ),
            linear_velocity=(
                packet.game_ball.physics.velocity.x,
                packet.game_ball.physics.velocity.y,
                packet.game_ball.physics.velocity.z
            ),
            angular_velocity=(
                packet.game_ball.physics.angular_velocity.x,
                packet.game_ball.physics.angular_velocity.y,
                packet.game_ball.physics.angular_velocity.z
            )
        )
        self.ball = ball_physics
        
        # Update players
        self.players = []
        for i in range(packet.num_cars):
            car = packet.game_cars[i]
            
            car_data = CarData(
                position=(
                    car.physics.location.x,
                    car.physics.location.y,
                    car.physics.location.z
                ),
                rotation=(
                    car.physics.rotation.pitch,
                    car.physics.rotation.yaw,
                    car.physics.rotation.roll
                ),
                linear_velocity=(
                    car.physics.velocity.x,
                    car.physics.velocity.y,
                    car.physics.velocity.z
                ),
                angular_velocity=(
                    car.physics.angular_velocity.x,
                    car.physics.angular_velocity.y,
                    car.physics.angular_velocity.z
                )
            )
            
            player = PlayerData(
                car_id=i,
                team_num=car.team,
                car_data=car_data,
                inverted_car_data=None,  # We'll skip this for simplicity
                is_demoed=car.is_demolished,
                on_ground=car.has_wheel_contact,
                ball_touched=packet.game_ball.latest_touch.player_index == i and packet.game_ball.latest_touch.time_seconds > 0,
                has_flip=not car.double_jumped,
                boost_amount=car.boost
            )
            
            self.players.append(player)
            
        # Update scores and boost pads
        self.blue_score = packet.teams[0].score
        self.orange_score = packet.teams[1].score
        self.boost_pads = [pad.is_active for pad in packet.game_boosts]


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

def get_latest_model_path(models_dir):
    # Find all .zip files in the directory
    model_files = glob.glob(os.path.join(models_dir, "*.zip"))
    if not model_files:
        return None
    # Sort by modification time, descending
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def rotation_to_euler(rotation):
    """Convert rotator struct (pitch, yaw, roll) to euler angles in radians."""
    return np.array([rotation.pitch, rotation.yaw, rotation.roll])

def build_obs_from_packet(packet: GameTickPacket, player_index: int) -> np.ndarray:
    """
    Extracts a 131-feature observation vector from GameTickPacket.
    """
    # Your car
    my_car = packet.game_cars[player_index]
    car_pos = np.array([
        my_car.physics.location.x,
        my_car.physics.location.y,
        my_car.physics.location.z
    ]) / 2300
    car_vel = np.array([
        my_car.physics.velocity.x,
        my_car.physics.velocity.y,
        my_car.physics.velocity.z
    ]) / 2300
    car_ang_vel = np.array([
        my_car.physics.angular_velocity.x,
        my_car.physics.angular_velocity.y,
        my_car.physics.angular_velocity.z
    ]) / 5.5
    car_rot = rotation_to_euler(my_car.physics.rotation) / np.pi
    car_boost = np.array([my_car.boost / 100])

    # Ball
    ball = packet.game_ball
    ball_pos = np.array([
        ball.physics.location.x,
        ball.physics.location.y,
        ball.physics.location.z
    ]) / 2300
    ball_vel = np.array([
        ball.physics.velocity.x,
        ball.physics.velocity.y,
        ball.physics.velocity.z
    ]) / 2300
    ball_ang_vel = np.array([
        ball.physics.angular_velocity.x,
        ball.physics.angular_velocity.y,
        ball.physics.angular_velocity.z
    ]) / 5.5

    # Other players (max 5 others)
    others = []
    for i, other_car in enumerate(packet.game_cars):
        if i == player_index:
            continue
        other_pos = np.array([
            other_car.physics.location.x,
            other_car.physics.location.y,
            other_car.physics.location.z
        ]) / 2300
        other_vel = np.array([
            other_car.physics.velocity.x,
            other_car.physics.velocity.y,
            other_car.physics.velocity.z
        ]) / 2300
        other_boost = np.array([other_car.boost / 100])
        others.extend(other_pos)
        others.extend(other_vel)
        others.extend(other_boost)

    # Pad if less than 5 others
    while len(others) < 5 * 7:
        others.extend([0] * 7)

    # Boost pads (always 34 pads)
    pads = []
    for i in range(34):  # fixed count
        pad = packet.game_boosts[i]
        pads.append(float(pad.is_active))

    obs = np.concatenate([
        car_pos, car_vel, car_ang_vel, car_rot, car_boost,
        ball_pos, ball_vel, ball_ang_vel,
        np.array(others),
        np.array(pads)
    ])

    return obs

class DictToObj:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    v = DictToObj(v)
                setattr(self, k, v)

class MyBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.model = None
        self.lookup_table = make_lookup_table()
        self.kickoff_index = -1  # -1 means not on kickoff, -2 means not kickoff taker
        self.hardcoded_kickoffs = True
        self.prev_time = 0.0
        self.ticks = 0
        self.game_state = GameState()
        self.update_action = True
        self.controls = SimpleControllerState()

    def initialize_agent(self):
        print(f"Initializing agent {self.name} (index {self.index})")
        models_dir = r"C:\Users\lexiv\AppData\Local\RLBotGUIX\MyBots\KaosX9\src\saved_models"
        model_path = get_latest_model_path(models_dir)
        if model_path and os.path.exists(model_path):
            self.model = PPO.load(
                model_path,
                custom_objects={
                    "clip_range": lambda _: 0.2,       # Use the value you originally trained with (default is 0.2)
                    "lr_schedule": lambda _: 3e-4      # Default learning rate (0.0003), change if you used different value
                })
            print(f"Model loaded from {model_path} \n\n")
        else:
            print(f"ERROR: Model not found at {model_path} \n\n")

    def maybe_do_kickoff(self, packet, ticks_elapsed):
        if packet.game_info.is_kickoff_pause:
            if self.kickoff_index >= 0:
                self.kickoff_index += round(ticks_elapsed)
            elif self.kickoff_index == -1:
                is_kickoff_taker = False
                ball_pos = np.array([packet.game_ball.physics.location.x, packet.game_ball.physics.location.y])
                positions = np.array([[car.physics.location.x, car.physics.location.y]
                                      for car in packet.game_cars[:packet.num_cars]])
                distances = np.linalg.norm(positions - ball_pos, axis=1)
                if abs(distances.min() - distances[self.index]) <= 10:
                    is_kickoff_taker = True
                    indices = np.argsort(distances)
                    for index in indices:
                        if abs(distances[index] - distances[self.index]) <= 10 \
                                and packet.game_cars[index].team == self.team \
                                and index != self.index:
                            # print("Potential collision with", index)
                            if self.team == 0:
                                is_left = positions[index, 0] < positions[self.index, 0]
                            else:
                                is_left = positions[index, 0] > positions[self.index, 0]
                            if not is_left:
                                is_kickoff_taker = False  # Left goes

                self.kickoff_index = 0 if is_kickoff_taker else -2

            if 0 <= self.kickoff_index < len(KICKOFF_NUMPY) \
                    and packet.game_ball.physics.location.y == 0:
                action = KICKOFF_NUMPY[self.kickoff_index]
                # self.action = action  # If needed in your structure
                self.update_controls(action)  # Call to apply kickoff controls
            else:
                self.kickoff_index = -1

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

    def convert_packet_to_state(self, packet: GameTickPacket) -> RLState:
        # === Ball physics ===
        ball_physics = PhysicsData(
            position=(
                packet.game_ball.physics.location.x,
                packet.game_ball.physics.location.y,
                packet.game_ball.physics.location.z
            ),
            linear_velocity=(
                packet.game_ball.physics.velocity.x,
                packet.game_ball.physics.velocity.y,
                packet.game_ball.physics.velocity.z
            ),
            angular_velocity=(
                packet.game_ball.physics.angular_velocity.x,
                packet.game_ball.physics.angular_velocity.y,
                packet.game_ball.physics.angular_velocity.z
            )
        )

        # For inverted_ball, usually mirrored version on Y axis
        inverted_ball_physics = PhysicsData(
            position=(
                packet.game_ball.physics.location.x,
                -packet.game_ball.physics.location.y,
                packet.game_ball.physics.location.z
            ),
            linear_velocity=(
                packet.game_ball.physics.velocity.x,
                -packet.game_ball.physics.velocity.y,
                packet.game_ball.physics.velocity.z
            ),
            angular_velocity=(
                packet.game_ball.physics.angular_velocity.x,
                -packet.game_ball.physics.angular_velocity.y,
                packet.game_ball.physics.angular_velocity.z
            )
        )

        # === Players (cars) ===
        cars = {}
        for i in range(packet.num_cars):
            car = packet.game_cars[i]
    
            # Build a minimal Physics sub-object for this car
            class Physics:
                pass
            physics = Physics()
            physics.position = (
                car.physics.location.x,
                car.physics.location.y,
                car.physics.location.z
            )
            physics.linear_velocity = (
                car.physics.velocity.x,
                car.physics.velocity.y,
                car.physics.velocity.z
            )
            physics.angular_velocity = (
                car.physics.angular_velocity.x,
                car.physics.angular_velocity.y,
                car.physics.angular_velocity.z
            )
    
            # Build the car object as expected by compat/encode_gamestate
            class Car:
                pass
            car_obj = Car()
            car_obj.physics = physics
            car_obj.team_num = car.team
            car_obj.is_demoed = car.is_demolished
            car_obj.on_ground = car.has_wheel_contact
            car_obj.ball_touches = (packet.game_ball.latest_touch.player_index == i and packet.game_ball.latest_touch.time_seconds > 0)
            car_obj.has_flip = not car.double_jumped
            car_obj.boost_amount = car.boost
    
            cars[i] = car_obj

        # === Boost pads ===
        boost_pads = [pad.is_active for pad in packet.game_boosts]

        # === Scores ===
        blue_score = packet.teams[0].score
        orange_score = packet.teams[1].score

        tick_count = getattr(packet.game_info, "frame_num", 0)

        return RLState(
            blue_score=blue_score,
            orange_score=orange_score,
            boost_pads=boost_pads,
            ball=ball_physics,
            inverted_ball=inverted_ball_physics,
            cars=cars,
            tick_count=tick_count  
        )

    def prepare_state_for_inference(self, game_state):
        """Convert GameState to format expected by encode_gamestate"""
        # Create cars dict from players list
        cars = {}
        for player in game_state.players:
            # Create physics object
            class Physics:
                pass
            physics = Physics()
            physics.position = player.car_data.position
            physics.linear_velocity = player.car_data.linear_velocity
            physics.angular_velocity = player.car_data.angular_velocity
        
            # Create car object
            class Car:
                pass
            car_obj = Car()
            car_obj.physics = physics
            car_obj.team_num = player.team_num
            car_obj.is_demoed = player.is_demoed
            car_obj.on_ground = player.on_ground
            car_obj.has_flip = player.has_flip
            car_obj.boost_amount = player.boost_amount
        
            cars[player.car_id] = car_obj
    
        # Create inverted ball
        inverted_ball = PhysicsData(
            position=(
                game_state.ball.position[0],
                -game_state.ball.position[1],
                game_state.ball.position[2]
            ),
            linear_velocity=(
                game_state.ball.linear_velocity[0],
                -game_state.ball.linear_velocity[1],
                game_state.ball.linear_velocity[2]
            ),
            angular_velocity=(
                game_state.ball.angular_velocity[0],
                -game_state.ball.angular_velocity[1],
                game_state.ball.angular_velocity[2]
            )
        )
    
        # Create RLState
        return RLState(
            blue_score=game_state.blue_score,
            orange_score=game_state.orange_score,
            boost_pads=game_state.boost_pads,
            ball=game_state.ball,
            inverted_ball=inverted_ball,
            cars=cars,
            tick_count=game_state.tick_count
        )

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        #print(f"My index: {self.index}, num_cars: {packet.num_cars}")
        #for i, car in enumerate(packet.game_cars):
        #    print(f"Car {i}: team={car.team}, name={car.name}")
        
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = round(delta * 120)
        self.ticks += ticks_elapsed

        # Update game state
        self.game_state.decode(packet, ticks_elapsed)

        # Check if our player exists in the game state
        if len(self.game_state.players) <= self.index:
            return SimpleControllerState()  # Not spawned yet
        
        if not packet.game_info.is_round_active:
            return SimpleControllerState()  # Game paused or between rounds

        # Get our player
        player = self.game_state.players[self.index]
    
        # Convert game state to format expected by encode_gamestate
        state = self.prepare_state_for_inference(self.game_state)

        # Get model prediction
        if self.model is not None:
            obs = encode_gamestate(state)
            action_idx, _ = self.model.predict(obs, deterministic=True)
            action = self.lookup_table[action_idx]
            self.update_controls(action)

        # handle kickoff logic
        #if self.hardcoded_kickoffs:
        #    self.maybe_do_kickoff(packet, ticks_elapsed)
        #           
        if self.model is None:
            return SimpleControllerState()  # Model not loaded yet

        try:
            # Convert packet to custom state format
            state = self.convert_packet_to_state(packet)

            # Encode observation (must match training dimensions)
            obs = encode_gamestate(state)  # shape (454,)

            # Predict discrete action index
            action_idx, _ = self.model.predict(obs, deterministic=True)

            # Map action index to action vector
            action_vector = self.lookup_table[action_idx]

            # Build RLBot control structure
            controls = SimpleControllerState(
                throttle=float(action_vector[0]),
                steer=float(action_vector[1]),
                pitch=float(action_vector[2]),
                yaw=float(action_vector[3]),
                roll=float(action_vector[4]),
                jump=bool(action_vector[5]),
                boost=bool(action_vector[6]),
                handbrake=bool(action_vector[7])
            )
            return controls

        except Exception as e:
            print(f"Error in get_output: {e}")
            import traceback
            traceback.print_exc()
            return SimpleControllerState()


    def build_compat_state_from_packet(self, packet):
        """Convert RLBot GameTickPacket to CompatGameState for observation encoding"""
        my_car = packet.game_cars[self.index]
        my_team = my_car.team
        
        # Ball data
        ball = PhysicsObject(
            position=(packet.game_ball.physics.location.x, 
                     packet.game_ball.physics.location.y, 
                     packet.game_ball.physics.location.z),
            linear_velocity=(packet.game_ball.physics.velocity.x, 
                            packet.game_ball.physics.velocity.y, 
                            packet.game_ball.physics.velocity.z),
            angular_velocity=(packet.game_ball.physics.angular_velocity.x, 
                             packet.game_ball.physics.angular_velocity.y, 
                             packet.game_ball.physics.angular_velocity.z)
        )
        
        # Inverted ball (mirror X and Y for opposite team perspective)
        inverted_ball = PhysicsObject(
            position=(-ball.position[0], -ball.position[1], ball.position[2]),
            linear_velocity=(-ball.linear_velocity[0], -ball.linear_velocity[1], ball.linear_velocity[2]),
            angular_velocity=ball.angular_velocity
        )
        
        # Players data
        players = []
        for i in range(packet.num_cars):
            car = packet.game_cars[i]
            
            player = PlayerData()
            player.team_num = car.team
            
            # Car physics data
            player.car_data = PhysicsObject(
                position=(car.physics.location.x, car.physics.location.y, car.physics.location.z),
                linear_velocity=(car.physics.velocity.x, car.physics.velocity.y, car.physics.velocity.z),
                angular_velocity=(car.physics.angular_velocity.x, car.physics.angular_velocity.y, car.physics.angular_velocity.z)
            )
            
            # Inverted car data (mirror X and Y)
            player.inverted_car_data = PhysicsObject(
                position=(-player.car_data.position[0], -player.car_data.position[1], player.car_data.position[2]),
                linear_velocity=(-player.car_data.linear_velocity[0], -player.car_data.linear_velocity[1], player.car_data.linear_velocity[2]),
                angular_velocity=player.car_data.angular_velocity
            )
            
            # Car state
            player.is_demoed = car.is_demolished
            player.on_ground = car.has_wheel_contact
            player.ball_touched = False  # Not directly available in packet
            player.has_flip = not car.double_jumped
            player.boost_amount = car.boost / 100.0  # Convert to 0-1 range
            
            players.append(player)
        
        # Scores
        blue_score = packet.teams[0].score
        orange_score = packet.teams[1].score
        
        # Boost pads (simplified - set all as active)
        boost_pads = np.ones(34, dtype=bool)
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

    def update_controls(self, action):
        """
        Applies the action vector to the SimpleControllerState.
        """
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0  # Assuming 0/1 for jump
        self.controls.boost = action[6] > 0  # Assuming 0/1 for boost
        self.controls.handbrake = action[7] > 0  # Assuming 0/1 for handbrake
