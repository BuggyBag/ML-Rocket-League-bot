import os
import json

# Paths
parsed_replays_folder = "./parsed"  # Folder where your parsed JSONs are
output_folder = "./extracted_data"  # Folder where you want to save extracted info

os.makedirs(output_folder, exist_ok=True)

# Helper to extract data safely
def safe_get(d, keys, default=0.0):
    for key in keys:
        d = d.get(key, {})
    return d if isinstance(d, (int, float)) else default

# Process each parsed replay
for filename in os.listdir(parsed_replays_folder):
    if not filename.endswith(".json"):
        continue

    with open(os.path.join(parsed_replays_folder, filename), "r") as f:
        parsed = json.load(f)

    #print(f"Parsed content type: {type(parsed)}")  # Check the type of parsed data
    #print(f"Parsed keys: {parsed.keys()}")  # Print the keys of the parsed data

    # Check the type of frames and print the first element to see its structure
    network_frames = parsed.get("network_frames", {})
    frames = network_frames.get("frames", [])
    #print(f"Type of frames: {type(frames)}")

    #if frames:
    #    print(f"First frame content: {frames[0]}")

    # Debugging lines: Check the type and structure of 'frames'
    print(f"Processing {filename}")
    #print(f"Type of 'frames': {type(frames)}")  # Check if frames is a list or dict
    #if isinstance(frames, list):
    #    print(f"First 2 frames: {frames[:2]}")  # Print the first two frames for inspection
    #else:
    #    print(f"'frames' is not a list. Here's its content: {frames}")

    actors = parsed.get("actors", {})
    objects = parsed.get("objects", {})

    extracted_frames = []

    # Build reverse maps for easier lookup
    id_to_object = {i: obj for i, obj in enumerate(objects)}
    id_to_actor = {int(k): v for k, v in actors.items()}

    # Identify ball actor
    ball_actor_ids = [
        aid for aid, data in id_to_actor.items()
        if data.get("TypeName", "").endswith("Ball_TA")
    ]

    # Identify car actors
    car_actor_ids = [
        aid for aid, data in id_to_actor.items()
        if data.get("TypeName", "").endswith("Car_TA")
    ]

    for frame in frames:
        #print(f"Type of frame: {type(frame)}")  # Debugging: Check the type of each frame
        #print(f"First frame content: {frame}")  # Debugging: Print the content of the first frame

        timestamp = frame.get("time", 0.0)
        actor_data = frame.get("updated_actors", {})

        frame_info = {
            "timestamp": timestamp,
            "cars": [],
            "ball": None
        }

        # Extract cars
        for actor_id in car_actor_ids:
            actor_update = actor_data.get(str(actor_id))
            if actor_update:
                car_info = {
                    "id": actor_id,
                    "pos": [
                        safe_get(actor_update, ["RigidBody", "Location", "x"]),
                        safe_get(actor_update, ["RigidBody", "Location", "y"]),
                        safe_get(actor_update, ["RigidBody", "Location", "z"]),
                    ],
                    "vel": [
                        safe_get(actor_update, ["RigidBody", "LinearVelocity", "x"]),
                        safe_get(actor_update, ["RigidBody", "LinearVelocity", "y"]),
                        safe_get(actor_update, ["RigidBody", "LinearVelocity", "z"]),
                    ],
                    "ang_vel": [
                        safe_get(actor_update, ["RigidBody", "AngularVelocity", "x"]),
                        safe_get(actor_update, ["RigidBody", "AngularVelocity", "y"]),
                        safe_get(actor_update, ["RigidBody", "AngularVelocity", "z"]),
                    ],
                    "rot": [
                        safe_get(actor_update, ["RigidBody", "Rotation", "pitch"]),
                        safe_get(actor_update, ["RigidBody", "Rotation", "yaw"]),
                        safe_get(actor_update, ["RigidBody", "Rotation", "roll"]),
                    ]
                }
                frame_info["cars"].append(car_info)

        # Extract ball
        for actor_id in ball_actor_ids:
            actor_update = actor_data.get(str(actor_id))
            if actor_update:
                ball_info = {
                    "id": actor_id,
                    "pos": [
                        safe_get(actor_update, ["RigidBody", "Location", "x"]),
                        safe_get(actor_update, ["RigidBody", "Location", "y"]),
                        safe_get(actor_update, ["RigidBody", "Location", "z"]),
                    ],
                    "vel": [
                        safe_get(actor_update, ["RigidBody", "LinearVelocity", "x"]),
                        safe_get(actor_update, ["RigidBody", "LinearVelocity", "y"]),
                        safe_get(actor_update, ["RigidBody", "LinearVelocity", "z"]),
                    ],
                    "ang_vel": [
                        safe_get(actor_update, ["RigidBody", "AngularVelocity", "x"]),
                        safe_get(actor_update, ["RigidBody", "AngularVelocity", "y"]),
                        safe_get(actor_update, ["RigidBody", "AngularVelocity", "z"]),
                    ],
                    "rot": [
                        safe_get(actor_update, ["RigidBody", "Rotation", "pitch"]),
                        safe_get(actor_update, ["RigidBody", "Rotation", "yaw"]),
                        safe_get(actor_update, ["RigidBody", "Rotation", "roll"]),
                    ]
                }
                frame_info["ball"] = ball_info
                break  # Only one ball expected

        extracted_frames.append(frame_info)

    # Save extracted frames
    output_path = os.path.join(output_folder, filename)
    with open(output_path, "w") as out_f:
        json.dump(extracted_frames, out_f, indent=2)

    print(f"Extracted {len(extracted_frames)} frames from {filename} -> {output_path}")
