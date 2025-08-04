import json

# Load the replay JSON
file_path = './parsed/00e658c8-78f7-4abe-b5f6-45eee99139c2_RAW.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# Output file
output_path = './output.txt'
output_lines = []

# Object and stream mappings
object_map = {str(i): name for i, name in enumerate(data.get("objects", []))}
stream_map = {str(i): name for i, name in enumerate(data.get("streams", []))}

# Human-readable aliases
alias_streams = {
    "TAGame.Car_TA:ReplicatedThrottle": "ReplicatedThrottle",
    "TAGame.Car_TA:ReplicatedSteer": "ReplicatedSteer",
    "TAGame.Car_TA:ReplicatedBoost": "ReplicatedBoost",
    "TAGame.Car_TA:ReplicatedLocation": "ReplicatedLocation",
    "TAGame.Car_TA:ReplicatedRotation": "ReplicatedRotation",
    "TAGame.Ball_TA:ReplicatedLocation": "ReplicatedLocation",
    "TAGame.Ball_TA:ReplicatedRotation": "ReplicatedRotation",
}

actor_object_lookup = {}

frames = data.get('network_frames', {}).get('frames', [])

for frame_idx, frame in enumerate(frames):
    output_lines.append(f"--- Frame {frame_idx + 1} | Time: {frame.get('time', 'N/A')} ---")

    # Handle new_actors
    for actor in frame.get("new_actors", []):
        actor_id = actor.get("actor_id")
        object_id = actor.get("object_id")
        actor_object_lookup[actor_id] = object_id

        obj_name = object_map.get(str(object_id), f"Object_{object_id}")
        output_lines.append(f"Object: {obj_name}")

        traj = actor.get("initial_trajectory", {})
        location = traj.get("location")
        rotation = traj.get("rotation")

        if location:
            output_lines.append(f"    ReplicatedLocation: {json.dumps(location)}")
        else:
            output_lines.append(f"    ReplicatedLocation: N/A")

        if rotation:
            output_lines.append(f"    ReplicatedRotation: {json.dumps(rotation)}")
        else:
            output_lines.append(f"    ReplicatedRotation: N/A")

    # Handle updated_actors
    for update in frame.get("updated_actors", []):
        actor_id = update.get("actor_id")
        stream_updates = update.get("streams", {})

        object_id = actor_object_lookup.get(actor_id)
        if object_id is None:
            continue

        obj_name = object_map.get(str(object_id), f"Object_{object_id}")
        output_lines.append(f"Object: {obj_name}")

        for stream_id, value in stream_updates.items():
            stream_name = stream_map.get(str(stream_id), f"Stream_{stream_id}")
            readable = alias_streams.get(stream_name, stream_name)

            if isinstance(value, dict):
                output_lines.append(f"    {readable}: {json.dumps(value)}")
            else:
                output_lines.append(f"    {readable}: {value}")

    output_lines.append("\n---")

# Write to file
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"Output saved to {output_path}")
