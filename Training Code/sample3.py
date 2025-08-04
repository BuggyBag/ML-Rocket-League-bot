import os
import json

parsed_replays_folder = "./parsed"
output_folder = "./extracted_data"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(parsed_replays_folder):
    if not filename.endswith(".json"):
        continue

    with open(os.path.join(parsed_replays_folder, filename), "r") as f:
        data = json.load(f)

    frames = data.get("network_frames", {}).get("frames", [])
    if not frames:
        print(f"No frames found in {filename}")
        continue

    # Identify ball and car actor_ids from first frame's new_actors
    first_frame = frames[0]
    ball_actor_ids = set()
    car_actor_ids = set()

    for actor in first_frame.get("new_actors", []):
        actor_id = actor.get("actor_id")
        object_id = actor.get("object_id")
        # Adjust these object_ids based on your data; example ball object_ids:
        if object_id in [114, 120]:  # Example ball object_ids
            ball_actor_ids.add(actor_id)
        else:
            car_actor_ids.add(actor_id)

    print(f"Detected ball actor_ids: {ball_actor_ids}")
    print(f"Detected car actor_ids: {car_actor_ids}")

    extracted_frames = []

    for frame in frames:
        timestamp = frame.get("time", 0.0)
        frame_info = {"timestamp": timestamp, "cars": [], "ball": None}

        for actor in frame.get("updated_actors", []):
            attr = actor.get("attribute", {})
            rb = attr.get("RigidBody", None)
            if not isinstance(rb, dict):
                print(f"Skipping actor_id {actor.get('actor_id')} - RigidBody: {rb}")
                continue  # Skip if RigidBody is missing or not a dict

            actor_id = actor.get("actor_id")
            info = {
                "id": actor_id,
                "pos": [
                    rb.get("location", {}).get("x", 0.0),
                    rb.get("location", {}).get("y", 0.0),
                    rb.get("location", {}).get("z", 0.0)
                ],
                "vel": [
                    rb.get("linear_velocity", {}).get("x", 0.0),
                    rb.get("linear_velocity", {}).get("y", 0.0),
                    rb.get("linear_velocity", {}).get("z", 0.0)
                ],
                "ang_vel": [
                    rb.get("angular_velocity", {}).get("x", 0.0),
                    rb.get("angular_velocity", {}).get("y", 0.0),
                    rb.get("angular_velocity", {}).get("z", 0.0)
                ],
                "rot": [
                    rb.get("rotation", {}).get("x", 0.0),
                    rb.get("rotation", {}).get("y", 0.0),
                    rb.get("rotation", {}).get("z", 0.0),
                    rb.get("rotation", {}).get("w", 0.0)
                ]
            }

            if actor_id in ball_actor_ids:
                frame_info["ball"] = info
            elif actor_id in car_actor_ids:
                frame_info["cars"].append(info)


        extracted_frames.append(frame_info)

    output_path = os.path.join(output_folder, filename)
    with open(output_path, "w") as out_f:
        json.dump(extracted_frames, out_f, indent=2)

    print(f"Extracted {len(extracted_frames)} frames from {filename} -> {output_path}")
