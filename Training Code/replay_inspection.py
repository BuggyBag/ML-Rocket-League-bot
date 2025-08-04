import json
import os

# ---- Config ----
file_path = r"C:\Users\lexiv\AppData\Local\RLBotGUIX\MyBots\KaosX9\BotTraining\parsed\00e658c8-78f7-4abe-b5f6-45eee99139c2_RAW.json"

# ---- Utility: Recursively search for values matching a target ----
def recursive_value_search(data, target_value, path="root"):
    matches = []
    if isinstance(data, dict):
        for k, v in data.items():
            if v == target_value:
                matches.append(f"{path}.{k}")
            else:
                matches.extend(recursive_value_search(v, target_value, f"{path}.{k}"))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            matches.extend(recursive_value_search(item, target_value, f"{path}[{i}]"))
    return matches

# ---- Load the JSON replay ----
if not os.path.exists(file_path):
    raise FileNotFoundError("Replay file not found!")

with open(file_path, "r") as f:
    replay_data = json.load(f)

# ---- Load all object entries ----
objects = replay_data.get("objects", [])
print(f"Total object entries found: {len(objects)}")

# ---- Preview the first few objects (optional) ----
print("\nPreview of first 10 objects:\n")
for i, obj in enumerate(objects[:10]):
    print(f"Index {i}: {repr(obj)}")

# ---- Find key matches ----
print("\nSearching top-level keys and nested dictionaries for object names...\n")

def find_key_matches(data, object_strings):
    matched = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if key in object_strings:
                matched[key] = value
            elif isinstance(value, (dict, list)):
                sub_matches = find_key_matches(value, object_strings)
                if sub_matches:
                    matched[key] = sub_matches
    elif isinstance(data, list):
        for item in data:
            sub_matches = find_key_matches(item, object_strings)
            if sub_matches:
                matched.update(sub_matches)
    return matched

object_strings = [str(obj) for obj in objects]
key_matches = find_key_matches(replay_data, object_strings)

if key_matches:
    print("Found matches for object keys:\n")
    print(json.dumps(key_matches, indent=4))
else:
    print("No object strings found as keys in the structure.\n")

# ---- Recursively search for object values ----
print("\nRecursively searching entire JSON for object value matches...\n")

for obj in object_strings:
    matches = recursive_value_search(replay_data, obj)
    if matches:
        print(f"Value '{obj}' found at:")
        for match in matches:
            print(f" {match}")
    else:
        print(f"Value '{obj}' not found anywhere else in the JSON.")

# ---- Map 'class' index to object name ----
def map_class_indices_to_names(json_data, objects):
    print("\nMapping numeric class indices to object names...\n")
    resolved = []

    def recursive_search_for_class(data, path="root"):
        if isinstance(data, dict):
            for k, v in data.items():
                if k == "class" and isinstance(v, int) and 0 <= v < len(objects):
                    readable = objects[v]
                    resolved.append((path + "." + k, v, readable))
                else:
                    recursive_search_for_class(v, path + "." + k)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                recursive_search_for_class(item, f"{path}[{i}]")

    recursive_search_for_class(json_data)

    if resolved:
        for path, idx, name in resolved:
            print(f"{path}: index {idx} ? '{name}'")
    else:
        print("No 'class' indices found.")

map_class_indices_to_names(replay_data, objects)

# ---- Search all integer references and print actual value ----
def find_object_index_references_with_values(json_data, objects):
    print("\nSearching for integer references that match indices in 'objects' and printing their values...\n")
    matches = []

    def recursive_search(data, path="root"):
        if isinstance(data, dict):
            for k, v in data.items():
                recursive_search(v, f"{path}.{k}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                recursive_search(item, f"{path}[{i}]")
        elif isinstance(data, int):
            if 0 <= data < len(objects):
                matches.append((path, data, objects[data], data))

    recursive_search(json_data)

    if matches:
        for path, idx, obj_name, value in matches:
            print(f"{path}: index {idx} ? '{obj_name}' ? Value: {value}")
    else:
        print("No matching object index references found.")

find_object_index_references_with_values(replay_data, objects)
