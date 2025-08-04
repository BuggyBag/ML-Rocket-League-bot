import os
import json
import logging
import traceback
import subprocess

# ----------- CONFIGURATION -----------

RRROCKET_PATH = r".\rrrocket\target\release\rrrocket.exe"  # Path to rrrocket binary
DOWNLOAD_DIR = "./replays"   # Input folder with .replay files
PARSED_DIR = "./parsed"      # Output folder for parsed JSON files
os.makedirs(PARSED_DIR, exist_ok=True)

# ----------- LOGGING -----------

logging.basicConfig(
    filename="replay_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------- HELPER FUNCTIONS -----------

def collect_all_tags(obj, found=None):
    if found is None:
        found = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            k_str = str(k)
            if "TAGame." in k_str or "Engine." in k_str or ":" in k_str:
                if k_str not in found:
                    found[k_str] = []
                found[k_str].append(v)
            if isinstance(v, (dict, list)):
                collect_all_tags(v, found)
    elif isinstance(obj, list):
        for item in obj:
            collect_all_tags(item, found)
    return found

# ----------- MAIN EXTRACTOR -----------

def extract_replay_data(replay_path):
    try:
        result = subprocess.run(
            [RRROCKET_PATH, "--network-parse", "--pretty", replay_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        output = result.stdout if result.stdout.strip() else result.stderr
        if not output.strip():
            logging.error(f"No output from rrrocket for {replay_path}")
            return False

        replay_data = json.loads(output)

        base_name = os.path.basename(replay_path).replace(".replay", "")
        raw_output_path = os.path.join(PARSED_DIR, f"{base_name}_RAW.json")
        with open(raw_output_path, "w") as f:
            f.write(output)
        logging.info(f"Saved raw rrrocket output to: {raw_output_path}")

        # Collect tags just for logging
        tag_data = collect_all_tags(replay_data)
        logging.info(f"{replay_path}: Found {len(tag_data)} unique tags")

        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"rrrocket failed: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Error processing {replay_path}: {e}")
        logging.error(traceback.format_exc())
        print(f"[ERROR] {replay_path}: {e}")
        return False

# ----------- PARSE ALL REPLAYS -----------

def process_replays():
    for replay_file in os.listdir(DOWNLOAD_DIR):
        if not replay_file.endswith(".replay"):
            continue

        replay_path = os.path.join(DOWNLOAD_DIR, replay_file)
        print(f"Processing: {replay_file}")

        success = extract_replay_data(replay_path)
        if success:
            print(f"Processed correctly: {replay_file}")
        else:
            print(f"Error processing: {replay_file}")

# ----------- ENTRY POINT -----------

if __name__ == "__main__":
    process_replays()
