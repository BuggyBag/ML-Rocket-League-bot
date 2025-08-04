import subprocess
import sys
import os

# Optional: Check current working directory
print(f"Running from: {os.getcwd()}")

# Define the scripts in the correct order
pipeline_scripts = [
    "download_replays.py",
    "parse_replays.py",
    "process_replays.py",
    "ppo_training.py"
]

def run_script(script_name):
    print(f"\n Running {script_name} ...")
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"Finished {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}")
        print(e)
        sys.exit(1)

def main():
    for script in pipeline_scripts:
        run_script(script)
    print("\n Pipeline complete!")

if __name__ == "__main__":
    main()
