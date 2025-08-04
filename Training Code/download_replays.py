import requests
import os
import time

API_KEY = "ODZud1q1Mjad75bYhi2KB2jQXDrgv2dp8BM0w56D"
DOWNLOAD_DIR = "./replays"
MIN_DATE = "2023-01-01T00:00:00"

RANKED_PLAYLISTS = [
    "ranked-duels", "ranked-doubles", "ranked-standard", "ranked-solo-standard"
]

MIN_TIER = 22 # SSL
MAX_TIER = 22  # Supersonic Legend
REPLAYS_PER_REQUEST = 200
NUM_REQUESTS_PER_COMBO = 5


def download_replays():
    headers = {"Authorization": API_KEY}

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for playlist in RANKED_PLAYLISTS:
        for tier in range(MIN_TIER, MAX_TIER + 1):
            total_downloaded = 0
            print(f"\n Playlist: {playlist} | Tier: {tier}")

            for request_idx in range(NUM_REQUESTS_PER_COMBO):
                params = {
                    "playlist": playlist,
                    "min-tier": tier,
                    "max-tier": tier,
                    "min-date": MIN_DATE,
                    "sort-by": "replay-date",
                    "count": REPLAYS_PER_REQUEST,
                    "page": request_idx + 1,
                }

                print(f"\nRequest {request_idx + 1}/5")

                try:
                    response = requests.get("https://ballchasing.com/api/replays", headers=headers, params=params)
                    response.raise_for_status()
                except requests.RequestException as e:
                    print(f"Error during request: {e}")
                    break

                replays = response.json().get("list", [])
                if not replays:
                    print("No replays found for this query.")
                    break

                for replay in replays:
                    replay_id = replay["id"]
                    replay_path = os.path.join(DOWNLOAD_DIR, f"{replay_id}.replay")

                    if os.path.exists(replay_path):
                        print(f"Already downloaded: {replay_id}")
                        continue

                    replay_url = f"https://ballchasing.com/api/replays/{replay_id}/file"
                    try:
                        file_response = requests.get(replay_url, headers=headers)
                        file_response.raise_for_status()

                        with open(replay_path, "wb") as f:
                            f.write(file_response.content)

                        total_downloaded += 1
                        print(f"Downloaded: {replay_id}")

                        time.sleep(0.1)  # Be polite to the API
                    except requests.RequestException as e:
                        print(f"Failed to download replay {replay_id}: {e}")

                print(f"Total downloaded so far for tier {tier}: {total_downloaded}")

    print("\nFinished downloading all replays.")


if __name__ == "__main__":
    download_replays()
