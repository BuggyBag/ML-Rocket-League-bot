
# RLBot PPO Bot Setup Guide

This README explains how to use a bot script (`bot.py`) integrated with the RLBot framework, which loads PPO-trained models from a specified directory. The training script (`_PPO_self_play.py`) is also provided and depends on a set of packages listed in `requirements.txt`.

## Folder Structure

```
project_root/
│
├── bot.py                      # The bot script for RLBot
├── _PPO_self_play.py          # PPO training script
├── requirements.txt           # Environment dependencies for PPO script
├── models/
│   └── PPO_model.zip          # Trained PPO model(s)
```

## Prerequisites

- Python 3.7–3.10
- Rocket League installed (Steam preferred)
- Virtual environment (recommended)
- RLBot installed: `pip install rlbot`
- PPO dependencies (Stable-Baselines3, Gym, etc.)

## Step-by-Step Usage

### 1. Set Up Virtual Environment

```bash
python -m venv rlbot-env
source rlbot-env/bin/activate         # On Linux/macOS
rlbot-env\Scripts\activate          # On Windows
```

### 2. Install RLBot

```bash
pip install rlbot
```

### 3. Install PPO Script Dependencies

Install the required libraries for training/inference with PPO:

```bash
pip install -r requirements.txt
```

> ⚠️ Make sure `requirements.txt` includes packages like `stable-baselines3`, `gym`, `torch`, etc.

### 4. Load Trained PPO Model in `bot.py`

Ensure that `bot.py` includes logic similar to:

```python
from stable_baselines3 import PPO

# Load the model
model = PPO.load("models/PPO_model.zip")

# Use `model.predict()` during the game loop
```

Make sure the relative or absolute path to `PPO_model.zip` is correct.

### 5. Running the Bot

To launch the bot with the RLBot framework:

```bash
python bot.py
```

Ensure Rocket League is running before launching the bot.

### 6. (Optional) Training with `_PPO_self_play.py`

You can re-train or fine-tune the model using:

```bash
python _PPO_self_play.py
```

Make sure this script saves models into the `models/` directory or updates the path accordingly in `bot.py`.

## Notes

- Always check that the model input/output shapes match what the environment expects.
- PPO models are usually zipped (`.zip`) with both weights and hyperparameters.
- If Rocket League updates break compatibility, check the RLBot Discord or GitHub for patches.

## References

- [RLBot Docs](https://www.rlbot.org)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Docs](https://gymnasium.farama.org/)
