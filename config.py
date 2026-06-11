import os
import json

CONFIG_FILE = "config.json"

def load_config():
    default_config = {
        "carla_path": "F:\\pluginFiles\\simulation\\CARLA_0.9.16",
        "grid_enabled": True,
        "scanlines_enabled": True
    }
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w") as f:
            json.dump(default_config, f, indent=4)
        return default_config
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return default_config

def save_config(config_data):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")
