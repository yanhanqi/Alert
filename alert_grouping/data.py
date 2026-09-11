"""Load label-free AIT-ADS sequences within scene-local ten-minute windows."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ("short", "host")


def load_split(config_path, split):
    path = Path(config_path).resolve()
    config = json.loads(path.read_text())
    if set(config["train"]) & set(config["test"]):
        raise ValueError("Training and test scenarios overlap")
    root = path.parent.parent
    scenes = {}
    for name in config[split]:
        file = root / config["data_dir"] / config["file_pattern"].format(scenario=name)
        frame = pd.read_csv(file, usecols=["time", *FEATURES], keep_default_na=False)
        frame["time"] = pd.to_numeric(frame["time"], errors="raise")
        if not np.isfinite(frame["time"]).all():
            raise ValueError(f"Invalid timestamp: {file}")
        frame = frame.sort_values("time", kind="stable")
        scenes[name] = {f: frame[f].astype(str).to_numpy() for f in FEATURES}
        scenes[name]["raw_time"] = frame["time"].to_numpy(dtype=np.float64)
    return scenes


def sequence_slices(scene, context_size=256, window_seconds=600):
    times = scene["raw_time"]
    buckets = np.floor(times / window_seconds)
    boundaries = np.r_[0, np.flatnonzero(np.diff(buckets)) + 1, len(times)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        for start in range(left, right, context_size):
            yield slice(start, min(start + context_size, right))


def get_sequence(scene, selection):
    return {key: values[selection] for key, values in scene.items()}
