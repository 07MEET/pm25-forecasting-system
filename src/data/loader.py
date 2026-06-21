import os
import pickle
from pathlib import Path

import numpy as np
import yaml


# Project root (works regardless of current working directory)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path=None):
    """
    Load YAML configuration file.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "params.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_month(month, raw_path, met_vars, emission_vars):
    """
    Load all available features for a single month.
    """
    path = os.path.join(raw_path, month)

    data = {
        "cpm25": np.load(
            os.path.join(path, "cpm25.npy")
        ).astype(np.float32)
    }

    for var in met_vars + emission_vars:
        file_path = os.path.join(path, f"{var}.npy")

        if os.path.exists(file_path):
            data[var] = np.load(file_path).astype(np.float32)

    return data


def load_all_months(config):
    """
    Load all months defined in params.yaml.
    """
    all_data = {}

    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    met_vars = config["features"]["met_vars"]
    emission_vars = config["features"]["emission_vars"]

    for month in config["months"]:
        print(f"Loading {month}...")
        all_data[month] = load_month(
            month=month,
            raw_path=raw_path,
            met_vars=met_vars,
            emission_vars=emission_vars,
        )

    return all_data


def compute_stats(key, all_data, log_features):
    """
    Compute mean/std statistics for normalization.
    """

    arrays = [
        data[key]
        for data in all_data.values()
        if key in data
    ]

    combined = np.concatenate(
        arrays,
        axis=0,
    ).astype(np.float32)

    if key in log_features:
        combined = np.log1p(combined * 1e9)

    return (
        float(combined.mean()),
        float(combined.std() + 1e-8),
    )


def compute_norm_stats(config, all_data):
    """
    Compute normalization statistics for every feature.
    """
    norm_stats = {}

    met_vars = config["features"]["met_vars"]
    emission_vars = config["features"]["emission_vars"]
    log_features = set(config["features"]["log_features"])

    for key in ["cpm25"] + met_vars + emission_vars:
        try:
            norm_stats[key] = compute_stats(
                key=key,
                all_data=all_data,
                log_features=log_features,
            )

            mean, std = norm_stats[key]
            print(f"{key}: mean={mean:.4f}, std={std:.4f}")

        except Exception as e:
            print(f"{key}: SKIPPED ({e})")

    return norm_stats


def save_norm_stats(norm_stats, save_path):
    """
    Save normalization statistics to disk.
    """
    save_path = Path(save_path)

    if not save_path.is_absolute():
        save_path = PROJECT_ROOT / save_path
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(save_path, "wb") as f:
        pickle.dump(norm_stats, f)


def load_norm_stats(save_path):
    save_path = Path(save_path)

    if not save_path.is_absolute():
        save_path = PROJECT_ROOT / save_path

    with open(save_path, "rb") as f:
        return pickle.load(f)


def normalize(arr, key, norm_stats, log_features):
    """
    Normalize feature using precomputed statistics.
    """
    mean, std = norm_stats[key]

    arr = arr.astype(np.float32)

    if key in log_features:
        arr = np.log1p(arr * 1e9)

    return (arr - mean) / std


def denormalize_pm25(arr_norm, norm_stats):
    """
    Convert normalized PM2.5 back to original scale.
    """
    mean, std = norm_stats["cpm25"]
    return np.expm1(arr_norm * std + mean) / 1e9


if __name__ == "__main__":
    config = load_config()

    all_data = load_all_months(config)

    norm_stats = compute_norm_stats(
        config=config,
        all_data=all_data,
    )

    save_norm_stats(
        norm_stats=norm_stats,
        save_path=config["artifacts"]["norm_stats_path"],
    )

    print("\n✅ Normalization statistics saved successfully.")
    print(f"Total processed features: {len(norm_stats)}")