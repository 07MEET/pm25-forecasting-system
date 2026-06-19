import os
import pickle
import yaml
import numpy as np

def load_config(config_path="params.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def load_month(month, raw_path, met_vars, emission_vars):
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
    all_data = {}

    raw_path = config["data"]["raw_path"]
    months = config["months"]
    met_vars = config["features"]["met_vars"]
    emission_vars = config["features"]["emission_vars"]

    for month in months:
        print(f"Loading {month}...")
        all_data[month] = load_month(
            month,
            raw_path,
            met_vars,
            emission_vars,
        )

    return all_data

def compute_stats(key, all_data, log_features):
    arrays = [
        d[key]
        for d in all_data.values()
        if key in d
    ]

    combined = np.concatenate(arrays, axis=0).astype(np.float32)

    if key in log_features:
        combined = np.log1p(combined * 1e9)

    return (
        float(combined.mean()),
        float(combined.std() + 1e-8),
    )
    
def compute_norm_stats(config, all_data):
    norm_stats = {}

    log_features = set(config["features"]["log_features"])
    met_vars = config["features"]["met_vars"]
    emission_vars = config["features"]["emission_vars"]

    for key in ["cpm25"] + met_vars + emission_vars:
        norm_stats[key] = compute_stats(
            key,
            all_data,
            log_features,
        )

    return norm_stats

def save_norm_stats(norm_stats, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(norm_stats, f)


def load_norm_stats(save_path):
    with open(save_path, "rb") as f:
        return pickle.load(f)
    
def normalize(arr, key, norm_stats, log_features):
    mean, std = norm_stats[key]

    arr = arr.astype(np.float32)

    if key in log_features:
        arr = np.log1p(arr * 1e9)

    return (arr - mean) / std

def denormalize_pm25(arr_norm, norm_stats):
    mean, std = norm_stats["cpm25"]
    return np.expm1(arr_norm * std + mean) / 1e9

if __name__ == "__main__":
    config = load_config()

    all_data = load_all_months(config)
    norm_stats = compute_norm_stats(config, all_data)

    save_norm_stats(
        norm_stats,
        config["artifacts"]["norm_stats_path"],
    )

    print("✅ Saved normalization statistics successfully.")
    print(f"Features processed: {list(norm_stats.keys())}")