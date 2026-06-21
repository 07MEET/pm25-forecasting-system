import gc
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from src.models.phase2model import Phase2Model
from src.data.loader import (
    load_config,
    load_norm_stats,
    normalize,
    denormalize_pm25,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

config = load_config()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = (
    config["training"]["use_amp"]
    and DEVICE.type == "cuda"
)

norm_stats = load_norm_stats(
    config["artifacts"]["norm_stats_path"]
)

# ------------------------------------------------------------------
# Feature ordering (must match training)
# ------------------------------------------------------------------

other_keys = [
    key
    for key in (
        config["features"]["met_vars"]
        + config["features"]["emission_vars"]
    )
    if key in norm_stats
]

u_idx = (
    other_keys.index("u10")
    if "u10" in other_keys
    else None
)

v_idx = (
    other_keys.index("v10")
    if "v10" in other_keys
    else None
)

C_met = len(other_keys)




# ── Inference ──────────────────────────────────────────────────────────────────
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

_model_inf = Phase2Model(
    met_channels=C_met,
    hidden_dim=config["model"]["hidden_dim"],
    kernel_size=config["model"]["kernel_size"],
    num_layers=config["model"]["num_layers"],
    u_idx=u_idx,
    v_idx=v_idx,
    forecast_steps=config["forecast"]["forecast_steps"],
)
_model_inf.load_state_dict(
    torch.load(
        config["paths"]["model_path"],
        map_location=DEVICE,
    )
)
_model_inf = _model_inf.to(DEVICE).eval()
print('Best model loaded for inference.')

# Load test PM2.5
TEST_IN = Path(config["data"]["test_path"])

test_pm25_raw = np.load(
    TEST_IN / "cpm25.npy"
).astype(np.float32)

N              = test_pm25_raw.shape[0]
print(f'Test samples: {N}')  # 218
test_pm25_norm = normalize(
    test_pm25_raw,
    "cpm25",
    norm_stats,
    set(config["features"]["log_features"]),
)
del test_pm25_raw; gc.collect()

avail_keys = [
    key
    for key in other_keys
    if (TEST_IN / f"{key}.npy").exists()
]
print(f'Available test met: {len(avail_keys)} keys')

all_preds   = []
INFER_BATCH = 8  # larger batch for faster inference

with torch.no_grad():
    for i in tqdm(range(0, N, INFER_BATCH), desc='Inference'):
        batch_mets = []
        for k in avail_keys:
            arr = np.load(
                TEST_IN / f"{k}.npy",
                mmap_mode="r",
            )
            slc = np.array(arr[i:i+INFER_BATCH]).astype(np.float32)
            batch_mets.append(
                normalize(
                    slc,
                    k,
                    norm_stats,
                    set(config["features"]["log_features"]),
                )
            )

        met_b = torch.tensor(np.stack(batch_mets, axis=2)).to(DEVICE)
        pm_b  = torch.tensor(test_pm25_norm[i:i+INFER_BATCH]).to(DEVICE)

        with autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP,
        ):
            pred_norm = _model_inf(pm_b, met_b).cpu().float().numpy()

        pred = np.clip(
            denormalize_pm25(
                pred_norm,
                norm_stats,
            ),
            0,
            500,
        )
        all_preds.append(pred)
        del pm_b, met_b, batch_mets
        if torch.cuda.is_available(): torch.cuda.empty_cache()

preds     = np.concatenate(all_preds, axis=0)  # (218, 16, H, W)
preds_out = preds.transpose(0, 2, 3, 1)        # (218, 140, 124, 16)

assert preds_out.shape == (N, 140, 124, 16), f'Shape mismatch: {preds_out.shape}'
OUTPUT = Path(config["paths"]["output_path"])
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

np.save(OUTPUT, preds_out)

print(f'Saved: {OUTPUT}')
print(f'Shape: {preds_out.shape} | min={preds_out.min():.4f} max={preds_out.max():.2f} mean={preds_out.mean():.2f}')