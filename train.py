import random
import time
from pathlib import Path
import shutil
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from src.data.loader import (
    load_config,
    load_all_months,
    compute_norm_stats,
    save_norm_stats,
)
from src.data.episodes import identify_episodes
from src.data.dataset import PM25Dataset
from src.models.phase2model import Phase2Model
from src.training.losses import combined_loss
from src.data.loader import PROJECT_ROOT
from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
)
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)

from torch.amp import (
    GradScaler,
    autocast,
)

def main():
    # -----------------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------------

    config = load_config()


    # -----------------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------------

    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # -----------------------------------------------------------------------------
    # Runtime
    # -----------------------------------------------------------------------------

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    USE_AMP = (
        config["training"]["use_amp"]
        and DEVICE.type == "cuda"
    )

    ACCUM_STEPS = config["training"]["accum_steps"]


    # -----------------------------------------------------------------------------
    # Output paths
    # -----------------------------------------------------------------------------

    CKPT = Path(config["paths"]["model_path"])
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    params_copy = CKPT.parent / "params_used.yaml"

    if not params_copy.exists():
        shutil.copy(
            PROJECT_ROOT / "params.yaml",
            params_copy,
        )

    # -----------------------------------------------------------------------------
    # Load training data
    # -----------------------------------------------------------------------------

    print("=" * 80)
    print("Loading training data...")
    print("=" * 80)

    all_data = load_all_months(config)

    print(f"Loaded months: {list(all_data.keys())}")


    # -----------------------------------------------------------------------------
    # Compute normalization statistics
    # -----------------------------------------------------------------------------

    print("=" * 80)
    print("Computing normalization statistics...")
    print("=" * 80)

    norm_stats = compute_norm_stats(
        config=config,
        all_data=all_data,
    )

    save_norm_stats(
        norm_stats=norm_stats,
        save_path=config["artifacts"]["norm_stats_path"],
    )

    print("Normalization statistics saved successfully.")

    # -----------------------------------------------------------------------------
    # Compute episode masks
    # -----------------------------------------------------------------------------

    print("=" * 80)
    print("Computing episode masks...")
    print("=" * 80)

    episode_masks = {}

    for month, month_data in all_data.items():
        print(f"Processing {month}...")
        episode_masks[month] = identify_episodes(
            month_data["cpm25"]
        )

    print("Episode masks computed successfully.")


    # -----------------------------------------------------------------------------
    # Train / Validation Split
    # -----------------------------------------------------------------------------

    TRAIN_MONTHS = config["months"][:3]
    VAL_MONTH = config["months"][-1]

    train_data = {}
    val_data = {}

    train_ep = {}
    val_ep = {}

    # First three months → training
    for month in TRAIN_MONTHS:
        train_data[month] = {
            k: v
            for k, v in all_data[month].items()
            if hasattr(v, "shape")
        }
        train_ep[month] = episode_masks[month]

    # DEC_16 → 5% train, 95% validation
    dec_data = all_data[VAL_MONTH]

    T = dec_data["cpm25"].shape[0]
    split = int(T * 0.05)

    train_data[VAL_MONTH] = {
        k: v[:split]
        for k, v in dec_data.items()
        if hasattr(v, "shape")
        and v.shape[0] == T
    }

    val_data[VAL_MONTH] = {
        k: v[split:]
        for k, v in dec_data.items()
        if hasattr(v, "shape")
        and v.shape[0] == T
    }

    train_ep[VAL_MONTH] = episode_masks[VAL_MONTH][:split]
    val_ep[VAL_MONTH] = episode_masks[VAL_MONTH][split:]

    train_ds = PM25Dataset(
        data_dict=train_data,
        episode_masks=train_ep,
        norm_stats=norm_stats,
        config=config,
        stride=1,
    )

    val_ds = PM25Dataset(
        data_dict=val_data,
        episode_masks=val_ep,
        norm_stats=norm_stats,
        config=config,
        stride=2,
    )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            train_ds.weights,
            dtype=torch.float32,
        ),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"Train samples : {len(train_ds)}")
    print(f"Validation samples : {len(val_ds)}")

    C_met = train_ds[0][1].shape[1]

    print(f"Meteorological channels: {C_met}")

    # -----------------------------------------------------------------------------
    # Model Initialization
    # -----------------------------------------------------------------------------

    # Keep feature ordering identical to the notebook
    other_keys = [
        key
        for key in (
            config["features"]["met_vars"]
            + config["features"]["emission_vars"]
        )
        if key in norm_stats
    ]

    # Wind indices
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


    # Build model
    model = Phase2Model(
        met_channels=C_met,
        hidden_dim=config["model"]["hidden_dim"],
        kernel_size=config["model"]["kernel_size"],
        num_layers=config["model"]["num_layers"],
        u_idx=u_idx,
        v_idx=v_idx,
        forecast_steps=config["forecast"]["forecast_steps"],
    )

    # Multi-GPU support (same behavior as notebook)
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel on {num_gpus} GPUs")

    model = model.to(DEVICE)

    # Parameter count
    base_model = (
        model.module
        if hasattr(model, "module")
        else model
    )

    num_params = sum(
        p.numel()
        for p in base_model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {num_params:,}")

    # -----------------------------------------------------------------------------
    # Model sanity check
    # -----------------------------------------------------------------------------

    with torch.no_grad():

        dummy_pm = torch.zeros(
            1,
            config["forecast"]["history_pm"],
            config["forecast"]["height"],
            config["forecast"]["width"],
            device=DEVICE,
        )

        dummy_met = torch.zeros(
            1,
            config["forecast"]["history_pm"],
            C_met,
            config["forecast"]["height"],
            config["forecast"]["width"],
            device=DEVICE,
        )

        output = model(dummy_pm, dummy_met)

        print(f"Model output shape: {output.shape}")

    del dummy_pm
    del dummy_met
    del output

    # -----------------------------------------------------------------------------
    # Optimizer / Scheduler / AMP
    # -----------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config["scheduler"]["warmup_epochs"],
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(
            config["training"]["epochs"]
            - config["scheduler"]["warmup_epochs"],
            1,
        ),
        eta_min=config["scheduler"]["min_lr"],
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[
            config["scheduler"]["warmup_epochs"]
        ],
    )

    scaler = GradScaler(
        DEVICE.type,
        enabled=USE_AMP,
    )

    def get_tf_ratio(epoch, total_epochs):
        start = config["training"]["teacher_forcing_start"]
        end = config["training"]["teacher_forcing_end"]

        progress = min(epoch / total_epochs, 1.0)

        return start * (1.0 - progress) + end * progress

    # -----------------------------------------------------------------------------
    # Training / Validation Epoch
    # -----------------------------------------------------------------------------

    def run_epoch(dl, train=True, tf_ratio=0.0):

        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0

        context = torch.enable_grad() if train else torch.no_grad()

        with context:

            if train:
                optimizer.zero_grad()

            for step, (pm_hist, met_hist, pm_fut) in enumerate(tqdm(dl, leave=False)):

                pm_hist = pm_hist.to(
                    DEVICE,
                    non_blocking=True,
                )

                met_hist = met_hist.to(
                    DEVICE,
                    non_blocking=True,
                )

                pm_fut = pm_fut.to(
                    DEVICE,
                    non_blocking=True,
                )

                with autocast(
                    device_type=DEVICE.type,
                    enabled=USE_AMP,
                ):

                    if train and tf_ratio > 0:

                        pred = model(
                            pm_hist,
                            met_hist,
                            teacher_forcing_ratio=tf_ratio,
                            pm_fut_gt=pm_fut,
                        )

                    else:

                        pred = model(
                            pm_hist,
                            met_hist,
                        )

                    loss = combined_loss(
                        pred,
                        pm_fut,
                        pm_hist,
                    )

                if train:

                    scaler.scale(
                        loss / ACCUM_STEPS
                    ).backward()

                    if (
                        (step + 1) % ACCUM_STEPS == 0
                        or
                        (step + 1) == len(dl)
                    ):

                        scaler.unscale_(optimizer)

                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config["training"]["grad_clip"],
                        )

                        scaler.step(optimizer)
                        scaler.update()

                        optimizer.zero_grad()

                total_loss += loss.item()

        return total_loss / max(len(dl), 1)

    # -----------------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------------

    best_val = float("inf")
    patience_counter = 0

    training_start = time.time()

    print(
        f"Training: {config['training']['epochs']} epochs | "
        f"batch={config['training']['batch_size']} "
        f"(eff={config['training']['batch_size'] * ACCUM_STEPS}) | "
        f"AMP={USE_AMP}"
    )

    print(
        f'{"Ep":>4} {"Train":>8} {"Val":>8} {"LR":>9} {"TF":>5} {"Time":>7}'
    )

    print("-" * 52)

    for epoch in range(1, config["training"]["epochs"] + 1):

        epoch_start = time.time()

        tf_ratio = get_tf_ratio(
            epoch,
            config["training"]["epochs"],
        )

        train_loss = run_epoch(
            train_dl,
            train=True,
            tf_ratio=tf_ratio,
        )

        val_loss = run_epoch(
            val_dl,
            train=False,
            tf_ratio=0.0,
        )

        scheduler.step()

        lr = scheduler.get_last_lr()[0]

        minutes = (time.time() - epoch_start) / 60

        tag = ""

        if val_loss < best_val:

            best_val = val_loss
            patience_counter = 0

            state_dict = (
                model.module.state_dict()
                if hasattr(model, "module")
                else model.state_dict()
            )

            torch.save(
                state_dict,
                CKPT,
            )

            tag = " ✓"

        else:

            patience_counter += 1

        elapsed_hours = (
            time.time() - training_start
        ) / 3600

        print(
            f"{epoch:>4d} "
            f"{train_loss:>8.4f} "
            f"{val_loss:>8.4f} "
            f"{lr:>9.2e} "
            f"{tf_ratio:>5.2f} "
            f"{minutes:>6.1f}m"
            f"{tag}"
        )

        if elapsed_hours > config["training"]["max_training_hours"]:

            print(
                f"Time limit approaching ({elapsed_hours:.1f}h). "
                "Stopping training."
            )

            break

        if patience_counter >= config["training"]["patience"]:

            print(
                f"Early stopping at epoch {epoch} "
                f"(patience={config['training']['patience']})"
            )

            break

    print()

    print(
        f"Best validation loss: {best_val:.4f}"
    )

    print(
        f"Total training time: "
        f"{(time.time() - training_start) / 3600:.2f} hours"
    )

if __name__ == "__main__":
    main()