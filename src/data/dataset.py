import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.loader import normalize


class PM25Dataset(Dataset):
    """
    Dataset for PM2.5 forecasting.

    Creates sliding-window samples consisting of:
    - PM2.5 history
    - Meteorological history
    - Future PM2.5 targets

    Also computes sample weights based on detected pollution
    episodes for use with WeightedRandomSampler.
    """

    def __init__(
        self,
        data_dict,
        episode_masks,
        norm_stats,
        config,
        stride=1,
    ):
        self.samples = []
        self.weights = []

        met_vars = config["features"]["met_vars"]
        emission_vars = config["features"]["emission_vars"]
        log_features = set(config["features"]["log_features"])

        history_pm = config["forecast"]["history_pm"]
        forecast_steps = config["forecast"]["forecast_steps"]
        window = history_pm + forecast_steps

        # Keep feature order identical to notebook
        other_keys = [
            key
            for key in (met_vars + emission_vars)
            if key in norm_stats
        ]

        for month, data in data_dict.items():

            if month not in episode_masks:
                raise KeyError(
                    f"Episode mask not found for month: {month}"
                )

            total_steps = data["cpm25"].shape[0]

            pm = normalize(
                data["cpm25"],
                "cpm25",
                norm_stats,
                log_features,
            )

            mets = np.stack(
                [
                    normalize(
                        data[key],
                        key,
                        norm_stats,
                        log_features,
                    )
                    for key in other_keys
                    if key in data
                ],
                axis=1,
            )

            ep_mask = episode_masks[month]

            for t in range(
                0,
                total_steps - window + 1,
                stride,
            ):
                pm_hist = pm[
                    t : t + history_pm
                ]

                pm_future = pm[
                    t + history_pm : t + window
                ]

                met_hist = mets[
                    t : t + history_pm
                ]

                ep_frac = ep_mask[
                    t + history_pm : t + window
                ].mean()

                self.samples.append(
                    (
                        pm_hist,
                        met_hist,
                        pm_future,
                    )
                )

                # Same weighting as notebook
                self.weights.append(
                    1.0 + 9.0 * ep_frac
                )

        self.weights = np.asarray(
            self.weights,
            dtype=np.float32,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pm_hist, met_hist, pm_future = self.samples[idx]

        return (
            torch.from_numpy(pm_hist).float(),
            torch.from_numpy(met_hist).float(),
            torch.from_numpy(pm_future).float(),
        )