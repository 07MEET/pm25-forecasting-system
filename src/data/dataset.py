import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.loader import normalize


class PM25Dataset(Dataset):
    """
    Dataset for PM2.5 forecasting.

    Each sample consists of:
        - PM2.5 history      : (history_pm, H, W)
        - Meteorology history: (history_pm, C, H, W)
        - PM2.5 future       : (forecast_steps, H, W)
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

        other_keys = [
            key
            for key in (met_vars + emission_vars)
            if key in norm_stats
        ]

        for month, data in data_dict.items():

            total_timesteps = data["cpm25"].shape[0]

            pm = normalize(
                data["cpm25"],
                "cpm25",
                norm_stats,
                log_features,
            )

            met_stack = np.stack(
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
                total_timesteps - window + 1,
                stride,
            ):

                pm_hist = pm[
                    t : t + history_pm
                ]

                pm_future = pm[
                    t + history_pm :
                    t + window
                ]

                met_hist = met_stack[
                    t : t + history_pm
                ]

                ep_frac = ep_mask[
                    t + history_pm :
                    t + window
                ].mean()

                self.samples.append(
                    (
                        pm_hist,
                        met_hist,
                        pm_future,
                    )
                )

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
            torch.tensor(
                pm_hist,
                dtype=torch.float32,
            ),
            torch.tensor(
                met_hist,
                dtype=torch.float32,
            ),
            torch.tensor(
                pm_future,
                dtype=torch.float32,
            ),
        )