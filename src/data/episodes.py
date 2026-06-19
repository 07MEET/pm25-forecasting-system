from scipy.ndimage import uniform_filter1d
import numpy as np


def identify_episodes(pm_arr, seasonal_period=24):
    T, H, W = pm_arr.shape

    pm_2d = pm_arr.reshape(T, -1).astype(np.float64)

    trend = uniform_filter1d(
        pm_2d,
        size=seasonal_period,
        axis=0
    )

    detrended = pm_2d - trend

    seasonal = np.zeros_like(detrended)

    for h in range(seasonal_period):
        idx = np.arange(h, T, seasonal_period)
        seasonal[idx] = detrended[idx].mean(
            axis=0,
            keepdims=True,
        )

    residual = detrended - seasonal

    sigma = residual.std(axis=0, keepdims=True) + 1e-8

    return (residual > 3 * sigma).reshape(T, H, W)