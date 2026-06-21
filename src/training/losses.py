import torch
import torch.nn.functional as F

# ── Loss Functions ─────────────────────────────────────────────────────────────

def smape_loss(pred, target):
    return (
        (pred - target)
        .abs()
        .div(
            0.5 * (pred.abs() + target.abs()) + 1e-8
        )
        .mean()
    )

def episode_weighted_loss(pred, target, pm_hist):
    hist_mean = pm_hist.mean(dim=1, keepdim=True)
    hist_std = pm_hist.std(dim=1, keepdim=True) + 1e-8

    weights = (
        1.0
        + 2.0
        * (
            (target - hist_mean)
            / hist_std
        ).clamp(min=0).detach()
    )

    return smape_loss(
        pred * weights,
        target * weights,
    )

def spatial_gradients(x):
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]

    return grad_x, grad_y

def spatial_gradient_loss(pred, target):
    pred_dx, pred_dy = spatial_gradients(pred)
    target_dx, target_dy = spatial_gradients(target)

    return (
        F.mse_loss(pred_dx, target_dx)
        + F.mse_loss(pred_dy, target_dy)
    )
    
def pearson_corr_loss(pred, target):
    """
    Pearson correlation loss.
    Minimizing this maximizes correlation.
    """

    batch_size = pred.shape[0]

    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)

    pred_centered = (
        pred_flat
        - pred_flat.mean(dim=1, keepdim=True)
    )

    target_centered = (
        target_flat
        - target_flat.mean(dim=1, keepdim=True)
    )

    corr = (
        (pred_centered * target_centered).sum(dim=1)
        / (
            torch.sqrt(
                (pred_centered ** 2).sum(dim=1)
                * (target_centered ** 2).sum(dim=1)
            )
            + 1e-8
        )
    )

    return 1.0 - corr.mean()

def combined_loss(
    pred,
    target,
    pm_hist,
    alpha=0.5,
    beta=0.25,
    gamma=0.1,
    delta=0.15,
):
    return (
        alpha * episode_weighted_loss(
            pred,
            target,
            pm_hist,
        )
        + beta * smape_loss(
            pred,
            target,
        )
        + gamma * spatial_gradient_loss(
            pred,
            target,
        )
        + delta * pearson_corr_loss(
            pred,
            target,
        )
    )