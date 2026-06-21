import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Model ──────────────────────────────────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch,
                              kernel_size, padding=kernel_size // 2)
        self.ln   = nn.GroupNorm(1, 4 * hidden_ch)

    def forward(self, x, h, c):
        gates      = self.ln(self.conv(torch.cat([x, h], 1)))
        i, f, g, o = gates.chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

    def init_hidden(self, B, H, W, device):
        return (torch.zeros(B, self.hidden_ch, H, W, device=device),
                torch.zeros(B, self.hidden_ch, H, W, device=device))


class SpatialAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 1), nn.ReLU(),
            nn.Conv2d(in_ch // 4, 1, 1),     nn.Sigmoid()
        )
    def forward(self, x): return x * self.conv(x)


class WindWarp(nn.Module):
    """Physics transport via last known wind. Scale decays with forecast step."""
    def __init__(self, u_idx, v_idx):
        super().__init__()
        self.u_idx, self.v_idx = u_idx, v_idx
        # Learnable per-step scale — starts at 0.05, model adapts
        self.log_scale = nn.Parameter(torch.tensor(-3.0))  # exp(-3) ≈ 0.05

    def forward(self, pm, last_met, step=0):
        if self.u_idx is None: return pm
        B, _, H, W = pm.shape
        scale = torch.exp(self.log_scale) / (1.0 + 0.1 * step)  # decay with step
        u  = last_met[:, self.u_idx:self.u_idx+1]
        v  = last_met[:, self.v_idx:self.v_idx+1]
        xs = torch.linspace(
            -1, 1, W,
            device=pm.device,
            dtype=pm.dtype,
        )

        ys = torch.linspace(
            -1, 1, H,
            device=pm.device,
            dtype=pm.dtype,
        )
        gy, gx   = torch.meshgrid(ys, xs, indexing='ij')
        grid     = torch.stack([gx, gy], -1).unsqueeze(0).expand(B, -1, -1, -1)
        flow     = torch.stack([u.squeeze(1), v.squeeze(1)], -1) * scale
        warped_g = (grid + flow).clamp(-1, 1)
        return F.grid_sample(pm, warped_g, align_corners=False,
                             mode='bilinear', padding_mode='border')


class SpatialEncoder(nn.Module):
    """Multi-scale dilated conv for local + regional spatial patterns."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.e1   = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, dilation=1), nn.ReLU())
        self.e2   = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2), nn.ReLU())
        self.e3   = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=4, dilation=4), nn.ReLU())
        self.fuse = nn.Conv2d(out_ch * 3, out_ch, 1)
    def forward(self, x):
        return self.fuse(torch.cat([self.e1(x), self.e2(x), self.e3(x)], 1))


class EpisodeDetector(nn.Module):
    """Learns WHERE episodes occur from hidden state → soft map for decoder."""
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),    nn.ReLU(),
            nn.Conv2d(16, 1,  1),               nn.Sigmoid()
        )
    def forward(self, h): return self.net(h)


class Phase2Model(nn.Module):
    """
    Autoregressive ConvLSTM encoder-decoder with:
    - WindWarp (learnable decay scale)
    - EpisodeDetector (soft episode map fed to decoder)
    - SpatialEncoder (multi-scale dilated)
    - SpatialAttention
    - Episode amplitude amplifier
    - Scheduled teacher forcing
    """
    def __init__(
        self,
        met_channels,
        hidden_dim,
        kernel_size,
        num_layers,
        u_idx,
        v_idx,
        forecast_steps,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.forecast_steps = forecast_steps
        enc_met_ch      = hidden_dim // 2

        self.spatial_enc    = SpatialEncoder(met_channels, enc_met_ch)
        self.wind_warp      = WindWarp(u_idx, v_idx)
        self.episode_detect = EpisodeDetector(hidden_dim)

        self.enc_cells = nn.ModuleList()
        self.dec_cells = nn.ModuleList()
        for i in range(num_layers):
            enc_in = (1 + enc_met_ch)     if i == 0 else hidden_dim
            dec_in = (1 + enc_met_ch + 1) if i == 0 else hidden_dim
            self.enc_cells.append(ConvLSTMCell(enc_in, hidden_dim, kernel_size))
            self.dec_cells.append(ConvLSTMCell(dec_in, hidden_dim, kernel_size))

        self.attn        = SpatialAttention(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )
        self.episode_amp = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1), nn.Softplus()
        )

    def forward(self, pm_hist, met_hist, teacher_forcing_ratio=0.0, pm_fut_gt=None):
        """
        pm_hist             : (B, 10, H, W)
        met_hist            : (B, 10, C, H, W)
        teacher_forcing_ratio: during training, use ground truth with this probability
        pm_fut_gt           : (B, 16, H, W) ground truth for teacher forcing
        """
        B, T_hist, H, W = pm_hist.shape
        states = [cell.init_hidden(B, H, W, pm_hist.device) for cell in self.enc_cells]

        # Encoder: process 10h history
        for t in range(T_hist):
            enc_met = self.spatial_enc(met_hist[:, t])
            x = torch.cat([pm_hist[:, t:t+1], enc_met], 1)
            for i, cell in enumerate(self.enc_cells):
                h, c = states[i]; h, c = cell(x, h, c); states[i] = (h, c); x = h

        last_met  = met_hist[:, -1]
        enc_met_d = self.spatial_enc(last_met)  # pre-encode once

        # Autoregressive decoder: predict 16 steps
        preds      = []
        prev_pm    = pm_hist[:, -1:].clone()
        dec_states = list(states)

        for t in range(self.forecast_steps):
            warped = self.wind_warp(prev_pm, last_met, step=t)
            h_cur  = dec_states[-1][0]
            ep_map = self.episode_detect(h_cur)

            x = torch.cat([warped, enc_met_d, ep_map], 1)
            for i, cell in enumerate(self.dec_cells):
                h, c = dec_states[i]; h, c = cell(x, h, c); dec_states[i] = (h, c); x = h

            x      = self.attn(x)
            delta  = self.output_head(x)
            amp    = self.episode_amp(x)
            pred_t = prev_pm + delta + ep_map * amp
            # Clip to prevent runaway autoregressive predictions
            pred_t = pred_t.clamp(-10, 30)
            preds.append(pred_t)

            # Scheduled teacher forcing
            if teacher_forcing_ratio > 0.0 and pm_fut_gt is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_pm = pm_fut_gt[:, t:t+1]
            else:
                prev_pm = pred_t

        return torch.cat(preds, dim=1)  # (B, 16, H, W)

