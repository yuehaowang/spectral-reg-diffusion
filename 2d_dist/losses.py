"""Rectified-flow training losses.

  - rf_loss_fn: plain MSE (baseline MLP).
  - ss_rf_loss_fn: MSE + spectral regularization on the projector ψ
    (always uses stop-grad on one branch of the off-diagonal term).
"""
import math
import torch


def rf_loss_fn(net, x1, device):
    """Plain rectified-flow MSE loss."""
    # Draw t on CPU then move (matches the original code's RNG ordering: t
    # consumes CPU RNG, x0 = randn_like consumes CUDA RNG via x1.device).
    t = torch.rand(x1.shape[0], 1).to(device)
    x0 = torch.randn_like(x1)
    xt = (1 - t) * x0 + t * x1
    gt_v = x1 - x0
    pred_v = net(t, xt)
    mse = ((pred_v - gt_v) ** 2).sum(-1).mean()
    return {"loss": mse, "mse_loss": mse}


def ss_rf_loss_fn(net, x1, device,
                  proj_triu_coeff=1.0, proj_loss_weight=1.0, proj_psi_scale=1.0):
    """MSE + spectral-regularization loss.

    Two independent noise draws per sample (concat in the batch dim) give a
    pair of projector outputs (ψ1, ψ2) with which we estimate both the
    diagonal energy and the off-diagonal cross terms of the spectral feature
    correlation matrix. The off-diagonal term uses .detach() on one branch
    (stop-grad).
    """
    # See rf_loss_fn: CPU draw to match original RNG ordering.
    t = torch.rand(x1.shape[0], 1).to(device)
    t = torch.cat([t, t], dim=0)
    x1 = torch.cat([x1, x1], dim=0)
    x0 = torch.randn_like(x1)
    xt = (1 - t) * x0 + t * x1
    gt_v = x1 - x0

    pred_v, zs = net(t, xt, return_probed=True)
    mse = ((pred_v - gt_v) ** 2).sum(-1).mean()

    psi1, psi2 = zs.chunk(2)
    B, K = psi1.shape

    norm = (psi1.pow(2) + psi2.pow(2)).sum(dim=0).sqrt().clamp(min=1e-6)
    scale = math.sqrt(2.0 * proj_psi_scale)
    psi1 = psi1 / norm.view(1, -1) * scale
    psi2 = psi2 / norm.view(1, -1) * scale

    t1, t2 = t.chunk(2)
    psi1 = psi1 * t2 ** 2
    psi2 = psi2 * t1 ** 2

    diag = (psi1 * psi2).sum(dim=0)
    M1 = torch.triu(torch.einsum("bk,bm->km", psi1.detach(), psi2), diagonal=1)
    M2 = torch.triu(torch.einsum("bk,bm->km", psi2.detach(), psi1), diagonal=1)
    triu = M1.pow(2).sum() + M2.pow(2).sum()

    reg = -2 * diag.sum() / K + proj_triu_coeff * triu / K
    loss = mse + proj_loss_weight * reg
    return {"loss": loss, "mse_loss": mse.detach(), "proj_loss": reg.detach()}
