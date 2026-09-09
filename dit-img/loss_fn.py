import math

import torch
import torch.nn as nn
from accelerate.utils import broadcast
from utils import all_reduce_sum


def timestep_sampler_ln(size, device=torch.device("cpu")):
    t = torch.rand(size, device=device)
    return t


def timestep_sampler_sigmoid(size, device=torch.device("cpu")):
    nt = torch.randn(size, device=device)
    t = torch.sigmoid(nt)
    return t


def rf_loss_fn(x1, y, model):
    # sample time
    t = timestep_sampler_sigmoid((x1.shape[0], 1, 1, 1), x1.device)
    # sample a random seed
    x0 = torch.randn_like(x1)
    # linear interpolation
    xt = t * x1 + (1 - t) * x0
    dot_xt = x1 - x0

    # model prediction
    vt = model(xt, t.squeeze(1, 2, 3), y)
    # MSE loss
    mse = ((vt - dot_xt) ** 2).mean()

    losses = {"mse": mse}
    return losses


def ss_repl_rf_loss_fn(
    x1, y, model, psi_scale=1, single_t=False, use_global_batch=True, half_batch=False
):
    # sample time
    if single_t:
        t = timestep_sampler_sigmoid((1, 1, 1, 1), x1.device)
        t = t.repeat(x1.shape[0], 1, 1, 1)
        if use_global_batch:
            t = broadcast(t, from_process=0)
    else:
        t = timestep_sampler_sigmoid((x1.shape[0], 1, 1, 1), x1.device)

    # sample random seeds
    x0_1 = torch.randn_like(x1)
    x0_2 = torch.randn_like(x1)

    # linear interpolation
    xt_1 = t * x1 + (1 - t) * x0_1
    xt_2 = t * x1 + (1 - t) * x0_2

    # vt_1, psi_1 = model(xt_1, t.squeeze(1, 2, 3), y, return_projector_output=True)
    # vt_2, psi_2 = model(xt_2, t.squeeze(1, 2, 3), y, return_projector_output=True)
    vt, psi = model(
        torch.cat([xt_1, xt_2]),
        torch.cat([t.squeeze(1, 2, 3), t.squeeze(1, 2, 3)]),
        torch.cat([y, y]) if y is not None else None,
        return_projector_output=True,
    )
    vt_1, vt_2 = vt.chunk(2)
    psi_1, psi_2 = psi.chunk(2)  # each (B, L, K)

    ## MSE loss

    mse = ((vt_1 - (x1 - x0_1)) ** 2).mean()
    if not half_batch:
        mse += ((vt_2 - (x1 - x0_2)) ** 2).mean()

    ## Representation alignment

    B, L, K = psi_1.shape

    # token‐wise norm squared over batch & features
    local_norm_sq = (psi_1.pow(2) + psi_2.pow(2)).sum(dim=0)
    norm_sq = all_reduce_sum(local_norm_sq) if use_global_batch else local_norm_sq
    norm = norm_sq.sqrt().clamp(min=1e-6)  # (L, K)
    scale = math.sqrt(2.0 * psi_scale)

    # normalize each sample by the same per‐token norm
    psi_1 = psi_1 / norm.view(1, L, -1) * scale
    psi_2 = psi_2 / norm.view(1, L, -1) * scale

    # diagonal term per token
    local_diag = (psi_1 * psi_2).sum(dim=0)  # (L, K)
    diag = all_reduce_sum(local_diag) if use_global_batch else local_diag

    # off‐diagonal term per token via batched einsum
    M1 = torch.einsum("blk,blm->lkm", psi_1, psi_2)  # (L, K, K)
    M1 = torch.triu(M1, diagonal=1)
    local_triu = (M1.pow(2)).sum(dim=(1, 2))
    if not half_batch:
        M2 = torch.einsum("blk,blm->lkm", psi_2, psi_1)  # (L, K, K)
        M2 = torch.triu(M2, diagonal=1)
        local_triu = local_triu + (M2.pow(2)).sum(dim=(1, 2))  # (L,)
    triu = all_reduce_sum(local_triu) if use_global_batch else local_triu

    reg_diag = (-diag.sum(-1) * (2 if not half_batch else 1)) / K
    reg_triu = triu / K

    losses = {
        "mse": mse,
        "reg_diag": torch.mean(reg_diag),
        "reg_triu": torch.mean(reg_triu),
    }
    return losses


def ss_repl_dispersive_rf_loss_fn(
    x1,
    y,
    model,
    tau=0.5,
):
    # sample time
    t = timestep_sampler_sigmoid((x1.shape[0], 1, 1, 1), x1.device)
    # sample a random seed
    x0 = torch.randn_like(x1)
    # linear interpolation
    xt = t * x1 + (1 - t) * x0
    dot_xt = x1 - x0

    vt, zs = model(
        xt,
        t.squeeze(1, 2, 3),
        y,
        return_projector_output=True,
    )

    ## MSE loss
    mse = ((vt - dot_xt) ** 2).mean()

    ## Representation alignment
    B, L, K = zs.shape

    # zs = zs.transpose(0, 1)  # (L, B, K)

    # sq = (zs * zs).sum(-1, keepdim=True)  # (L,B,1)
    # D2 = torch.clamp(
    #     sq + sq.transpose(1, 2) - 2 * torch.bmm(zs, zs.transpose(1, 2)), min=0
    # )
    # i, j = torch.triu_indices(B, B, 1, device=zs.device)
    # s = -D2[:, i, j] / float(tau)  # (L, P)
    # proj_loss = torch.logsumexp(s, 1) - math.log(s.size(1))

    # D = torch.norm(zs[:, :, None, :] - zs[:, None, :, :], dim=-1, p=2)
    # i, j = torch.triu_indices(B, B, 1, device=zs.device)
    # s = -(D[:, i, j].square()) / float(tau)  # (L, P)
    # proj_loss = torch.logsumexp(s, 1) - math.log(s.size(1))

    z = zs.reshape((zs.shape[0], -1))  # flatten
    diff = torch.nn.functional.pdist(z).pow(2) / z.shape[1]  # pairwise distance
    diff = torch.concat((diff, diff, torch.zeros(z.shape[0]).cuda()))
    proj_loss = torch.log(torch.exp(-diff).mean())  # calculate loss

    losses = {
        "mse": mse,
        "reg_diag": torch.mean(proj_loss),
        "reg_triu": torch.Tensor([0.0]).mean().to(x1.device),
    }
    return losses
