"""Training driver for 2D rectified-flow models.

Public entry point: ``train_single(cfg, model_type, save_dir, device)``.
Per-dataset wrappers live under ``exps/run_<dataset>.py``.
"""
from dataclasses import dataclass, field
from functools import partial
from typing import List, Tuple
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import data
from models import DiffusionMLP, SS_DiffusionMLP
from losses import rf_loss_fn, ss_rf_loss_fn


torch.set_default_dtype(torch.float32)


@dataclass
class Config:
    # Optimization
    num_epochs: int = 2000
    bs: int = 4096
    lr: float = 1.6e-3
    seed: int = 0

    # Data
    n_train_samples: int = 1000
    space_dims: int = 2
    dist_name: str = "2spirals"
    normalize_scale: float = 4.0

    # MLP architecture
    hidden_dim: int = 512
    hidden_layers: int = 8

    # SS-MLP additions
    probe_layer: int = 3
    projector_dims: Tuple[int, ...] = (128, 64)

    # Spectral regularization
    proj_triu_coeff: float = 1.0
    proj_loss_weight: float = 1.0
    proj_psi_scale: float = 1.0

    # Sampling
    T: int = 300
    num_samples: int = 3000

    # Checkpointing (-1 => only the final model is saved)
    checkpoint_interval: int = -1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(cfg: Config):
    np.random.seed(cfg.seed)
    x = data.get_2d_data_points(cfg.dist_name, cfg.n_train_samples) / cfg.normalize_scale
    x = torch.from_numpy(x.astype(np.float32))
    return TensorDataset(x), x


def create_model_and_optimizer(cfg: Config, device: torch.device, model_type: str):
    if model_type == "mlp":
        model = DiffusionMLP(
            in_dim=cfg.space_dims, out_dim=cfg.space_dims,
            hidden_dim=cfg.hidden_dim, hidden_layers=cfg.hidden_layers,
        )
    elif model_type == "ssmlp":
        model = SS_DiffusionMLP(
            in_dim=cfg.space_dims, out_dim=cfg.space_dims,
            hidden_dim=cfg.hidden_dim, hidden_layers=cfg.hidden_layers,
            probe_layer=cfg.probe_layer, projector_dims=cfg.projector_dims,
        )
    else:
        raise ValueError(f"unknown model_type: {model_type!r}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    return model, optimizer


@torch.no_grad()
def generate_samples(cfg: Config, model: nn.Module, device: torch.device):
    """Euler-step rectified-flow sampling from N(0, I) to the data manifold.

    Initial noise is drawn on CPU then moved to device (matches the original
    code's RNG ordering — CUDA randn would consume from a different stream).
    """
    eval_xt = torch.randn(cfg.num_samples, cfg.space_dims).to(device)
    dt = 1.0 / cfg.T
    for t in torch.linspace(0, 1, cfg.T):
        eval_xt = eval_xt + model(torch.Tensor([t]).to(device), eval_xt) * dt
    return eval_xt.cpu().numpy()


def samples_from_ckpts(ckpts_dir: str, epochs, model_type: str,
                       device: torch.device, sample_seed: int = 1234,
                       num_samples: int = 5000):
    """For each epoch in ``epochs``, load checkpoint_epoch_<ep>.pt (falls back
    to final_model.pt when missing) and return its samples.

    Returns {epoch: (num_samples, 2) ndarray}.
    """
    import os
    out = {}
    for ep in epochs:
        path = os.path.join(ckpts_dir, f"checkpoint_epoch_{ep}.pt")
        if not os.path.exists(path):
            path = os.path.join(ckpts_dir, "final_model.pt")
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = ck["config"]
        model, _ = create_model_and_optimizer(cfg, device, model_type)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()

        cfg_sample = Config(**{**cfg.__dict__})
        cfg_sample.num_samples = num_samples
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
        out[ep] = generate_samples(cfg_sample, model, device)
    return out


def train(cfg: Config, model: nn.Module, optimizer: torch.optim.Optimizer,
          train_dataset, loss_fn, device: torch.device, ckpts_dir: str):
    loss_history = {}
    intermediate_samples = {}
    dataloader = DataLoader(train_dataset, batch_size=cfg.bs, shuffle=True)
    pbar = tqdm(range(cfg.num_epochs))

    for epoch in pbar:
        epoch_losses = {}
        for (x1,) in dataloader:
            x1 = x1.to(device)
            optimizer.zero_grad()
            loss_dict = loss_fn(model, x1)
            loss_dict["loss"].backward()
            optimizer.step()
            for k, v in loss_dict.items():
                epoch_losses.setdefault(k, []).append(v.item())

        for k, vs in epoch_losses.items():
            loss_history.setdefault(k, []).append(float(np.mean(vs)))

        if cfg.checkpoint_interval > 0 and (epoch + 1) % cfg.checkpoint_interval == 0:
            # Record intermediate samples FIRST (advances both CPU and CUDA
            # RNG — matches the original code, which called generate_samples
            # at the same cadence). Then save the checkpoint.
            model.eval()
            samples = generate_samples(cfg, model, device)
            intermediate_samples[epoch + 1] = samples
            model.train()
            ckpt = os.path.join(ckpts_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
            }, ckpt)

        pbar.set_postfix_str(", ".join(f"{k}: {np.mean(v):.3f}"
                                       for k, v in epoch_losses.items()))
    return loss_history, intermediate_samples


def train_single(cfg: Config, model_type: str, save_dir: str, device: torch.device):
    """End-to-end: build, train, save final ckpt. Returns
    (training_samples, final_samples, loss_history).
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpts_dir = os.path.join(save_dir, "ckpts")
    os.makedirs(ckpts_dir, exist_ok=True)

    set_seed(cfg.seed)
    train_dataset, training_samples = load_data(cfg)
    model, optimizer = create_model_and_optimizer(cfg, device, model_type)

    if model_type == "mlp":
        loss_fn = partial(rf_loss_fn, device=device)
    else:
        loss_fn = partial(
            ss_rf_loss_fn, device=device,
            proj_triu_coeff=cfg.proj_triu_coeff,
            proj_loss_weight=cfg.proj_loss_weight,
            proj_psi_scale=cfg.proj_psi_scale,
        )

    loss_history, intermediate_samples = train(
        cfg, model, optimizer, train_dataset, loss_fn, device, ckpts_dir)
    final_samples = generate_samples(cfg, model, device)

    torch.save({
        "epoch": cfg.num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "model_type": model_type,
        "loss_history": loss_history,
        "intermediate_samples": intermediate_samples,
        "final_samples": final_samples,
    }, os.path.join(ckpts_dir, "final_model.pt"))

    return training_samples.numpy(), final_samples, loss_history
