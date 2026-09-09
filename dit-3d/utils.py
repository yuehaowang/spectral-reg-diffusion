"""Logging, EMA, and point-cloud visualization helpers."""
import logging
import os
import random
import sys
from collections import OrderedDict
from shutil import copyfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


# ---------------------------------------------------------------------------
#                           Logging / IO helpers
# ---------------------------------------------------------------------------

def setup_logging(output_dir):
    fmt = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    logger.handlers = []
    fh = logging.FileHandler(os.path.join(output_dir, "output.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    return logger


def get_output_dir(model_dir, experiment_name):
    out = os.path.join(model_dir, experiment_name)
    os.makedirs(out, exist_ok=True)
    return out


def make_subdir(parent, name):
    p = os.path.join(parent, name)
    os.makedirs(p, exist_ok=True)
    return p


def copy_source(file, output_dir):
    copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
#                                  EMA
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA params towards the live model, handling DDP-prefixed keys."""
    ema_params = OrderedDict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        name = name.replace("model.module.", "model.")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def grad_norm(net):
    p_norm = torch.sqrt(sum(p.pow(2).sum() for p in net.parameters()))
    g_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in net.parameters() if p.grad is not None))
    return p_norm, g_norm


# ---------------------------------------------------------------------------
#                          Point-cloud visualization
# ---------------------------------------------------------------------------

def visualize_pointcloud_batch(path, pointclouds, elev=30, azim=225, color="g"):
    """Plot a grid of point clouds and save to ``path``."""
    n = len(pointclouds)
    ncols = max(1, int(np.sqrt(n)))
    nrows = max(1, (n - 1) // ncols + 1)
    fig = plt.figure(figsize=(20, 20))
    for i, pc in enumerate(pointclouds):
        pc = pc.cpu().numpy()
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=color, s=5)
        ax.view_init(elev=elev, azim=azim)
        ax.axis("off")
    plt.savefig(path)
    plt.close(fig)
