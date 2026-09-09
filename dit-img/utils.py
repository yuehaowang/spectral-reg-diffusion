import os
import random

import imageio

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from torch.autograd import Function


def make_grid_uint8(im):
    im = torchvision.utils.make_grid(im, nrow=8).permute(1, 2, 0)
    im = (im.cpu().numpy() * 255).astype(np.uint8)
    return im


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def save_checkpoint(
    checkpoint_path,
    model=None,
    ema=None,
    optimizer=None,
    configs=None,
    global_step=None,
):
    checkpoint = {}
    if model is not None:
        checkpoint["model"] = model.state_dict()
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if configs is not None:
        checkpoint["configs"] = configs
    if global_step is not None:
        checkpoint["global_step"] = global_step
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model=None, ema=None, optimizer=None):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if model is not None and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    global_step = ckpt["global_step"]
    return global_step


def save_image(tgt, img):
    imageio.imwrite(tgt, img)


def save_npz(path, **kwargs):
    np.savez(path, **kwargs)


class StatsDict(dict):
    def __init__(self, default_factory=None, *args, **kwargs):
        self.default_factory = default_factory
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        # Initialize the missing key with an empty list
        self[key] = []
        return self[key]


class AllReduceSum(Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


all_reduce_sum = AllReduceSum.apply
