"""Evaluate one DiT-3D checkpoint.

This script accumulates all generated evaluation samples and corresponding
reference point clouds before computing the usual PointFlow/PVD set metrics:
1-NNA, COV, and MMD with CD/EMD distances.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.utils.data

from timm_compat import ensure_timm_mlp

ensure_timm_mlp()

from diffusion import Model, get_betas
from dit3d import DiT3D_models
from metrics.pointcloud_metrics import compute_all_metrics
from shapenet_pc import ShapeNet15kPointClouds
from utils import (
    copy_source,
    get_output_dir,
    make_subdir,
    set_seed,
    setup_logging,
    visualize_pointcloud_batch,
)


CANONICAL_SPLIT_COUNTS = {
    "airplane": {"train": 2832, "val": 405, "test": 808},
    "chair": {"train": 4612, "val": 662, "test": 1317},
}


def _validate_dataset_count(category: str, split: str, actual: int) -> None:
    expected = CANONICAL_SPLIT_COUNTS.get(category, {}).get(split)
    if expected is not None and actual != expected:
        raise RuntimeError(
            f"incomplete/noncanonical ShapeNet split for {category}/{split}: "
            f"found {actual} shapes, expected {expected}"
        )


def _build_test_dataset(opt):
    train = ShapeNet15kPointClouds(
        root_dir=opt.dataroot,
        categories=[opt.category],
        split="train",
        tr_sample_size=opt.npoints,
        te_sample_size=opt.npoints,
        random_subsample=True,
    )
    dataset = ShapeNet15kPointClouds(
        root_dir=opt.dataroot,
        categories=[opt.category],
        split=opt.eval_split,
        tr_sample_size=opt.npoints,
        te_sample_size=opt.npoints,
        all_points_mean=train.all_points_mean,
        all_points_std=train.all_points_std,
    )
    if opt.strict_dataset_counts:
        _validate_dataset_count(opt.category, "train", len(train))
        _validate_dataset_count(opt.category, opt.eval_split, len(dataset))
    opt.train_shape_count = len(train)
    opt.eval_shape_count = len(dataset)
    return dataset


def _normalize_state_dict_for_single_gpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("model.module.") for key in state_dict):
        return {
            key.replace("model.module.", "model.", 1): value
            for key, value in state_dict.items()
        }
    return state_dict


def _write_metric_outputs(output_dir: Path, metrics: dict[str, float], metadata: dict[str, object]) -> None:
    payload = {**metadata, "metrics": metrics}
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics):
            writer.writerow([key, metrics[key]])


@torch.no_grad()
def generate_samples(opt, model: Model, device: torch.device, output_dir: Path, logger) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = _build_test_dataset(opt)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.bs,
        shuffle=False,
        num_workers=int(opt.workers),
        drop_last=False,
    )
    logger.info(f"evaluation dataset size={len(dataset)} batch_size={opt.bs}")

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    model.model.eval()
    samples, refs = [], []
    syn_dir = Path(make_subdir(str(output_dir), "syn"))
    remaining = opt.max_eval_shapes if opt.max_eval_shapes > 0 else None

    for step, data in enumerate(loader):
        x = data["test_points"].transpose(1, 2).to(device)
        mean = data["mean"].float()
        std = data["std"].float()
        y = data["cate_idx"]
        if remaining is not None and x.shape[0] > remaining:
            x = x[:remaining]
            mean = mean[:remaining]
            std = std[:remaining]
            y = y[:remaining]

        if opt.rand_cls:
            y_eval = torch.randint(0, opt.num_classes, (x.shape[0],), device=device)
        else:
            y_eval = y.to(device)

        gen = model.gen_samples(x.shape, device, y_eval, clip_denoised=False).detach().cpu()
        gen = gen.transpose(1, 2).contiguous()
        ref = x.detach().cpu().transpose(1, 2).contiguous()
        gen = gen * std + mean
        ref = ref * std + mean
        samples.append(gen)
        refs.append(ref)

        if step == 0 and opt.visualize_count > 0:
            n = min(opt.visualize_count, gen.shape[0])
            visualize_pointcloud_batch(str(syn_dir / "samples_preview.png"), gen[:n])
            visualize_pointcloud_batch(str(syn_dir / "refs_preview.png"), ref[:n])

        if remaining is not None:
            remaining -= x.shape[0]
            if remaining <= 0:
                break

    sample_pcs = torch.cat(samples, dim=0).contiguous()
    ref_pcs = torch.cat(refs, dim=0).contiguous()
    torch.save(sample_pcs, syn_dir / "samples.pth")
    torch.save(ref_pcs, syn_dir / "refs.pth")
    logger.info(f"saved samples={tuple(sample_pcs.shape)} refs={tuple(ref_pcs.shape)}")
    return sample_pcs, ref_pcs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="./eval_output")
    p.add_argument("--experiment_name", type=str, default="dit3d_eval")

    p.add_argument("--dataroot", default="ShapeNetCore.v2.PC15k/")
    p.add_argument("--category", default="chair")
    p.add_argument("--eval-split", choices=("val", "test"), default="val")
    p.add_argument("--num_classes", type=int, default=55)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--npoints", default=2048, type=int)
    p.add_argument("--voxel_size", type=int, choices=[16, 32, 64], default=32)
    p.add_argument("--max-eval-shapes", type=int, default=0)
    p.add_argument("--visualize-count", type=int, default=16)

    p.add_argument("--model_type", type=str, choices=list(DiT3D_models.keys()), default="DiT-S/4")
    p.add_argument("--window_size", type=int, default=0)
    p.add_argument("--window_block_indexes", type=lambda s: tuple(int(x) for x in s.split(",")), default=(0, 3, 6, 9))
    p.add_argument("--loss_type", default="mse")
    p.add_argument("--model_mean_type", default="eps")
    p.add_argument("--model_var_type", default="fixedsmall")
    p.add_argument("--beta_start", type=float, default=0.0001)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--schedule_type", default="linear")
    p.add_argument("--time_num", type=int, default=1000)

    p.add_argument("--ss", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--encoder-depth", type=int, default=8)
    p.add_argument("--proj-dims", nargs="*", type=int, default=[2048, 768])
    p.add_argument("--proj-adaln", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--proj-bn", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--proj-coeff", type=float, default=0.5)
    p.add_argument("--proj-triu-coeff", type=float, default=0.025)
    p.add_argument("--proj-global-batch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--proj-single-t-batch", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--model", required=True, help="checkpoint path (.pth) to evaluate")
    p.add_argument(
        "--rand_cls",
        "--rand-cls",
        dest="rand_cls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use random conditioning labels instead of dataset category labels.",
    )
    p.add_argument("--manualSeed", default=0, type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--metrics", choices=("cd", "emd", "both"), default="both")
    p.add_argument("--metric-batch-size", type=int, default=64)
    p.add_argument("--metric-sample-batch-size", type=int, default=1)
    p.add_argument(
        "--strict-dataset-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require canonical ShapeNet train/eval counts for known categories.",
    )
    return p.parse_args()


def main() -> None:
    opt = parse_args()
    if opt.category == "airplane":
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = "warm0.1"

    set_seed(opt.manualSeed)
    output_dir = Path(get_output_dir(opt.model_dir, opt.experiment_name))
    copy_source(__file__, str(output_dir))
    logger = setup_logging(str(output_dir))
    logger.info(opt)

    device = torch.device(opt.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(args=opt, betas=betas).to(device)
    state = torch.load(opt.model, map_location="cpu")
    model_state = _normalize_state_dict_for_single_gpu(state["model_state"])
    model.load_state_dict(model_state, strict=True)
    logger.info(f"loaded checkpoint={opt.model}")

    sample_pcs, ref_pcs = generate_samples(opt, model, device, output_dir, logger)
    logger.info("computing point-cloud metrics")
    metrics = compute_all_metrics(
        sample_pcs,
        ref_pcs,
        metrics=opt.metrics,
        ref_batch_size=opt.metric_batch_size,
        sample_batch_size=opt.metric_sample_batch_size,
        device=device,
    )
    metadata = {
        "checkpoint": str(opt.model),
        "category": opt.category,
        "eval_split": opt.eval_split,
        "ss": bool(opt.ss),
        "metrics_backend": "pointcloud_metrics",
        "metric_batch_size": opt.metric_batch_size,
        "metric_sample_batch_size": opt.metric_sample_batch_size,
        "max_eval_shapes": opt.max_eval_shapes,
        "rand_cls": bool(opt.rand_cls),
        "beta_start": opt.beta_start,
        "beta_end": opt.beta_end,
        "schedule_type": opt.schedule_type,
        "train_shape_count": opt.train_shape_count,
        "eval_shape_count": opt.eval_shape_count,
        "manualSeed": opt.manualSeed,
    }
    _write_metric_outputs(output_dir, metrics, metadata)
    for key in sorted(metrics):
        logger.info(f"  {key:>18s} : {metrics[key]:.8g}")


if __name__ == "__main__":
    main()
