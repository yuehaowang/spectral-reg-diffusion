import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import json
import logging
import math
import re
import time
from datetime import datetime

import accelerate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from config import get_config_argparse
from dataset import load_dataset, resolve_dataset_path
from evaluation.evaluator import compute_metrics
from models.model_utils import create_dit_model
from sampler import euler_sampler
from torch.utils.data import DataLoader, TensorDataset
from utils import (
    load_checkpoint,
    make_grid_uint8,
    save_image,
    save_npz,
    StatsDict,
)

LoggerFileHandler = logging.FileHandler


def generation(config, data_info, accelerator, model, ckpt_iter, logger):
    eval_dir = os.path.join(config.log_dir, "eval")
    if accelerator.is_main_process:
        os.makedirs(eval_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Include sampler_cfg in results_dir name
    results_dir_base = f"gen_{'ema' if config.reload_ema else 'online'}_{ckpt_iter:08d}_nsteps{config.sampler_max_steps}_cfg{config.sampler_cfg}"
    # Count existing results directories with the same base pattern
    n_generations = len(
        [
            d
            for d in (os.listdir(eval_dir) if os.path.isdir(eval_dir) else [])
            if d.startswith(results_dir_base)
        ]
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Create unique results directory with generation count
        results_dir = os.path.join(eval_dir, f"{results_dir_base}_{n_generations:02d}")

        os.makedirs(results_dir, exist_ok=True)

        # Write config to JSON
        with open(os.path.join(results_dir, "cfg.json"), "w") as f:
            json.dump(vars(config), f, indent=4)
    accelerator.wait_for_everyone()

    device = accelerator.device

    sampler = euler_sampler

    model.eval()

    # Initialize x0 and class labels
    n_total_samples = (
        math.ceil(config.eval_sample_num / accelerator.num_processes)
        * accelerator.num_processes
    )
    n_samples_per_proc = n_total_samples // accelerator.num_processes
    x0_eval = torch.randn(
        n_samples_per_proc,
        data_info.image_channels,
        data_info.image_size,
        data_info.image_size,
    )
    if data_info.num_classes <= 0:
        # if no class labels, use -1 as a dummy value
        cls_eval = torch.ones((n_samples_per_proc,)).long() * -1
    else:
        cls_eval = torch.randint(0, data_info.num_classes, (n_samples_per_proc,))

    x1_eval_cond_ls = []
    x1_eval_uncond_ls = []

    # Create dataloader
    dataset = TensorDataset(x0_eval, cls_eval)
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # Recorders for logging
    stats_dict = StatsDict()

    cur_iter = 0
    total_iters = len(dataloader)

    logger.info(f"Start generation")

    for batch in dataloader:
        x0, cond = batch
        x0 = x0.to(device)
        cond = cond.to(device)

        if accelerator.is_main_process:
            t0 = time.time()

        with torch.no_grad():
            x1_eval_cond = None
            x1_eval_uncond = None

            ## Conditional generation
            if config.cond:
                # unconditional class label: last class index + 1
                uncond = torch.ones_like(cond) * data_info.num_classes

                x1_eval_cond = sampler(
                    model,
                    x0,
                    cond,
                    uncond,
                    num_steps=config.sampler_max_steps,
                    cfg=config.sampler_cfg,
                    sampler_min_t=config.sampler_min_t,
                    sampler_max_t=config.sampler_max_t,
                )

                x1_eval_cond = ((x1_eval_cond + 1.0) / 2.0).clamp(0, 1.0)

                cond = uncond
            else:
                cond = None

            if not config.no_eval_uncond:
                ## Unconditional generation
                uncond = None
                x1_eval_uncond = sampler(
                    model,
                    x0,
                    cond,
                    uncond,
                    num_steps=config.sampler_max_steps,
                    cfg=config.sampler_cfg,
                    sampler_min_t=config.sampler_min_t,
                    sampler_max_t=config.sampler_max_t,
                )

                x1_eval_uncond = ((x1_eval_uncond + 1.0) / 2.0).clamp(0, 1.0)

        if accelerator.is_main_process:
            elapsed_t = time.time() - t0
            stats_dict["elapsed_ts"].append(elapsed_t)

        if not config.no_eval_uncond:
            x1_eval_uncond_ls.append(x1_eval_uncond)
        if config.cond:
            x1_eval_cond_ls.append(x1_eval_cond)

        """
        Logging
        """
        if accelerator.is_main_process:
            if config.log_every > 0 and (cur_iter + 1) % config.log_every == 0:
                logger.info(
                    f"Iter {cur_iter + 1}/{total_iters}: AvgTime={np.mean(stats_dict['elapsed_ts']).item():.3f}"
                )
                stats_dict["elapsed_ts"] = []

        cur_iter += 1

    if not config.no_eval_uncond:
        all_x1_eval_uncond = torch.concat(x1_eval_uncond_ls, dim=0)
        all_x1_eval_uncond = accelerator.gather(all_x1_eval_uncond)
        all_x1_eval_uncond = all_x1_eval_uncond[: config.eval_sample_num]
    if config.cond:
        all_x1_eval_cond = torch.concat(x1_eval_cond_ls, dim=0)
        all_x1_eval_cond = accelerator.gather(all_x1_eval_cond)
        all_x1_eval_cond = all_x1_eval_cond[: config.eval_sample_num]

    if accelerator.is_main_process:
        if not config.no_eval_uncond:
            all_x1_eval_uncond = all_x1_eval_uncond.cpu()
            # Save numpy array
            data_to_save = (
                all_x1_eval_uncond.permute(0, 2, 3, 1).numpy() * 255
            ).astype(np.uint8)
            npz_path = os.path.join(
                results_dir,
                f"gen_uncond_sample_N{config.eval_sample_num}.npz",
            )
            save_npz(npz_path, arr_0=data_to_save)

            # Save image
            image_path = os.path.join(
                results_dir,
                f"gen_uncond_sample_N{config.eval_sample_num}.png",
            )
            image_data = make_grid_uint8(all_x1_eval_uncond[: config.eval_vis_num])
            save_image(image_path, image_data)

        if config.cond:
            all_x1_eval_cond = all_x1_eval_cond.cpu()
            # Save numpy array
            data_to_save = (all_x1_eval_cond.permute(0, 2, 3, 1).numpy() * 255).astype(
                np.uint8
            )
            npz_path = os.path.join(
                results_dir,
                f"gen_cond_sample_N{config.eval_sample_num}.npz",
            )
            save_npz(npz_path, arr_0=data_to_save)

            # Save image
            image_path = os.path.join(
                results_dir,
                f"gen_cond_sample_N{config.eval_sample_num}.png",
            )
            image_data = make_grid_uint8(all_x1_eval_cond[: config.eval_vis_num])
            save_image(image_path, image_data)

        # Compute metrics if requested
        if config.compute_metrics:
            if not config.ref_batch_path:
                logger.warning(
                    "Metrics computation requested but no reference batch path provided. Skipping metrics computation."
                )
            else:
                logger.info("Computing metrics...")

                # Compute metrics for unconditional samples if available
                if not config.no_eval_uncond:
                    uncond_npz_path = os.path.join(
                        results_dir,
                        f"gen_uncond_sample_N{config.eval_sample_num}.npz",
                    )
                    try:
                        compute_metrics(config.ref_batch_path, uncond_npz_path)
                        logger.info(
                            f"Metrics computed for unconditional samples and saved alongside {uncond_npz_path}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to compute metrics for unconditional samples: {e}"
                        )

                # Compute metrics for conditional samples if available
                if config.cond:
                    cond_npz_path = os.path.join(
                        results_dir,
                        f"gen_cond_sample_N{config.eval_sample_num}.npz",
                    )
                    try:
                        compute_metrics(config.ref_batch_path, cond_npz_path)
                        logger.info(
                            f"Metrics computed for conditional samples and saved alongside {cond_npz_path}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to compute metrics for conditional samples: {e}"
                        )


def main():
    argparser = get_config_argparse()
    argparser.add_argument(
        "--output_path", type=str, default="", help="Output path for generation."
    )
    argparser.add_argument(
        "--no_eval_uncond",
        action="store_true",
        help="If True, disable unconditional generation.",
    )
    argparser.add_argument(
        "--eval_batch", type=int, default=256, help="Evaluation batch size."
    )
    argparser.add_argument(
        "--eval_vis_num",
        type=int,
        default=64,
        help="Number of generated images to visualize.",
    )
    argparser.add_argument(
        "--eval_all_ckpts",
        action="store_true",
        help="If True, evaluate all checkpoints (--expname should be specified). Otherwise, only evaluate the last one.",
    )
    argparser.add_argument(
        "--eval_ckpt_iters",
        type=int,
        nargs="*",
        default=[-1],
        help="Iterations of checkpoints to evaluate.",
    )
    argparser.add_argument(
        "--reload_ema",
        action="store_true",
        help="If True, reload EMA weights from checkpoint. Otherwise, reload model weights.",
    )
    argparser.add_argument(
        "--compute_metrics",
        action="store_true",
        help="If True, compute metrics (FID, IS, etc.) after generation.",
    )
    argparser.add_argument(
        "--ref_batch_path",
        type=str,
        default="",
        help="Path to reference batch npz file for metrics computation.",
    )

    config = argparser.parse_args()

    accelerator = Accelerator(mixed_precision="no")

    if config.seed >= 0:
        set_seed(config.seed + accelerator.process_index)

    if not config.expname and not config.output_path:
        raise ValueError("Either --output_path or --expname should be specified")
    elif config.expname:
        config.log_dir = os.path.join(config.base_dir, config.expname)
        if not os.path.isdir(config.log_dir):
            raise ValueError(f"{config.log_dir} does not exist")
    else:
        config.log_dir = config.output_path
        if not config.ckpt_path:
            raise ValueError("--ckpt_path should be specified")
    if accelerator.is_main_process:
        os.makedirs(config.log_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Setup logger
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if accelerator.is_main_process:
        log_handlers += [
            LoggerFileHandler(
                os.path.join(
                    config.log_dir,
                    f'log_generate_{datetime.now().strftime("%y%m%d%H%M%S")}.txt',
                )
            )
        ]
    logging.basicConfig(
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=log_handlers,
        level=logging.INFO,
    )
    logger = accelerate.logging.get_logger(__name__)
    logger.info(config)
    logger.info(accelerator.state, main_process_only=False)

    # Build dataset
    config.dataset_path = resolve_dataset_path(config, no_loading=True)
    data_info = load_dataset(config, no_loading=True)
    logger.info(
        f"image size: {data_info.image_size}, image channels: {data_info.image_channels}, "
        f"# classes: {data_info.num_classes if config.cond else 0}"
    )

    # Reload checkpoint paths
    ckpt_paths = []
    if config.ckpt_path:
        ckpt_paths.append(config.ckpt_path)
    else:
        ckpt_dir = os.path.join(config.log_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            # Get all checkpoint files
            all_files = os.listdir(ckpt_dir)
            all_ckpts = sorted(
                [os.path.join(ckpt_dir, f) for f in all_files if f.endswith(".pt")]
            )
            if config.eval_all_ckpts:
                ckpts_to_reload = all_ckpts
            else:
                ckpt_iters_dict = {-1: all_ckpts[-1]} if all_ckpts else {}
                for ckpt_path in all_ckpts:
                    _m = re.search(r"ckpt_(\d+)\.pt$", ckpt_path)
                    if _m:
                        ckpt_iter = int(_m.group(1))
                        ckpt_iters_dict[ckpt_iter] = ckpt_path

                ckpts_to_reload = [
                    ckpt_iters_dict[i]
                    for i in config.eval_ckpt_iters
                    if i in ckpt_iters_dict
                ]

            ckpt_paths += ckpts_to_reload

    if not ckpt_paths:
        logger.info("No checkpoint found.")
        exit(0)

    logger.info(f"{len(ckpt_paths)} checkpoints to evaluate: {', '.join(ckpt_paths)}")

    for ckpt_path in ckpt_paths:
        # Build model
        model = create_dit_model(config, data_info)

        logger.info(f"Load checkpoint: {ckpt_path}")
        if config.reload_ema:
            ckpt_iter = load_checkpoint(ckpt_path, ema=model)
        else:
            ckpt_iter = load_checkpoint(ckpt_path, model=model)

        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"# parameters: {model_size}, {model_size / 1e6}M")

        # Prepare models for training
        model = accelerator.prepare(model)

        generation(config, data_info, accelerator, model, ckpt_iter, logger)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            logger.info(f"Completed generation for checkpoint: {ckpt_path}")

    logger.info("Generation completed.")


if __name__ == "__main__":
    main()
