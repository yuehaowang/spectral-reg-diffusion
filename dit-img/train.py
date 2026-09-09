import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import json
import logging
import time
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from functools import partial

import accelerate
import imageio
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed

from config import get_config_argparse
from dataset import load_dataset, resolve_dataset_path
from loss_fn import rf_loss_fn, ss_repl_dispersive_rf_loss_fn, ss_repl_rf_loss_fn
from models.model_utils import create_dit_model
from sampler import euler_sampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    load_checkpoint,
    make_grid_uint8,
    save_checkpoint,
    save_image,
    StatsDict,
)

LoggerFileHandler = logging.FileHandler


def generation(
    config, data_info, device, model, x0=None, cond=None, sampler=euler_sampler
):
    model.eval()
    with torch.no_grad():
        if x0 is None:
            x0 = torch.randn(
                config.eval_sample_num,
                data_info.image_channels,
                data_info.image_size,
                data_info.image_size,
            ).to(device)

        x1_eval_cond = None

        ## Conditional generation
        if config.cond:
            if not cond:
                cond = torch.arange(0, x0.shape[0]).to(device) % data_info.num_classes
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

    # output are normalized into [0, 1]
    return (x1_eval_uncond, x1_eval_cond)


@torch.no_grad()
def update_ema(ema, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def train(
    config,
    data_info,
    accelerator,
    train_dataloader,
    model,
    ema,
    optimizer,
    start_iter,
    logger,
):
    ckpt_dir = os.path.join(config.log_dir, "checkpoints")
    trainvis_dir = os.path.join(config.log_dir, "train_vis")
    # traineval_dir = os.path.join(config.log_dir, 'train_eval')
    tblogs_dir = os.path.join(config.log_dir, "tb_logs")

    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(trainvis_dir, exist_ok=True)
        os.makedirs(tblogs_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_log_dir = os.path.join(tblogs_dir, timestamp)
        os.makedirs(tb_log_dir, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=tb_log_dir)

    accelerator.wait_for_everyone()

    device = accelerator.device

    # Set sampler & loss function
    if config.ssrepl:
        if config.ssrepl_loss_type == "spectral":
            loss_fn = partial(
                ss_repl_rf_loss_fn,
                psi_scale=1,
                single_t=config.ssrepl_single_t,
                use_global_batch=config.ssrepl_global_batch,
                half_batch=config.ssrepl_half_batch,
            )
        elif config.ssrepl_loss_type == "dispersive":
            loss_fn = partial(
                ss_repl_dispersive_rf_loss_fn,
                tau=0.5,
            )
    else:
        loss_fn = rf_loss_fn
    test_sampler = euler_sampler

    # Initialize noise latent for evaluation
    x0_eval = None
    if config.eval_fixed_x0:
        x0_eval = torch.randn(
            config.eval_sample_num,
            data_info.image_channels,
            data_info.image_size,
            data_info.image_size,
        )

    # Create data loader
    dataloader_iter = iter(train_dataloader)

    # Recorders for logging
    stats_dict = StatsDict()

    logger.info(f"Start training from iter {start_iter}")

    for cur_iter in range(start_iter, config.max_iters):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_dataloader)
            batch = next(dataloader_iter)

        model.train()

        x1, y = batch
        x1 = x1
        if not config.cond:
            y = None

        if accelerator.is_main_process:
            t0 = time.time()

        losses = loss_fn(x1, y, model)
        if config.ssrepl:
            # Determine which loss components to apply based on iteration
            mse_start, mse_end = config.ssrepl_mse_iters
            reg_start, reg_end = config.ssrepl_reg_iters
            apply_mse = mse_start <= cur_iter and (mse_end == -1 or cur_iter < mse_end)
            apply_reg = reg_start <= cur_iter and (reg_end == -1 or cur_iter < reg_end)

            loss = 0.0
            if apply_mse:
                loss += losses["mse"]
            if apply_reg:
                loss += config.ssrepl_alpha * (
                    losses["reg_diag"] + config.ssrepl_lambda * losses["reg_triu"]
                )

            # Ensure we have at least one loss component
            if not apply_mse and not apply_reg:
                logger.warning(
                    f"No loss components applied at iteration {cur_iter + 1}. Using MSE loss as fallback."
                )
                loss = losses["mse"]
        else:
            loss = losses["mse"]

        accelerator.backward(loss)
        if accelerator.sync_gradients and config.max_grad_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            update_ema(ema, model, config.ema_decay)

        if accelerator.is_main_process:
            elapsed_t = time.time() - t0
            stats_dict["elapsed_ts"].append(elapsed_t)

        """
        Logging & checkpointing
        """
        loss = accelerator.gather(loss.detach()).mean().item()
        for k in losses:
            losses[k] = accelerator.gather(losses[k].detach()).mean().item()

        if accelerator.is_main_process:
            # Tensorboard logging
            summary_writer.add_scalar("train/total_loss", loss, cur_iter + 1)
            summary_writer.add_scalar("train/mse_loss", losses["mse"], cur_iter + 1)
            if config.ssrepl:
                summary_writer.add_scalar(
                    "train/reg_diag_loss", losses["reg_diag"], cur_iter + 1
                )
                summary_writer.add_scalar(
                    "train/reg_triu_loss", losses["reg_triu"], cur_iter + 1
                )
                combined_proj_loss = (
                    losses["reg_diag"] + config.ssrepl_lambda * losses["reg_triu"]
                )
                summary_writer.add_scalar(
                    "train/combined_proj_loss", combined_proj_loss, cur_iter + 1
                )

            # Logging
            if config.log_every > 0 and (cur_iter + 1) % config.log_every == 0:
                losses_log_str = ", ".join(
                    f"{name}={value:.3f}" for name, value in losses.items()
                )

                # Add loss component status for ssrepl
                loss_status_str = ""
                if config.ssrepl:
                    mse_start, mse_end = config.ssrepl_mse_iters
                    reg_start, reg_end = config.ssrepl_reg_iters
                    apply_mse = mse_start <= cur_iter and (
                        mse_end == -1 or cur_iter < mse_end
                    )
                    apply_reg = reg_start <= cur_iter and (
                        reg_end == -1 or cur_iter < reg_end
                    )
                    loss_status_str = f", MSE={'ON' if apply_mse else 'OFF'}, REG={'ON' if apply_reg else 'OFF'}"

                logger.info(
                    f"Iter {cur_iter + 1}: Loss={loss:.3f}, AvgTime={np.mean(stats_dict['elapsed_ts']).item():.3f}, {losses_log_str}{loss_status_str}"
                )
                stats_dict["elapsed_ts"] = []

            # Save checkpoints
            if config.ckpt_every > 0 and (cur_iter + 1) % config.ckpt_every == 0:
                logger.info(f"Saving checkpoint ... (iter {cur_iter + 1})")
                save_checkpoint(
                    os.path.join(ckpt_dir, f"ckpt_{cur_iter + 1:08d}.pt"),
                    accelerator.unwrap_model(model),
                    ema,
                    optimizer,
                    config,
                    cur_iter + 1,
                )

        ## Test sampling
        if (
            config.eval_sample_every > 0
            and (cur_iter + 1) % config.eval_sample_every == 0
        ):
            logger.info(f"Evaluation sampling... (iter {cur_iter + 1})")

            if not config.eval_fixed_x0:
                x0_eval = torch.randn(
                    config.eval_sample_num,
                    data_info.image_channels,
                    data_info.image_size,
                    data_info.image_size,
                )

            eval_model_ls = []
            for model_eval_name in config.eval_models:
                if model_eval_name == "online":
                    eval_model_ls.append((model, "online"))
                elif model_eval_name == "ema":
                    eval_model_ls.append((ema, "ema"))

            for model_eval, model_eval_name in eval_model_ls:
                model_eval.eval()

                if accelerator.is_main_process:
                    eval_t0 = time.time()

                with accelerator.split_between_processes(x0_eval) as x0_eval_:
                    x0_eval_ = x0_eval_.to(accelerator.device)
                    x1_eval_uncond, x1_eval_cond = generation(
                        config,
                        data_info,
                        device,
                        model_eval,
                        x0_eval_,
                        sampler=test_sampler,
                    )
                    if x1_eval_uncond is not None and len(x1_eval_uncond) > 0:
                        x1_eval_uncond = accelerator.gather(x1_eval_uncond)
                    if x1_eval_cond is not None and len(x1_eval_cond) > 0:
                        x1_eval_cond = accelerator.gather(x1_eval_cond)

                if accelerator.is_main_process:
                    eval_elapsed_t = time.time() - eval_t0
                    logger.info(
                        f"Sampling time (iter {cur_iter + 1}) of {model_eval_name} model: {eval_elapsed_t:0.3f}s"
                    )

                    if x1_eval_uncond is not None and len(x1_eval_uncond) > 0:
                        x1_eval_uncond = x1_eval_uncond.cpu()
                        # torch.save(x1_eval_uncond, os.path.join(traineval_dir, f'gen_uncond_sample_{cur_iter + 1:08d}.pt'))
                        x1_eval_uncond_vis = make_grid_uint8(x1_eval_uncond)
                        save_image(
                            os.path.join(
                                trainvis_dir,
                                f"gen_{model_eval_name}_uncond_sample_{cur_iter + 1:08d}.png",
                            ),
                            x1_eval_uncond_vis,
                        )
                    if x1_eval_cond is not None and len(x1_eval_cond) > 0:
                        x1_eval_cond = x1_eval_cond.cpu()
                        # torch.save(x1_eval_cond, os.path.join(traineval_dir, f'gen_cond_sample_{cur_iter + 1:08d}.pt'))
                        x1_eval_cond_vis = make_grid_uint8(x1_eval_cond)
                        save_image(
                            os.path.join(
                                trainvis_dir,
                                f"gen_{model_eval_name}_cond_sample_{cur_iter + 1:08d}.png",
                            ),
                            x1_eval_cond_vis,
                        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Saving final checkpoint...")
        save_checkpoint(
            os.path.join(ckpt_dir, f"last_ckpt_{config.max_iters:08d}.pt"),
            accelerator.unwrap_model(model),
            ema,
            optimizer,
            config,
            config.max_iters,
        )

    logger.info("Training completed.")
    accelerator.end_training()


def main():
    argparser = get_config_argparse()

    # Add iteration control arguments for ssrepl
    argparser.add_argument(
        "--ssrepl_reg_iters",
        type=int,
        nargs=2,
        default=[0, -1],
        metavar=("START", "END"),
        help="Start and end iterations for regularizing loss in ssrepl. Use -1 for end to apply until the end.",
    )
    argparser.add_argument(
        "--ssrepl_mse_iters",
        type=int,
        nargs=2,
        default=[0, -1],
        metavar=("START", "END"),
        help="Start and end iterations for MSE loss in ssrepl. Use -1 for end to apply until the end.",
    )

    config = argparser.parse_args()

    accelerator = Accelerator(mixed_precision="no")

    if config.seed >= 0:
        set_seed(config.seed + accelerator.process_index)

    if not config.expname:
        raise ValueError("--expname should not be empty")
    config.log_dir = os.path.join(config.base_dir, config.expname)
    if accelerator.is_main_process:
        # Create output directories
        os.makedirs(config.base_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Write config to JSON
        with open(os.path.join(config.log_dir, "cfg.json"), "w") as f:
            json.dump(vars(config), f, indent=4)
    accelerator.wait_for_everyone()

    # Setup logger
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if accelerator.is_main_process:
        log_handlers += [
            LoggerFileHandler(
                os.path.join(
                    config.log_dir,
                    f'log_train_{datetime.now().strftime("%y%m%d%H%M%S")}.txt',
                )
            ),
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
    config.dataset_path = resolve_dataset_path(config)
    data_info = load_dataset(config)
    train_data = data_info.train_data
    train_dataloader = DataLoader(train_data, batch_size=config.batch, shuffle=True)
    logger.info(
        f"# training images: {len(train_data)}, image size: {data_info.image_size}, image channels: {data_info.image_channels}, "
        f"# classes: {data_info.num_classes if config.cond else 0}"
    )

    # Build model
    model = create_dit_model(config, data_info)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"# parameters: {model_size}, {model_size / 1e6}M")

    # EMA copy
    ema = deepcopy(model).eval().requires_grad_(False)
    update_ema(ema, model, decay=0)
    ema = ema.to(accelerator.device)

    # Build optimizer
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=config.lr)

    # Reload model & optimizer
    start_iter = 0
    if config.no_reload:
        logger.info("Checkpoint reloading disabled.")
    else:
        path = ""
        if config.ckpt_path:
            path = config.ckpt_path
        else:
            ckpt_dir = os.path.join(config.log_dir, "checkpoints")
            if os.path.isdir(ckpt_dir):
                ckpt_list = sorted(os.listdir(ckpt_dir))
                path = os.path.join(ckpt_dir, ckpt_list[-1]) if ckpt_list else ""

        if not path:
            logger.info("No checkpoint found.")
        else:
            logger.info(f"Load checkpoint: {path}")
            start_iter = load_checkpoint(path, model, ema, optimizer)

    # Prepare models for training
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    train(
        config,
        data_info,
        accelerator,
        train_dataloader,
        model,
        ema,
        optimizer,
        start_iter,
        logger,
    )


if __name__ == "__main__":
    main()
