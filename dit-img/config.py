import argparse
import json

from models.model_utils import MODEL_CONFIGS


def get_config_argparse():
    argparser = argparse.ArgumentParser()
    ## Base
    argparser.add_argument(
        "--base_dir", type=str, default="./output/", help="Base directory."
    )
    argparser.add_argument("--expname", type=str, default="", help="Experiment name.")
    argparser.add_argument("--seed", type=int, default=0, help="Random seed.")

    ## Model
    MODEL_KEYS = list(MODEL_CONFIGS.keys())
    argparser.add_argument(
        "--model_type",
        type=str,
        default=MODEL_KEYS[0],
        choices=MODEL_KEYS,
        help="Model type.",
    )
    argparser.add_argument(
        "--cond", action="store_true", help="If True, generation is conditional."
    )

    argparser.add_argument(
        "--enc_depth", type=int, default=2, help="The DiT depth to apply projector."
    )
    argparser.add_argument(
        "--proj_bn", action="store_true", help="If True, enable bn in projector layers."
    )
    argparser.add_argument(
        "--proj_dims",
        nargs="*",
        default=[256, 256],
        type=int,
        help="Projector hidden dimensions.",
    )
    argparser.add_argument(
        "--proj_adaln", action="store_true", help="If True, enable adaLN in projector."
    )

    ## Dataset
    argparser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset."
    )
    argparser.add_argument(
        "--image_size",
        type=int,
        default=-1,
        help="Image size. If -1 or 0, original image size will be used.",
    )
    argparser.add_argument(
        "--dataset_type",
        type=str,
        default="cifar10",
        choices=["cifar10", "celeba", "zip"],
        help="Type of image dataset.",
    )
    argparser.add_argument(
        "--downsample_rate",
        type=float,
        default=1.0,
        help="Downsampling rate for the dataset. 1.0 means no downsampling, 0.5 means half the data.",
    )

    ## Training
    argparser.add_argument(
        "--max_iters", type=int, default=5000, help="Training iterations."
    )
    argparser.add_argument("--batch", type=int, default=512, help="Batch size.")
    argparser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    argparser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay.")
    argparser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max gradient norm"
    )
    argparser.add_argument(
        "--ssrepl",
        action="store_true",
        help="If True, enable self-supervised representation alignment.",
    )
    argparser.add_argument(
        "--ssrepl_alpha",
        type=float,
        default=0.05,
        help="Weight of the entire representation alignment regularization.",
    )
    argparser.add_argument(
        "--ssrepl_lambda",
        type=float,
        default=0.08,
        help="Weight of the off-diagonal term in representation alignment.",
    )
    argparser.add_argument(
        "--ssrepl_single_t",
        action="store_true",
        help="If True, share the same time step in a batch.",
    )
    argparser.add_argument(
        "--ssrepl_global_batch",
        action="store_true",
        help="If True, compute the loss over global batch in all processes rather than each local batch.",
    )
    argparser.add_argument(
        "--ssrepl_half_batch",
        action="store_true",
        help="If True, compute loss over a half of the batch.",
    )
    argparser.add_argument(
        "--ssrepl_loss_type",
        type=str,
        default="spectral",
        choices=["spectral", "dispersive"],
        help="Type of ss loss function.",
    )

    ## Sampler
    argparser.add_argument(
        "--sampler_cfg", type=float, default=2.0, help="Sampling CFG weight."
    )
    argparser.add_argument(
        "--sampler_max_steps", type=int, default=50, help="Max steps of sampling."
    )
    argparser.add_argument(
        "--sampler_min_t", type=float, default=0, help="Max time of sampling."
    )
    argparser.add_argument(
        "--sampler_max_t", type=float, default=1.0, help="Min time of sampling."
    )

    ## Log & checkpoint
    argparser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Checkpoint path to reload. If empty, load the lastest one.",
    )
    argparser.add_argument(
        "--no_reload",
        action="store_true",
        help="If True, no model and optimizer reloading.",
    )
    argparser.add_argument(
        "--log_every", type=int, default=100, help="Iterations between logging."
    )
    argparser.add_argument(
        "--ckpt_every",
        type=int,
        default=1000,
        help="Iterations between saving checkpoints.",
    )

    ## Eval
    argparser.add_argument(
        "--eval_fixed_x0",
        action="store_true",
        help="If True, use fixed random noise for evaluation.",
    )
    argparser.add_argument(
        "--eval_sample_every",
        type=int,
        default=100,
        help="Iterations between evaluation sampling.",
    )
    argparser.add_argument(
        "--eval_sample_num", type=int, default=64, help="Number of evaluation samples."
    )
    argparser.add_argument(
        "--eval_models",
        type=str,
        nargs="*",
        default=["ema", "online"],
        choices=["ema", "online"],
        help="Models for evaluation.",
    )

    return argparser
