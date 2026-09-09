"""Helper shared by all run_<dataset>.py scripts."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_rf import Config, train_single, samples_from_ckpts  # noqa: E402
from vis import plot_progress_comparison                       # noqa: E402


def train_baseline_and_ours(dist_name, num_epochs, save_dir,
                            bs_baseline=8192, bs_ours=4096,
                            checkpoint_interval=None, device=None):
    """Train the baseline (naive diffusion) and ours (diffusion with spectral
    regularizer) and save a multi-checkpoint progression plot at
    ``save_dir/progress_baseline_vs_ours.png``.

    Default ``checkpoint_interval`` is ``num_epochs // 10`` so the plot has 10
    epoch columns (matching the reference vis style).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_interval is None:
        checkpoint_interval = max(1, num_epochs // 10)
    print(f"=== {dist_name}: device={device}, "
          f"num_epochs={num_epochs}, ckpt_every={checkpoint_interval} ===")

    cfg_baseline = Config(dist_name=dist_name, num_epochs=num_epochs,
                          bs=bs_baseline, checkpoint_interval=checkpoint_interval)
    cfg_ours     = Config(dist_name=dist_name, num_epochs=num_epochs,
                          bs=bs_ours,     checkpoint_interval=checkpoint_interval)

    print("--- training baseline (naive diffusion) ---")
    train_x, _, _ = train_single(cfg_baseline, "mlp",
                                 os.path.join(save_dir, "baseline"), device)
    print("--- training ours (diffusion with spectral regularizer) ---")
    train_single(cfg_ours, "ssmlp", os.path.join(save_dir, "ours"), device)

    return render_progression(dist_name, num_epochs, checkpoint_interval,
                              save_dir, train_x, device)


def render_progression(dist_name, num_epochs, checkpoint_interval,
                       save_dir, training_np, device):
    """Load both models' ckpts at evenly-spaced epochs, generate samples,
    save the progression plot."""
    epochs = list(range(checkpoint_interval, num_epochs + 1, checkpoint_interval))
    baseline_dir = os.path.join(save_dir, "baseline", "ckpts")
    ours_dir     = os.path.join(save_dir, "ours",     "ckpts")
    baseline_samples = samples_from_ckpts(baseline_dir, epochs, "mlp",   device)
    ours_samples     = samples_from_ckpts(ours_dir,     epochs, "ssmlp", device)
    plot_path = os.path.join(save_dir, "progress_baseline_vs_ours.png")
    plot_progress_comparison(training_np, baseline_samples, ours_samples,
                             plot_path, dataset_name=dist_name)
    print(f"saved progression plot: {plot_path}")
    return plot_path
