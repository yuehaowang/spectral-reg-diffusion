"""Multi-checkpoint progression plot:
  Row 0: training data | Baseline (MLP) at each saved epoch
  Row 1: training data | Ours    (SS-MLP) at each saved epoch
Style: red training / blue baseline / green ours, small dots with low alpha.
"""
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 11

POINT_SIZE = 6
POINT_ALPHA = 0.25
PANEL_INCHES = 1.6

COLOR_TRAINING = "#8C2D2D"
COLOR_BASELINE = "#1F3A6E"
COLOR_OURS     = "#1E5A2E"


def _square_frame(training_np, margin=0.10):
    xmin, xmax = training_np[:, 0].min(), training_np[:, 0].max()
    ymin, ymax = training_np[:, 1].min(), training_np[:, 1].max()
    span = max(xmax - xmin, ymax - ymin) * (1.0 + 2.0 * margin)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    return (cx - span / 2.0, cx + span / 2.0), (cy - span / 2.0, cy + span / 2.0)


def _style(ax, xlim, ylim):
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xticks(np.linspace(xlim[0], xlim[1], 5))
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(True, which="major", color="gray", linestyle="-",
            linewidth=0.4, alpha=0.4)


def plot_progress_comparison(training_np, mlp_samples_by_epoch,
                             ssmlp_samples_by_epoch, save_path,
                             dataset_name=None):
    """training_np: (N, 2). *_by_epoch: dict {epoch: (M, 2) array}."""
    epochs = sorted(mlp_samples_by_epoch.keys())
    assert sorted(ssmlp_samples_by_epoch.keys()) == epochs, \
        "MLP and SS-MLP ckpt epochs must match"

    xlim, ylim = _square_frame(training_np)
    n_cols = 1 + len(epochs)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(PANEL_INCHES * n_cols, PANEL_INCHES * 2),
        gridspec_kw={"wspace": 0.05, "hspace": 0.08},
    )

    for r in range(2):
        _style(axes[r, 0], xlim, ylim)
        axes[r, 0].scatter(training_np[:, 0], training_np[:, 1],
                           s=POINT_SIZE * 2, color=COLOR_TRAINING,
                           alpha=0.5, edgecolors="none")
    axes[0, 0].set_title("Training Data", fontsize=10)
    axes[0, 0].set_ylabel("Baseline", fontsize=11)
    axes[1, 0].set_ylabel("Ours", fontsize=11)

    for ci, ep in enumerate(epochs, start=1):
        for r, (samples_dict, color) in enumerate([
            (mlp_samples_by_epoch,   COLOR_BASELINE),
            (ssmlp_samples_by_epoch, COLOR_OURS),
        ]):
            pts = samples_dict[ep]
            ax = axes[r, ci]
            _style(ax, xlim, ylim)
            ax.scatter(pts[:, 0], pts[:, 1],
                       s=POINT_SIZE, color=color, alpha=POINT_ALPHA,
                       edgecolors="none")
            if r == 0:
                ax.set_title(f"ep {ep}", fontsize=10)

    if dataset_name:
        fig.suptitle(f"{dataset_name}: Baseline vs Ours across epochs",
                     y=1.02, fontsize=12)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
