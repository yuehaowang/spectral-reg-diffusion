"""Train baseline (naive diffusion) and ours (diffusion w/ spectral regularizer) on circles."""
import os
from run_exp import train_baseline_and_ours


SAVE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, "output", "circles"))


if __name__ == "__main__":
    train_baseline_and_ours(dist_name="circles", num_epochs=2000, save_dir=SAVE_DIR)
