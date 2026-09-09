import os
import random
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR10, CelebA
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def save_batch_as_grid(batch: np.ndarray,
                       save_path: str,
                       grid_size: tuple[int,int] = (8, 8)):
    """
    batch: uint8 array of shape (N, H, W, 3), values in [0,255]
    grid_size: (rows, cols), rows*cols >= N
    """
    N, H, W, C = batch.shape
    rows, cols = grid_size
    assert C == 3, "Expect 3 channels"
    assert rows * cols >= N, f"Grid {rows}x{cols} too small for {N} images"

    # make a blank canvas
    grid_img = Image.new("RGB", (cols * W, rows * H))

    for idx in range(N):
        row = idx // cols
        col = idx % cols
        img = Image.fromarray(batch[idx], mode="RGB")
        grid_img.paste(img, (col * W, row * H))

    grid_img.save(save_path)
    print(f"Saved grid {rows}x{cols}: {save_path}")

def save_random_dataset_samples_npz(
    output_path: str,
    num_samples: int,
    data_root: str,
    data_type: str,
    seed: int,
    image_size: int,
    train: bool = True
):
    """
    Samples `num_samples` random images from the dataset and saves them
    in an .npz file under the key 'arr_0', with shape (N, H, W, C)
    and dtype uint8 in [0,255].
    """
    # Set seeds for full reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Load CIFAR10 / CelebA
    # os.makedirs(data_root, exist_ok=True)
    if data_type == 'cifar10':
        transform_ops = transforms.Compose([
            transforms.Resize(image_size)
        ])
        dataset = CIFAR10(root=data_root, train=train, download=False, transform=transform_ops)
    elif data_type == 'celeba':
        transform_ops = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ])
        dataset = CelebA(root=data_root, download=False, transform=transform_ops)
    else:
        raise ValueError(f'Dataset type {data_type} not supported')

    all_data = []
    for imgs, _ in dataset:
        all_data.append(np.array(imgs))
    all_data = np.stack(all_data, axis=0)

    print(all_data.shape)

    # Sample without replacement
    total = len(all_data)
    indices = random.sample(range(total), num_samples)
    
    # Slice out the selected images
    sampled = all_data[indices]  # shape (num_samples, 32, 32, 3)
    
    # Save to .npz with key 'arr_0'
    np.savez(output_path, arr_0=sampled)
    print(f"Saved {num_samples} images to '{output_path}' under key 'arr_0'")

    vis_file_path = Path(output_path)
    vis_file_path = str(vis_file_path.with_name(vis_file_path.stem + "_vis").with_suffix('.png'))
    save_batch_as_grid(sampled[:64], vis_file_path, (8, 8))


def main():
    parser = argparse.ArgumentParser(
        description="Sample random images and save to NPZ"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory"
    )
    parser.add_argument(
        "--data-type",
        type=str,
        required=True,
        choices=['cifar10', 'celeba'],
        help="Dataset type"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to the output .npz file"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Image size"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of random images to sample"
    )
    args = parser.parse_args()

    save_random_dataset_samples_npz(
        output_path=args.output_path,
        num_samples=args.num_samples,
        data_root=args.data_root,
        data_type=args.data_type,
        seed=args.seed,
        image_size=args.image_size,
    )

if __name__ == "__main__":
    main()