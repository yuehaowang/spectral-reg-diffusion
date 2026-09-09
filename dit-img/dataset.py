import json
import os
import zipfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

import PIL.Image

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Subset
from torchvision.datasets.utils import extract_archive

try:
    import pyspng
except ImportError:
    pyspng = None


@dataclass
class ImageDataInfo:
    image_size: int  # image resolution
    image_channels: int  # image channels
    num_classes: int  # number of classes
    train_data: Optional[torch.utils.data.Dataset]  # Training dataset
    test_data: Optional[torch.utils.data.Dataset]  # Test dataset


class ZipImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        supported_ext = {".png", ".jpg", ".jpeg", ".npy"}

        self.imgs_zip_path = data_path

        self.images_zipfile = None

        self.image_transform = transform

        print("Loading all files from zip archives")

        # images
        self.image_fnames = sorted(
            fname
            for fname in set(self._get_imgs_zipfile().namelist())
            if self._file_ext(fname) in supported_ext
        )

        # labels
        fname = "dataset.json"
        with self._get_imgs_zipfile().open(fname, "r") as f:
            labels = json.load(f)["labels"]
        if labels is None:
            self.labels = dict()
            self.num_classes = 0
        else:
            labels = dict(labels)
            labels = [labels[fname.replace("\\", "/")] for fname in self.image_fnames]
            labels = np.array(labels)
            self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

            self.num_classes = len(np.unique(self.labels))

        # trick: close files and reopen in __getitem__ to prevent
        # errors when using DataLoader with multiple workers
        self._close_imgs_zipfile()

    def _get_imgs_zipfile(self):
        if self.images_zipfile is None:
            self.images_zipfile = zipfile.ZipFile(self.imgs_zip_path)
        return self.images_zipfile

    def _close_imgs_zipfile(self):
        try:
            if self.images_zipfile is not None:
                self.images_zipfile.close()
        finally:
            self.images_zipfile = None

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_ext = self._file_ext(image_fname)

        with self._get_imgs_zipfile().open(image_fname, "r") as f:
            if image_ext == ".npy":
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
                image = torch.from_numpy(image)
            # elif image_ext == ".png" and pyspng is not None:
            #     image = pyspng.load(f.read())
            #     image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                pil_im = PIL.Image.open(f)
                if self.image_transform is not None:
                    image = self.image_transform(pil_im)
                else:
                    image = pil_im
                # else:
                #     image = np.array(pil_im)
                #     image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
                #     image = torch.from_numpy(image)
                if isinstance(image, PIL.Image.Image):
                    image = torch.from_numpy(np.array(image))  # HWC, [0, 255]

        return (
            image,
            torch.tensor(self.labels[idx]) if len(self.labels) else -1,
        )


def downsample_dataset(dataset, downsample_rate, seed=0):
    """
    Downsample a dataset by randomly selecting a subset of indices.

    Args:
        dataset: PyTorch dataset to downsample
        downsample_rate: Float between 0 and 1, fraction of data to keep
        seed: Random seed for reproducible downsampling

    Returns:
        Subset of the original dataset
    """
    if downsample_rate >= 1.0:
        return dataset

    if downsample_rate <= 0.0:
        raise ValueError("Downsample rate must be greater than 0")

    # Set random seed for reproducible downsampling
    rng = np.random.RandomState(seed)

    # Calculate number of samples to keep
    total_samples = len(dataset)
    num_samples = int(total_samples * downsample_rate)

    # Randomly select indices
    indices = rng.choice(total_samples, size=num_samples, replace=False)
    indices = sorted(indices)  # Sort for deterministic behavior

    return Subset(dataset, indices)


def resolve_dataset_path(config, no_loading=False):
    """
    Resolve the local (POSIX) dataset path.

    Datasets are read directly from the local filesystem:
      - cifar10: a directory; torchvision downloads it there on first use
        (``download=True`` in :func:`load_dataset`).
      - celeba:  a directory containing ``celeba/``; the aligned-image archive is
        extracted in place if present.
      - zip:     a path to the dataset ``.zip`` archive.
    """
    if config.dataset_type == "celeba" and not no_loading:
        zip_path = os.path.join(config.dataset_path, "celeba", "img_align_celeba.zip")
        if os.path.exists(zip_path):
            extract_archive(zip_path)
    return config.dataset_path


def load_dataset(config, no_loading=False):
    if config.dataset_type == "cifar10":
        ## CIFAR10
        num_ch = 3
        num_cls = 10

        if not no_loading:
            transform_ops = transforms.Compose(
                [
                    transforms.Resize(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [
                            0.5,
                        ]
                        * num_ch,
                        [
                            0.5,
                        ]
                        * num_ch,
                    ),
                ]
            )
            trainset = torchvision.datasets.CIFAR10(
                root=config.dataset_path,
                download=True,
                transform=transform_ops,
                train=True,
            )
        else:
            trainset = None
        testset = None

    elif config.dataset_type == "celeba":
        ## CelebA
        """
        # 0: 5 o'clock shadow
        # 15: Eyeglasses
        # 20: Male
        # 31: Smiling'
        """
        attrs_list = [0, 15, 20, 31]

        num_ch = 3
        num_cls = 2 ** len(attrs_list)

        if not no_loading:
            target_transform = lambda attrs: torch.tensor(
                int("".join(map(str, attrs[attrs_list].tolist())), 2)
            )

            train_kwargs = dict(
                split="train", target_type="attr", target_transform=target_transform
            )

            transform = [
                transforms.Resize(config.image_size),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [
                        0.5,
                    ]
                    * num_ch,
                    [
                        0.5,
                    ]
                    * num_ch,
                ),
            ]
            transform = transforms.Compose(transform)

            trainset = torchvision.datasets.CelebA(
                root=config.dataset_path,
                download=False,
                transform=transform,
                **train_kwargs,
            )
        else:
            trainset = None
        testset = None
    elif config.dataset_type == "zip":
        transform_ops = transforms.Compose(
            [
                transforms.Resize(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [
                        0.5,
                    ]
                    * 3,
                    [
                        0.5,
                    ]
                    * 3,
                ),
            ]
        )
        trainset = ZipImageFolderDataset(config.dataset_path, transform_ops)
        testset = None

        img0 = trainset[0][0]
        num_ch = img0.shape[0]
        # num_cls = len(set(trainset.labels.keys()))
        num_cls = trainset.num_classes
    else:
        raise NotImplementedError()

    # Apply downsampling if specified in config
    if (
        trainset is not None
        and hasattr(config, "downsample_rate")
        and config.downsample_rate < 1.0
    ):
        print(
            f"Downsampling training dataset from {len(trainset)} to {int(len(trainset) * config.downsample_rate)} samples"
        )
        trainset = downsample_dataset(
            trainset, config.downsample_rate, seed=getattr(config, "seed", 0)
        )

    if (
        testset is not None
        and hasattr(config, "downsample_rate")
        and config.downsample_rate < 1.0
    ):
        print(
            f"Downsampling test dataset from {len(testset)} to {int(len(testset) * config.downsample_rate)} samples"
        )
        testset = downsample_dataset(
            testset, config.downsample_rate, seed=getattr(config, "seed", 0)
        )

    return ImageDataInfo(
        image_size=config.image_size,
        image_channels=num_ch,
        num_classes=num_cls,
        train_data=trainset,
        test_data=testset,
    )
