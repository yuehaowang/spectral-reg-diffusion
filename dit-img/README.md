# DiT with spectral representation regularization

Training and evaluation code for rectified-flow DiT baselines and our proposed spectral representation-alignment method, on CIFAR-10, CelebA, FFHQ, and ImageNet-64.

The DiT and RF training code is adapted from [minRF](https://github.com/cloneofsimo/minRF), and the zipped dataset layout follows [EDM](https://github.com/NVlabs/edm).

## Install

```bash
# Python 3.9+ with a PyTorch build matching your CUDA driver (https://pytorch.org).
pip install -r requirements.txt
```

Training and evaluation are launched with `torchrun` on a single multi-GPU node; all reported numbers use eight GPUs.

## Data

CIFAR-10 is downloaded automatically by `torchvision, into whichever directory the script points at.

CelebA and FFHQ are available copies on huggingface:

```bash
# CelebA: torchvision layout -- the scripts take the parent directory of celeba/
hf download --repo-type dataset Yuehao/celeba --local-dir datasets/celeba_root/celeba

# FFHQ: a single EDM-style .zip, read without unpacking -- the scripts take the .zip itself
hf download --repo-type dataset Dmini/FFHQ-64x64 --local-dir datasets/ffhq
```

For ImageNet-64, build one from the ILSVRC2012 training images with the [REPA preprocessing tools](https://github.com/sihyun-yu/REPA/tree/main/preprocessing):

```bash
python dataset_tools.py convert --source=/path/to/imagenet/train \
    --dest=datasets/imagenet-64x64.zip --resolution=64x64 --transform=center-crop-dhariwal
```

## Train

Type the following commands to launch image diffusion model training for DiT baseline and our approach. Each script begins with a short settings block: edit the dataset path, output directory, and experiment name there to match your setup.

```bash
# Baseline (DiT)
bash scripts/train_cifar10.sh
bash scripts/train_celeba.sh
bash scripts/train_ffhq.sh
bash scripts/train_imagenet64.sh

# Ours (spectral-regularized DiT)
bash scripts/train_cifar10_spectral.sh
bash scripts/train_celeba_spectral.sh
bash scripts/train_ffhq_spectral.sh
bash scripts/train_imagenet64_spectral.sh
```

## Evaluate

We release the reference batches we processed for CIFAR-10, CelebA, and FFHQ; ImageNet-64 uses the standard OpenAI guided-diffusion batch.

```bash
hf download Yuehao/spectral-reg-diffusion --include "dit-img/ref-batches/*" --local-dir datasets/refs
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/VIRTUAL_imagenet64_labeled.npz
```

Use the scripts below to conduct image generation evaluation. You will need to adjust file paths there to match your setup.

```bash
# Baseline (DiT)
bash scripts/eval_cifar10.sh
bash scripts/eval_celeba.sh
bash scripts/eval_ffhq.sh
bash scripts/eval_imagenet64.sh

# Ours (spectral-regularized DiT)
bash scripts/eval_cifar10_spectral.sh
bash scripts/eval_celeba_spectral.sh
bash scripts/eval_ffhq_spectral.sh
bash scripts/eval_imagenet64_spectral.sh
```

## Reproduction results

Both baseline and our spectral-regularized models were re-trained and re-evaluated with this codebase using the scripts above, on one node with eight NVIDIA A100 80 GB GPUs. Our re-trained checkpoints are available at [`Yuehao/spectral-reg-diffusion`](https://huggingface.co/Yuehao/spectral-reg-diffusion/tree/main/dit-img). Below are the reproduction results.

| Method | CIFAR-10 (70k iters.) | CelebA (70k iters.) | FFHQ (70k iters.) | ImageNet-64 (100k iters.) |
|---|--:|--:|--:|--:|
| Baseline (DiT) | 11.08 | 26.85 | 14.22 | 8.55 |
| Ours (DiT + spectral reg.) | **8.63** | **23.38** | **13.08** | **7.92** |
