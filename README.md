# Revisiting Spectral Representations in Generative Diffusion Models


This repository contains the code release for [**Revisiting Spectral Representations in Generative Diffusion Models**](https://arxiv.org/pdf/2609.08253), including baseline models and our spectral representation regularization method for 2D toy distributions, image generation, and 3D point cloud generation.

Our method jointly learns spectral representations during diffusion training through a self-supervised regularizer, without requiring an external pretrained encoder. The paper connects the spectral learning objective to diffusion score distillation in representation space and demonstrates improvements in generation quality across images and 3D point clouds.

| Directory | Experiments | Documentation |
| --- | --- | --- |
| [2d_dist](./2d_dist/) | Rectified flow on 2D toy distributions, with baseline and spectral-regularized training and comparison plots. | [Setup and usage](./2d_dist/README.md) |
| [dit-img](./dit-img/) | Image generation with DiT on CIFAR-10, CelebA, FFHQ, and ImageNet-64. | [Setup, training, evaluation, and reproduction results](./dit-img/README.md) |
| [dit-3d](./dit-3d/) | 3D point cloud generation with DiT-3D on ShapeNet chairs, airplanes, and cars. | [Setup, training, evaluation, and reproduction results](./dit-3d/README.md) |

To get started, enter the relevant subfolder and follow its README for dependencies, data preparation, and experiment commands. The image-generation README also links to released checkpoints and evaluation reference batches.
