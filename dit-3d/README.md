# DiT-3D with spectral representation regularization

Training and evaluation code for DiT-3D baseline and our proposed method. 

This codebase is adapted from [DiT-3D](https://github.com/DiT-3D/DiT-3D).

## Install

```bash
# Python 3.9 + CUDA 12.4 toolkit on PATH (nvcc, libcublas, libcusolver).
pip install -r requirements.txt

# Build the EMD CUDA extension once (the Chamfer extension is JIT-built on
# first use inside eval.py).
cd metrics/PyTorchEMD && python setup.py install && cd ../..
```

## Data

[ShapeNetCore.v2.PC15k](https://github.com/stevenygd/PointFlow)
(`.npy` per shape, 15k uniformly-sampled points):

```
ShapeNetCore.v2.PC15k/
└── <synset_id>/{train,val,test}/<model_id>.npy
```

Pass the dataset root through `DATA_ROOT`; scripts default to
`./ShapeNetCore.v2.PC15k`.

## Train

One script per (category, method). Each script trains either the baseline
or our proposed spectral-regularized variant and writes checkpoints +
sample visualizations to `output/<experiment_name>/`.

```bash
# Baseline (DiT-w)
DATA_ROOT=/path/to/ShapeNetCore.v2.PC15k bash scripts/train_chair.sh
DATA_ROOT=/path/to/ShapeNetCore.v2.PC15k bash scripts/train_airplane.sh
DATA_ROOT=/path/to/ShapeNetCore.v2.PC15k bash scripts/train_car.sh

# Ours (spectral-regularized DiT-w)
DATA_ROOT=/path/to/ShapeNetCore.v2.PC15k bash scripts/train_ss_chair.sh
DATA_ROOT=/path/to/ShapeNetCore.v2.PC15k bash scripts/train_ss_airplane.sh
DATA_ROOT=/path/to/ShapeNetCore.v2.PC15k bash scripts/train_ss_car.sh
```


## Evaluate

The test scripts use `eval.py`: samples and references are accumulated
over the complete validation split before PointFlow/PVD-style 1-NNA, COV, and
MMD are computed. For 1-NNA, values near 0.5 indicate that generated and real
sets are difficult to distinguish; values near 1.0 indicate separable sets.
The CUDA voxelizer uses atomic reductions, so independently regenerated samples
can differ slightly even with the same seed; metrics are reproducible exactly
when computed from the same saved `syn/samples.pth` and `syn/refs.pth` tensors.

Below shows an example of evaluating the 5k-epoch checkpoints. The canonical
ShapeNet train/validation counts are checked by default.

```bash
# Baseline
DATA_ROOT=/path/to/ShapeNetCore.v2.PC15k MODEL=output/ditw_S4_chair/epoch_4999.pth bash scripts/test_chair.sh
DATA_ROOT=/path/to/ShapeNetCore.v2.PC15k MODEL=output/ditw_S4_airplane/epoch_4999.pth bash scripts/test_airplane.sh
DATA_ROOT=/path/to/ShapeNetCore.v2.PC15k MODEL=output/ditw_S4_car/epoch_4999.pth bash scripts/test_car.sh

# Ours
DATA_ROOT=/path/to/ShapeNetCore.v2.PC15k MODEL=output/ssditw_S4_chair_enc6_pcoeff0.05_tcoeff0.08/epoch_4999.pth bash scripts/test_ss_chair.sh
DATA_ROOT=/path/to/ShapeNetCore.v2.PC15k MODEL=output/ssditw_S4_airplane_enc6_pcoeff0.05_tcoeff0.08/epoch_4999.pth bash scripts/test_ss_airplane.sh
DATA_ROOT=/path/to/ShapeNetCore.v2.PC15k MODEL=output/ssditw_S4_car_enc6_pcoeff0.05_tcoeff0.08/epoch_4999.pth bash scripts/test_ss_car.sh
```

## Reproduction results

The baseline and spectral-regularized models were each re-trained on one node
with eight NVIDIA A100 80 GB GPUs and evaluated on the canonical ShapeNet
validation split using the final `epoch_9999.pth` checkpoint. We updated the
evaluator because the previous implementation computed nearest-neighbor and
coverage statistics separately within each minibatch, even though these are
set-level metrics. The corrected evaluator accumulates the complete validation
set before computing 1-NNA and COV with CD and EMD.

| Class | Method | Iter | 1-NNA (CD) | 1-NNA (EMD) | COV (CD) | COV (EMD) |
|---|---|---:|---:|---:|---:|---:|
| Chair | Baselines (DiT-3D) | 9999 | 0.6699 | 0.6775 | 0.3882 | 0.4033 |
| Chair | Ours (DiT-3D + spectral reg.) | 9999 | **0.6329** | **0.5899** | **0.4215** | **0.4789** |
| Airplane | Baselines (DiT-3D) | 9999 | 0.9222 | 0.8198 | 0.3111 | 0.3012 |
| Airplane | Ours (DiT-3D + spectral reg.) | 9999 | **0.8259** | **0.7284** | **0.3951** | **0.4074** |
