# Fitting 2D toy distributions

Rectified flow training of the **baseline** (naive diffusion) and **ours**
(diffusion with spectral representation regularization).

## Install

```bash
pip install -r requirements.txt
```

## Run

One script per dataset, each trains both the **baseline** and **ours** and
writes a comparison plot of model training progress to
`output/<dataset>/progress_baseline_vs_ours.png`.

```bash
cd exps
python run_2spirals.py
python run_rings.py
python run_moons.py
python run_pinwheel.py
python run_swissroll.py
python run_8gaussians.py
python run_circles.py
python run_checkerboard.py
```

Each script writes:

```
output/<dataset>/
├── progress_baseline_vs_ours.png   # training data | baseline at each ckpt | ours at each ckpt
├── baseline/ckpts/final_model.pt
└── ours/ckpts/final_model.pt
```
