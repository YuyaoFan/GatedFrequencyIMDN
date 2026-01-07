# GatedFrequencyIMDN

Lightweight image super-resolution (x2) for the AI1003 course project. The goal is to design, train, and evaluate a ≤0.5M-parameter model on DF2K (train) and Set5 (test), reporting PSNR on the Y channel with 2-pixel shave, and comparing against Bicubic (baseline 32.25 dB).

## Features
- Lightweight SR model (≤0.5M params) without pretrained weights (train from scratch).
- DF2K training pipeline with configurable augmentations.
- Unified x2 evaluation on Set5 with PSNR on Y channel and 2-pixel border shave.
- BasicSR-based codebase with reproducible configs for train/test.

## Environment
```bash
conda create -n ai1003 python=3.8 -y
conda activate ai1003
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python setup.py develop
```

Tested on 8×RTX 3090 (24GB) per course server.

## Data
- Datasets are available on the server at `/mnt/8thdd/AI1003/public` (DF2K, Set5). Copy/link them into `datasets/` following `datasets/README.md`.
- Train: DF2K
- Test: Set5
- Scale: x2
- Color space: evaluate on Y channel only; shave 2 pixels for PSNR.

## Directory layout (key)
```
basicsr/         # Core framework and custom SR arch/model code
options/
  train/         # Training configs (YAML)
  test/          # Testing configs (YAML)
datasets/        # Symlink or place DF2K/Set5 per datasets/README.md
experiments/     # Training logs, checkpoints
results/         # Test outputs
```

## Training
- Edit `options/train/train_simpleCNN.yml` as needed (batch size, LR schedule, augmentations, loss, logging).
- Ensure param count ≤0.5M (e.g., `sum(p.numel() for p in model.parameters())`).
- Run:
```bash
python basicsr/train.py -opt options/train/train_simpleCNN.yml
```
- Outputs (logs/checkpoints) are saved to `experiments/`.

## Testing
- Edit `options/Test/test_simpleCNN.yml` (dataset paths, checkpoint path, chop settings).
- Optional: set `use_chop: True` for memory-friendly inference.
- Run:
```bash
python basicsr/test.py -opt options/Test/test_simpleCNN.yml
```
- Outputs go to `results/`. Report PSNR on Y channel with 2-pixel shave.

## Evaluation & Reporting (course requirements)
- Baseline: Bicubic 32.25 dB on Set5 (x2). Aim to exceed this; extra credit per +0.2 dB.
- Must report: param count (≤0.5M target), FLOPs (if available), PSNR, and qualitative comparisons.
- Train from scratch (no pretrained weights).
- Suggested report structure (CVPR 2026 template, 4–8 pages):
  - Introduction, (Related Work optional), Method (data processing, model, training), Experiments (hyperparams, results, ablations), Conclusion (plus team roles).
- Submission pack: PPT, PDF report, arch file (`basicsr/archs/...`), config files (`options/train`, `options/Test`), model file if customized (`basicsr/models/...`), checkpoints in `experiments/pretrained_models`.

## Repro tips
- Fix seeds in configs for reproducibility.
- Document data augmentations (crop size, flip/rotations), optimizer, LR schedule, epochs, batch size, hardware.
- Keep checkpoints and logs under `experiments/`; keep test outputs under `results/`.

## Acknowledgements
Built on [BasicSR](https://github.com/XPixelGroup/BasicSR).
