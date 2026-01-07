# GatedFrequencyIMDN

Lightweight (≤0.5M params) image super-resolution (x2/x3/x4) for the AI1003 course project. We train from scratch on DF2K and evaluate on Set5 (PSNR on Y channel with 2-pixel shave) against Bicubic (baseline 32.25 dB).

## What’s new (our innovation)
- **DFT-aware channel attention (DFTAttention):** uses `rfft2` amplitude, radial low/mid/high frequency masks, and learnable band gates to reweight channels with negligible parameter overhead.
- **Gated frequency IMDB (GateFreqIMDB):** multi-distillation blocks fused with the above frequency attention to preserve high-frequency details while staying lightweight.
- **Multi-scale upsampling:** supports ×2/×3/×4 via pixel shuffle with shared lightweight backbone.
- **Parameter budget friendly:** default `num_feat=48`, `num_blocks=6`, `distill=0.25` keeps total params <0.5M (verify with `sum(p.numel() for p in model.parameters())`).

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
- Server path: `/mnt/8thdd/AI1003/public` (DF2K, Set5). Copy/symlink into `datasets/` following `datasets/README.md`.
- Train: DF2K. Test: Set5. Scale: x2 (primary). Eval on **Y channel**, shave 2 px.

## Key files
- `basicsr/archs/GateFreqIMDN_arch.py` — core model (DFTAttention + GateFreqIMDB + multi-scale upsampling).
- `basicsr/train.py`, `basicsr/test.py` — entrypoints.
- `options/train/*.yml`, `options/Test/*.yml` — configs (set dataset paths, scale, logging, seeds).
- `experiments/` — logs/checkpoints. `results/` — test outputs.

## Training
- Set your train config (e.g., `options/train/train_gated_freq_imdn.yml`): data roots, scale, crop size, batch size, epochs, optimizer/LR schedule, losses, augmentations.
- Run:
```bash
python basicsr/train.py -opt options/train/train_gated_freq_imdn.yml
```
- Check param count (goal ≤0.5M) and log it.

## Testing
- Set test config (e.g., `options/Test/test_gated_freq_imdn.yml`): dataset roots, checkpoint path, `use_chop` if memory-limited.
- Run:
```bash
python basicsr/test.py -opt options/Test/test_gated_freq_imdn.yml
```
- Report PSNR on Y channel with 2-pixel shave; compare to Bicubic 32.25 dB.

## Evaluation & reporting (course requirements)
- Baseline: Bicubic 32.25 dB on Set5 (x2). Extra credit per +0.2 dB.
- Report: PSNR, param count (≤0.5M), optionally FLOPs; qualitative comparisons.
- Train **from scratch** (no pretrained weights).
- Deliverables: PPT, PDF (CVPR 2026 template, 4–8 pages), arch file, configs, (custom) model file, checkpoints (`experiments/pretrained_models`).

## Repro tips
- Fix seeds in configs; log hardware (8×3090), epochs, batch size, LR schedule, losses, aug (crop size, flip/rotate).
- Keep checkpoints under `experiments/` and outputs under `results/`.
- For stability, default band gates are zero-initialized to near-uniform softmax; adjust masks or gating only if you change input resolution regimes.

## Acknowledgements
Built on [BasicSR](https://github.com/XPixelGroup/BasicSR).
