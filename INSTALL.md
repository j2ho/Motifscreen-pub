# Installation Guide

## Requirements

- Linux (tested on Ubuntu 20.04+)
- CUDA 11.7+ compatible GPU
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- ~5GB disk space for environment

## Quick Start (Recommended)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/MotifScreen-Aff.git
cd MotifScreen-Aff
```

### 2. Create environment with uv

```bash
uv sync
```

This creates a `.venv/` directory and installs all dependencies including
PyTorch with CUDA 11.7 and DGL with CUDA 11.7.

### 3. Verify installation

```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
uv run python -c "import dgl; print(f'DGL: {dgl.__version__}')"
uv run python -c "import e3nn; print(f'e3nn: {e3nn.__version__}')"
```

Expected output:
```
PyTorch: 1.13.1+cu117, CUDA: True
DGL: 1.1.3+cu117
e3nn: 0.5.1
```

## Running

### Training

```bash
uv run python -u -m scripts.train.train --config configs/training/transfer.yaml
```

### Inference

```bash
uv run python scripts/inference/run_inference_general.py --config configs/inference/general.yaml
```

## Optional: Preprocessing Dependencies

If you need the legacy ligand preprocessing tools (`src/io/ligand_processer.py`):

```bash
# Requires system libopenbabel-dev
sudo apt install libopenbabel-dev
uv sync --extra preprocessing
```

## Troubleshooting

### CUDA version mismatch

Check your CUDA version:
```bash
nvidia-smi
```

The default configuration targets CUDA 11.7. For a different CUDA version,
update the index URL in `pyproject.toml`:

| CUDA Version | PyTorch Index URL |
|--------------|-------------------|
| 11.7 | `https://download.pytorch.org/whl/cu117` |
| 11.8 | `https://download.pytorch.org/whl/cu118` |
| 12.1 | `https://download.pytorch.org/whl/cu121` |

### e3nn compatibility

e3nn 0.5.1 requires PyTorch 1.13.x. If you need a different PyTorch version,
update the `e3nn` version in `pyproject.toml` and re-run `uv sync`.

## Weights & Biases Setup

For experiment tracking:
```bash
wandb login
```

Or disable in config:
```yaml
training:
  wandb_mode: "disabled"
```

## Hardware Requirements

| Task | GPU Memory | Recommended |
|------|------------|-------------|
| Inference | 4-8 GB | RTX 3060+ |
| Training | 16-24 GB | RTX 3090, A5000+ |
| Training (DDP) | 8+ GB x N GPUs | Multi-GPU setup |

## Test Installation

Run a quick test:
```bash
uv run python -c "
import torch
import dgl
import e3nn
from src.model.models.msk1 import EndtoEndModel
print('All imports successful!')
"
```
