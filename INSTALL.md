# Installation

## Requirements

- Linux (Ubuntu 20.04+ tested; other distros should work if the Python + CUDA versions are available)
- Python 3.9 (locked by `pyproject.toml`)
- CUDA 11.7-compatible NVIDIA GPU (or CPU fallback)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- ~5 GB free disk (env + wheels)

## Standard install

```bash
git clone https://github.com/j2ho/Motifscreen-pub.git
cd Motifscreen-pub
uv sync --extra preprocessing
```

Explanation:
- `uv sync` creates a `.venv/` and installs the base deps (PyTorch 1.13.1+cu117, DGL 1.1.3+cu117, e3nn 0.5.1, numpy, scipy, sklearn, pandas, torch, dgl, ...).
- `--extra preprocessing` adds `rdkit` and `openbabel-wheel`, both required for the `prepare` step. Skip this extra only if you're going straight to `predict` on a pre-featurized dataset (e.g. the Zenodo benchmark tarball).

Verify it worked:

```bash
uv run python -c "import torch, dgl, e3nn; print(torch.__version__, dgl.__version__, e3nn.__version__, torch.cuda.is_available())"
# expect: 1.13.1+cu117 1.1.3+cu117 0.5.1 True
```

## Different CUDA version

Edit `pyproject.toml`, replace the `pytorch-cu117` index URL:

| CUDA | Index URL |
|---|---|
| 11.7 (default) | `https://download.pytorch.org/whl/cu117` |
| 11.8 | `https://download.pytorch.org/whl/cu118` |
| 12.1 | `https://download.pytorch.org/whl/cu121` |

Also update the DGL find-links line to match (`https://data.dgl.ai/wheels/cu118/repo.html`, etc.), then re-run `uv sync`.

## CPU-only

CUDA is not strictly required — passing `--device cpu` to `predict` works but is ~10-50x slower than a single GPU. Practical only for a handful of compounds.

