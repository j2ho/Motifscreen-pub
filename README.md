# MotifScreen-Aff

Multi-task virtual screening model combining motif prediction, structure prediction, and binding classification. Uses an SE(3)-equivariant transformer (or EGNN) for receptor grids and a GAT for ligand graphs, connected by trigonal attention modules.

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and CUDA 11.7+.

```bash
uv sync

# With preprocessing dependencies (for preparing new targets)
uv sync --extra preprocessing
```

Key dependencies: PyTorch 1.13.1, DGL 1.1.3, e3nn 0.5.1 (all managed via `pyproject.toml` + `uv.lock`).

See [INSTALL.md](INSTALL.md) for detailed setup and troubleshooting.

## Inference

Two-step pipeline: prepare input files, then predict binding scores.

```bash
# 1. Prepare: protein PDB + ligand SDF -> model input format
uv run python motifscreen.py prepare \
    --protein receptor.pdb \
    --ligands compounds.sdf \
    --center 12.5,34.2,8.7 \
    --output prepared/

# 2. Predict: score compounds
uv run python motifscreen.py predict \
    --datapath prepared/ \
    --checkpoint models/best.pkl \
    --base-config configs/training/endtoend.yaml \
    --output scores.csv
```

Protein structures should have hydrogens added beforehand (see below). No Rosetta installation needed; atom typing parameters are bundled.

See [INFERENCE.md](INFERENCE.md) for full details: hydrogen addition, binding site definition, multi-target screening, troubleshooting.

### Hydrogen addition

The model expects protonated structures. Add hydrogens before `prepare`:

```bash
# OpenBabel (simplest)
obabel receptor.pdb -O receptor_h.pdb -h

# reduce (better for proteins)
reduce -BUILD receptor.pdb > receptor_h.pdb
```

### Multi-GPU inference

Multiple GPUs are auto-detected and used. Or specify explicitly:

```bash
uv run python motifscreen.py predict \
    --datapath prepared/ \
    --checkpoint models/best.pkl \
    --base-config configs/training/endtoend.yaml \
    --gpus 0,1,2,3 \
    --num-workers 8 \
    --output scores.csv
```

Single GPU: auto if only 1 GPU visible, or `--gpus 0`. CPU: `--device cpu`.

## Benchmarking

Run standard benchmarks (DUD-E, ChEMBL, LIT-PCBA) for any model checkpoint:

```bash
# All benchmarks, parallel SLURM jobs
bash inf_scripts/run_benchmark_parallel.sh my-model models/my-model/best.pkl chembl,dude,litpcba 2 nova015

# Compare runs
uv run python results/analysis/compare_benchmarks.py run1 run2
```

See [BENCHMARK.md](BENCHMARK.md) for target lists, output structure, multi-GPU options, and analysis tools.

## Training

Training uses DDP across multiple GPUs.

### Training modes

| Mode | Config | Description |
|------|--------|-------------|
| Pretrain | `configs/training/pretrain.yaml` | Motif + structure only (no screening) |
| Transfer | `configs/training/transfer.yaml` | Full model from pretrained checkpoint |
| End-to-end | `configs/training/endtoend.yaml` | Full model from scratch |

EGNN variants: `pretrain_egnn.yaml`, `transfer_egnn.yaml`, `endtoend_egnn.yaml`.

### Running

```bash
# Transfer learning
uv run python -m scripts.train.train \
    --config configs/training/transfer.yaml \
    --version ablation \
    --model_note my-transfer \
    --chkpt_name models/sm-pretrain/epoch150.pkl \
    --transfer

# End-to-end
uv run python -m scripts.train.train \
    --config configs/training/endtoend.yaml \
    --version ablation \
    --model_note my-e2e
```

### Config essentials

```yaml
data:
  datapath: "/path/to/your/database/"   # external DB with npz/mol2 per target

training:
  lr: 1.0e-4            # 5e-5 for transfer
  ddp: true              # multi-GPU
  wandb_mode: "online"   # "online", "offline", or "disabled"
```

### Loss system

```
loss = w_motif * (motif_pos + motif_neg)
     + w_motif_contrast * motif_contrast
     + warmup * (w_str * structure + w_str_pair * pairwise + w_str_attmap * attention)
     + screen_source_weight * (w_screen_bce * bce + w_screen_rank * ranking + w_screen_contrast * contrast)
     + w_penalty * l2_reg
```

Enrichment optimization options:
- `screen_rank_alpha`: top-weighted pairwise loss (0 = AUC, 20 = BEDROC-aware)
- `screen_cont_top_k/top_weight/margin`: hard decoy weighting in contrast loss
- `hard_neg_capacity`: memory bank of hard negatives across batches (0 = disabled)
- `screen_source_weight`: per-data-source screening weight (e.g., chembl: 1.0, pdbbind: 0.0)

### Output

```
models/{model_note}/
  model.pkl     # latest checkpoint
  best.pkl      # best validation loss
  epoch{N}.pkl  # periodic snapshots
```

## Data setup

Training data structure under `datapath`:

```
{datapath}/
  pdbbind/{target_id}/
    {target_id}.grid.npz          # binding site grid
    {target_id}.prop.npz          # receptor properties
    {target_id}.keyatom.def.npz   # key atom definitions
    ligand.mol2                   # native ligand
  biolip/{target_id}/...
  chembl/{target_id}/
    {target_id}.grid.npz
    {target_id}.prop.npz
    batch_mol2s/                  # batched compounds (active + decoys)
```

Target lists: `data/final-set-260301/pbc_train.txt`, `pbc_valid.txt`.

## Project structure

```
MotifScreen-Aff/
  motifscreen.py              # CLI entry point (prepare + predict)
  src/
    cli/                      # prepare.py, predict.py
    data/                     # dataset, mol2 parsing, SASA, graph building
    model/
      models/                 # SE(3) and EGNN architectures
      modules/                # trigon attention, featurizers
      loss/                   # screening, motif, structure losses
    io/                       # protein/ligand file utilities
  scripts/
    train/                    # training script
    inference/                # benchmark scripts (single + multi-GPU)
  configs/
    training/                 # pretrain, transfer, e2e configs
    inference/                # benchmark configs
    config_loader.py          # config dataclasses
  data/
    rosetta_params/           # atom typing parameters (bundled)
    generic_potential/        # ligand atom typing (bundled)
    final-set-260301/         # train/valid target lists
  train_scripts/              # SLURM scripts for training
  inf_scripts/                # SLURM scripts for inference + benchmarks
  results/analysis/           # comparison script + notebook
  INFERENCE.md                # inference guide
  BENCHMARK.md                # benchmarking guide
  INSTALL.md                  # installation details
```
