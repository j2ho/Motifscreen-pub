# Inference & Benchmark Scripts

## Overview

| Script | Use Case | Labels | Multi-GPU |
|--------|----------|--------|-----------|
| `run_inference_general.py` | Virtual screening (new data) | No | No |
| `run_chembl_benchmark_multigpu.py` | ChEMBL benchmark | Yes (CSV) | Yes |
| `run_dude_benchmark.py` | DUD-E benchmark | Yes (file) | No |
| `run_litpcba_benchmark.py` | LIT-PCBA benchmark | Yes (file) | No |
| `calculate_metrics.py` | Compute AUROC, EF, BEDROC | -- | -- |

## Configuration

All inference is driven by YAML configs in `configs/inference/`. Template configs for each benchmark are provided, plus per-model variants in `configs/inference/per_model/`.

### What to change in an inference config

```yaml
model:
  checkpoint: "models/your-model/best.pkl"        # <-- Your trained checkpoint
  base_config: "configs/training/endtoend.yaml"    # Training config (defines model architecture)

data:
  datapath: "/path/to/your/data/"                  # <-- Your preprocessed data directory
  targets_file: "data/litpcba_targets.txt"         # Target list (in-repo or absolute path)

inference:
  batch_size: 20                                   # Compounds per forward pass (GPU memory)
  output_dir: "results/my_run/"                    # Per-target score CSVs
  combined_output: "results/my_run_all_scores.csv" # All targets in one file
  device: "cuda"
```

`base_config` must match the architecture the checkpoint was trained with (it's used to instantiate the model, not for training parameters).

### Creating a per-model config

The easiest way to run a new model on an existing benchmark is to copy a per-model config and change the checkpoint:

```bash
cp configs/inference/per_model/inference_chembl_test_trans-b15e9-b20-cont.yaml \
   configs/inference/per_model/inference_chembl_test_my-new-model.yaml
# Edit: change checkpoint, output_dir, combined_output
```

## General Inference (Virtual Screening)

For scoring new compounds without active/decoy labels.

```bash
uv run python scripts/inference/run_inference_general.py \
    --config configs/inference/general.yaml
```

**Config:** `configs/inference/general.yaml`

**Data structure** (flat, per-target directories):
```
{datapath}/{target_id}/
├── {target_id}.grid.npz
├── {target_id}.prop.npz
├── {target_id}.keyatom.def.npz
└── *.mol2                        # All mol2 files are scored
```

**Output:**
```csv
target_id,compound_id,score,source_mol2
TARGET1,ZINC001,0.85,ligands
TARGET1,ZINC002,0.23,ligands
```

## ChEMBL Benchmark

For the ChEMBL test set with nested directory structure and CSV-based active labels.

```bash
# Single GPU
uv run python scripts/inference/run_chembl_benchmark_multigpu.py \
    --config configs/inference/chembl_test.yaml \
    --batch-size-per-gpu 20

# Multi-GPU (auto-splits targets across GPUs)
uv run python scripts/inference/run_chembl_benchmark_multigpu.py \
    --config configs/inference/chembl_test.yaml \
    --targets-file data/inference_targets_113.csv \
    --batch-size-per-gpu 20
```

**Config:** `configs/inference/chembl_test.yaml`

**Data structure** (nested, with batch mol2 subdirectories):
```
{datapath}/{source}/{target_id}/
├── {target_id}.grid.npz
├── {target_id}.prop.npz
├── active_smiles_clu_d3.csv           # Active compound IDs
└── batch_mol2s_sim_check/
    ├── {compound}_b.mol2
    └── {compound}_b.keyatom.def.npz
```

ChEMBL-specific config fields:
```yaml
data:
  source: "chembl"                           # Subdirectory under datapath
  mol2_subdir: "batch_mol2s_sim_check"       # Where mol2 files live per target
  mol2_pattern: "*_b.mol2"                   # Glob for mol2 files
  keyatom_pattern: "{stem}.keyatom.def.npz"  # Keyatom file naming ({stem} = mol2 filename stem)

evaluation:
  actives_csv: "active_smiles_clu_d3.csv"             # Per-target actives list
  actives_id_column: "chemblid(with_best_aff)"        # Column with compound IDs
```

**Output:**
```csv
target_id,compound_id,score,is_active,batch_mol2
B2RXC2,CHEMBL123,0.892,1,CHEMBL123_b
```

## DUD-E Benchmark

For DUD-E with flat file structure and file-based active/decoy labels.

```bash
uv run python scripts/inference/run_dude_benchmark.py \
    --config configs/inference/dude.yaml
```

**Config:** `configs/inference/dude.yaml`

**Data structure** (flat, all files in one directory):
```
{datapath}/
├── {target}.grid.npz
├── {target}.prop.npz
├── {target}.active.mol2       # Scored with is_active=1
├── {target}.decoy.mol2        # Scored with is_active=0
└── {target}.keyatom.def.npz
```

**Output:**
```csv
target_id,compound_id,score,is_active,source
aldr,ZINC001,0.85,1,active
aldr,ZINC002,0.23,0,decoy
```

## LIT-PCBA Benchmark

For LIT-PCBA with DUD-E-style flat structure. Uses streaming inference (build batch, infer, repeat) to handle large decoy sets.

```bash
# Optional: pre-split large mol2 files for streaming
uv run python scripts/split_mol2.py /path/to/litpcba/data \
    --pattern "*.decoy.mol2" --max-lines 100000

# Run benchmark
uv run python scripts/inference/run_litpcba_benchmark.py \
    --config configs/inference/litpcba.yaml
```

**Config:** `configs/inference/litpcba.yaml`

Same flat data structure as DUD-E.

## Metrics Calculation

Computes virtual screening metrics from scored results. Requires `is_active` column (produced by benchmark scripts, not general inference).

```bash
# Single target file
uv run python scripts/inference/calculate_metrics.py \
    --input results/target_scores.csv

# Combined file, per-target breakdown
uv run python scripts/inference/calculate_metrics.py \
    --input results/all_scores.csv \
    --by-target \
    --output results/metrics.csv

# All CSVs in a directory
uv run python scripts/inference/calculate_metrics.py \
    --input-dir results/chembl_test/ \
    --output results/metrics.csv
```

**Metrics:**
- **AUROC** -- Area Under ROC Curve
- **EF1%** -- Enrichment Factor at 1%
- **EF5%** -- Enrichment Factor at 5%
- **BEDROC (alpha=20)** -- Early recognition, standard virtual screening threshold
- **BEDROC (alpha=80.5)** -- Early recognition, stricter threshold

## Target File Formats

Both CSV and plain text are supported:

**CSV** (auto-detects column: `target_id`, `target_name`, `target`, `uniprot_id`, or `id`):
```csv
target_id,n_actives,n_decoys
B2RXC2,17,510
Q9Y233,16,480
```

**Plain text** (one target per line, `#` comments ignored):
```
B2RXC2
Q9Y233
# this line is skipped
```

## SLURM Shell Scripts

Template scripts are in the project root and `inf_scripts/`:

| Script | Benchmark | Notes |
|--------|-----------|-------|
| `run_general_inference.sh` | General | Example data |
| `run_chembl_inf.sh` | ChEMBL | All 113 targets, multi-GPU |
| `run_dude_inf.sh` | DUD-E | All targets |
| `run_litpcba_benchmark.sh` | LIT-PCBA | Includes mol2 splitting step |
| `inf_scripts/run_*_{model}.sh` | Per-model | ChEMBL/DUD-E/LIT-PCBA for specific models |

Update SLURM headers for your cluster:
```bash
#SBATCH --gres=gpu:1               # GPUs (inference typically needs 1)
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=nova[005,015,016]
```

## End-to-End Example

Running a newly trained model on ChEMBL test set:

```bash
# 1. Create per-model inference config
cp configs/inference/per_model/inference_chembl_test_trans-b15e9-b20-cont.yaml \
   configs/inference/per_model/inference_chembl_test_my-model.yaml

# 2. Edit: set checkpoint, output paths
#    checkpoint: "models/my-model/best.pkl"
#    output_dir: "results/chembl_test_my-model/"
#    combined_output: "results/chembl_test_my-model_all_scores.csv"

# 3. Run inference
uv run python scripts/inference/run_chembl_benchmark_multigpu.py \
    --config configs/inference/per_model/inference_chembl_test_my-model.yaml \
    --targets-file data/inference_targets_113.csv \
    --batch-size-per-gpu 20

# 4. Calculate metrics
uv run python scripts/inference/calculate_metrics.py \
    --input results/chembl_test_my-model_all_scores.csv \
    --by-target \
    --output results/chembl_test_my-model_metrics.csv
```
