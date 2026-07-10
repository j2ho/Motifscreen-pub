# Inference Guide

Screen compounds against a protein target using MotifScreen-Aff.

## Quick start

```bash
# 1. Prepare: protein PDB + ligand SDF -> preprocessed files
uv run python motifscreen.py prepare \
    --protein receptor_h.pdb \
    --ligands compounds.sdf \
    --center=12.5,34.2,8.7 \
    --output results/my-screen/prepared

# 2. Predict: score compounds
uv run python motifscreen.py predict \
    --datapath results/my-screen/prepared \
    --checkpoint models/best.pkl \
    --base-config configs/training/endtoend.yaml \
    --run-name my-screen
```

Output: `results/my-screen/scores.csv`

## Installation

```bash
# Core (training + inference)
uv sync

# With preprocessing dependencies (prepare step)
uv sync --extra preprocessing
```

Required for `prepare`: RDKit (BRICS keyatom decomposition), OpenBabel (mol2 conversion, charge assignment).

No Rosetta installation needed. Rosetta parameter files for atom typing are bundled in `data/rosetta_params/`.

## Protein hydrogen addition

The model was trained on protonated structures. Missing hydrogens will degrade prediction quality, particularly for hydrogen bond donor/acceptor features and partial charge assignment.

**Recommended: Rosetta `score_jd2`.** This is what the model was trained with. Best atom-name consistency with the featurizer, and best downstream numbers if you want to reproduce paper results exactly:

```bash
# Requires a Rosetta install (free for academic use, register at rosettacommons.org)
score_jd2.linuxgccrelease -s receptor.pdb -out:file:scorefile /dev/null \
    -no_optH false -out:pdb -overwrite -out:path:pdb .
# produces receptor_0001.pdb

# Or point prepare.py at your Rosetta binary and let it protonate for you:
uv run python motifscreen.py prepare \
    --protein receptor.pdb \
    --protonate-rosetta /path/to/score_jd2.linuxgccrelease \
    ...
```

If Rosetta isn't installed, any of the following work fine in practice (with mild quality/coverage tradeoffs):

```bash
# reduce (Richardson lab, MIT license; usually preinstalled on Linux)
reduce -BUILD receptor.pdb > receptor_h.pdb

# OpenBabel (simplest, works for most cases)
obabel receptor.pdb -O receptor_h.pdb -h

# PDBFixer (Python, also handles missing residues)
# pip install pdbfixer
python -c "
from pdbfixer import PDBFixer
from openmm.app import PDBFile
fixer = PDBFixer(filename='receptor.pdb')
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.4)
PDBFile.writeFile(fixer.topology, fixer.positions, open('receptor_h.pdb', 'w'))
"
```

`prepare` runs regardless of which tool you used, but expect ~1-3% of common PDBs to have residues with non-standard atom naming (metals, cofactors, alt-locs). The featurizer logs a warning and skips those residues rather than aborting.

## Step 1: Prepare

Converts protein PDB + ligand SDF/mol2 into the preprocessed format the model expects.

### Inputs

| Input | Format | Description |
|---|---|---|
| `--protein` | PDB | Protein structure (with hydrogens). |
| `--ligands` | SDF or mol2 | Compounds to screen. Multi-molecule file supported. |
| `--center` | x,y,z | Binding site center coordinates. Use `=` for negative values: `--center=-3.7,-3.6,-8.0` |
| `--output` | directory | Where to write preprocessed files. |

### Binding site center

Either provide coordinates directly or use a reference ligand:

```bash
# Explicit coordinates (use = for negative values)
--center=12.5,34.2,8.7

# From a reference ligand (center of mass)
--crystal-ligand crystal_ligand.mol2
```

### What prepare produces

```
results/my-screen/prepared/
  {target}/
    {target}.grid.npz               # 3D grid points around binding site
    {target}.prop.npz               # Receptor properties (coords, charges, types, SASA, bonds)
    batch_mol2s/
      {compound}_b.mol2             # Compound mol2 with key atom definitions
      {compound}_b.keyatom.def.npz  # BRICS fragment key atoms
```

### All prepare options

```
--protein           Protein PDB file (required)
--ligands           Compound file: SDF or mol2 (required)
--center            Binding site center as x,y,z (use = for negative values)
--crystal-ligand    Reference ligand for center (alternative to --center)
--output            Output directory (required)
--grid-spacing      Grid point spacing in Angstroms (default: 1.5)
--padding           Grid padding around binding site (default: 4.0)
--workers           Parallel workers for keyatom computation (default: 4)
```

## Step 2: Predict

Runs the model on prepared data and outputs binding scores.

### Using --run-name (recommended)

```bash
uv run python motifscreen.py predict \
    --datapath results/my-screen/prepared \
    --checkpoint models/best.pkl \
    --base-config configs/training/endtoend.yaml \
    --run-name my-screen
```

Output goes to `results/my-screen/scores.csv`. Keeps everything for a screening run under one directory.

### Using --output (custom path)

```bash
uv run python motifscreen.py predict \
    --datapath prepared/ \
    --checkpoint models/best.pkl \
    --base-config configs/training/endtoend.yaml \
    --output /path/to/scores.csv
```

### Output format

```csv
target_id,compound_id,score,source_mol2
3jdw,CHEMBL123,0.847,compounds_b
3jdw,CHEMBL456,0.023,compounds_b
3jdw,CHEMBL789,0.912,compounds_b
```

Score is a binding probability (0 to 1). Higher = more likely to bind.

### Multi-GPU

Multiple GPUs are auto-detected. Or specify explicitly:

```bash
uv run python motifscreen.py predict \
    --datapath results/my-screen/prepared \
    --checkpoint models/best.pkl \
    --base-config configs/training/endtoend.yaml \
    --run-name my-screen \
    --gpus 0,1,2,3 \
    --num-workers 8
```

Uses thread-based parallelism. Model replicas are loaded on each GPU, batches round-robin across them. Single GPU: `--gpus 0`. CPU: `--device cpu`.

### All predict options

```
--datapath          Directory with prepared data (required)
--checkpoint        Model checkpoint .pkl file (required)
--base-config       Model architecture config YAML (required)
--run-name          Run name -> results/{run-name}/scores.csv
--output            Output CSV path (alternative to --run-name)
--targets           Specific targets to score (default: auto-detect all)
--batch-size        Compounds per batch (default: 64)
--gpus              Comma-separated GPU IDs (default: auto-detect all)
--num-workers       Graph building threads for multi-GPU (default: 8)
--device            cuda or cpu (default: cuda)
```

### Multi-target screening

```bash
# Prepare multiple targets into the same run directory
uv run python motifscreen.py prepare \
    --protein target1_h.pdb --ligands lib.sdf \
    --center=10,20,30 --output results/my-screen/prepared/target1

uv run python motifscreen.py prepare \
    --protein target2_h.pdb --ligands lib.sdf \
    --center=15,25,35 --output results/my-screen/prepared/target2

# Score all targets
uv run python motifscreen.py predict \
    --datapath results/my-screen/prepared \
    --checkpoint models/best.pkl \
    --base-config configs/training/endtoend.yaml \
    --run-name my-screen
```

All targets auto-detected from subdirectories. Output: `results/my-screen/scores.csv` with `target_id` column distinguishing targets.

### Output directory structure

```
results/my-screen/
  prepared/
    target1/
      target1.grid.npz
      target1.prop.npz
      batch_mol2s/...
    target2/...
  scores.csv              # all scores, all targets
```

## Benchmarking

For running standard benchmarks (DUD-E, ChEMBL, LIT-PCBA) with pre-processed data, see [BENCHMARK.md](BENCHMARK.md).

## Troubleshooting

**"No grid points generated"**: binding site center is too far from the protein. Check `--center` coordinates or use `--crystal-ligand`.

**"KeyAtom computation failed"**: ligand couldn't be fragmented by BRICS. Usually happens with very small molecules (< 5 heavy atoms) or unusual chemistry. These compounds are skipped.

**GPU out of memory**: reduce `--batch-size` (default 64). For very large proteins (> 3000 atoms in binding site), try `--batch-size 5`.

**Negative coordinates in --center**: use `=` syntax: `--center=-3.7,-3.6,-8.0` (without `=`, bash interprets `-` as a flag).
