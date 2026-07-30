# Inference Guide

Screen a compound library against a protein target with MotifScreen-Aff.

Two commands: `prepare` featurizes a target (protein + compound library → npz + mol2 + keyatoms), `predict` scores compounds against the prepared target.

## Quick start

```bash
# 1. Add hydrogens (reduce is easiest; obabel / pdb2pqr also fine)
reduce -BUILD receptor.pdb > receptor_h.pdb

# 2. Prepare
uv run python motifscreen.py prepare \
    --protein receptor_h.pdb \
    --ligands compounds.sdf \
    --center=12.5,34.2,8.7 \
    --output prepared/

# 3. Predict
uv run python motifscreen.py predict \
    --datapath prepared/ \
    --checkpoint models/epoch70.pkl \
    --base-config configs/training/endtoend.yaml \
    --output scores.csv
```

Output `scores.csv`:
```
target_id,compound_id,score,source_mol2
mytarget,ZINC000012345,0.912,all_ligands
mytarget,ZINC000098765,0.023,all_ligands
```

Score is a sigmoid output in [0, 1]. Higher = predicted stronger binder.

## Install checklist

```bash
uv sync --extra preprocessing
```

The `preprocessing` extra pulls in `rdkit` and `openbabel-wheel` — both required for `prepare`. No Rosetta needed (params files are bundled at `data/rosetta_params/`).

## Protein hydrogen addition

`prepare` expects a protonated PDB. Add hydrogens with any of these:

```bash
# reduce (Richardson lab; typically preinstalled on Linux)
reduce -BUILD receptor.pdb > receptor_h.pdb

# OpenBabel
obabel receptor.pdb -O receptor_h.pdb -h

# PDBFixer (also handles missing residues; pip install pdbfixer openmm)
python -c "
from pdbfixer import PDBFixer
from openmm.app import PDBFile
fixer = PDBFixer(filename='receptor.pdb')
fixer.findMissingResidues(); fixer.findMissingAtoms()
fixer.addMissingAtoms(); fixer.addMissingHydrogens(7.4)
PDBFile.writeFile(fixer.topology, fixer.positions, open('receptor_h.pdb', 'w'))
"
```

reduce and obabel are the two we've tested most. Small (<0.03 mean AUROC) prediction quality differences between protonation tools on standard benchmarks; reduce is the default recommendation.

If you have Rosetta installed, you can also pass its `score_jd2` binary directly and let `prepare` handle protonation for you:
```bash
uv run python motifscreen.py prepare \
    --protein receptor.pdb \
    --ligands compounds.sdf \
    --center=12.5,34.2,8.7 \
    --protonate-rosetta /path/to/score_jd2.linuxgccrelease \
    --output prepared/
```

## Step 1: `prepare`

Featurizes the target once.

### Inputs

| Flag | Type | Meaning |
|---|---|---|
| `--protein` | PDB path | Protonated protein structure |
| `--ligands` | SDF or mol2 path | Compound library (multi-molecule) |
| `--center=X,Y,Z` OR `--crystal-ligand PATH` | either | Binding site: coords, or COM of a reference ligand mol2 |
| `--output` | dir | Target directory (created if missing) |
| `--target-id` | string (opt) | Name for the target subdir (default: derived from PDB filename) |

Negative coordinates need `=` syntax: `--center=-3.7,-3.6,-8.0` (without `=`, bash reads `-3.7` as a flag).

### Optional flags

```
--gridsize 1.5              # Grid point spacing in Å (default 1.5)
--padding 10.0              # Grid padding in Å around binding site (default 10.0)
--clash 1.1                 # Clash cutoff for grid point pruning (default 1.1)
--workers 4                 # Parallel workers for keyatom BRICS (default 4)
--keep-hetatms RES,RES,...  # Non-standard residues to keep as protein atoms (e.g. cofactors)
--skip-ligand-prep          # Skip obabel H-add + MMFF94 charge assignment (input mol2 must already have both)
--protonate-rosetta PATH    # Path to Rosetta score_jd2 binary (alternative to pre-protonating)
--precompute-graphs         # Also build DGL graphs at prep time (fast-path for reuse; opt-in)
```

### What `prepare` produces

```
prepared/<target-id>/
├── <target-id>_stripped.pdb        # Cleaned protein (HETATMs removed, whitelist honored)
├── <target-id>.grid.npz            # Binding-site grid (~800 points near --center)
├── <target-id>.prop.npz            # Per-atom receptor features (coords, charges, atypes, SASA, bonds)
├── all_ligands.mol2                # Ligands with polar H + MMFF94 charges (one file, all compounds)
└── all_ligands.keyatom.def.npz     # BRICS keyatoms per compound, dict keyed by compound ID
```

The `all_ligands.mol2` name comes from the `--ligands` filename stem. If you pass `--ligands foo.sdf`, you'll get `foo.mol2` and `foo.keyatom.def.npz` in the target dir.

## Step 2: `predict`

Runs the model over a prepared target (or many prepared targets).

### Basic

```bash
uv run python motifscreen.py predict \
    --datapath prepared/ \
    --checkpoint models/epoch70.pkl \
    --base-config configs/training/endtoend.yaml \
    --output scores.csv
```

`--datapath` is the parent directory containing per-target subdirs (each with a `.grid.npz`). All target subdirs are auto-detected and scored in one run. Output is a single CSV with a `target_id` column distinguishing them.

### All predict flags

```
--datapath          Parent directory containing prepared/<target>/ subdirs (required)
--checkpoint        Model checkpoint .pkl (required)
--base-config       Model architecture config YAML (required — use configs/training/endtoend.yaml)
--targets t1 t2 ... Only score these target IDs (default: auto-detect all)
--output PATH       Output CSV path
--run-name NAME     Alternative to --output: writes to results/<name>/scores.csv
--mol2-pattern GLOB Ligand mol2 glob within a target dir (default: *.mol2)
--batch-size N      Compounds per batch (default 64; drop to 16-32 on tight VRAM)
--gpus IDS          GPU IDs, comma-separated (default: all visible)
--num-workers N     CPU threads for graph building in multi-GPU mode (default 8)
--device DEV        cuda or cpu (default cuda; use cpu for tiny screens without GPU)
```

### Output format

```csv
target_id,compound_id,score,source_mol2
mytarget,ZINC000012345,0.9124,all_ligands
mytarget,ZINC000098765,0.0234,all_ligands
```

Rows sorted by descending score across all targets.

### Multi-GPU

```bash
uv run python motifscreen.py predict \
    --datapath prepared/ \
    --checkpoint models/epoch70.pkl \
    --base-config configs/training/endtoend.yaml \
    --gpus 0,1 \
    --batch-size 32 \
    --output scores.csv
```

One model replica per GPU, batches round-robin across them. **2 GPUs is the sweet spot** — scaling is sub-linear above that (CPU-side featurization becomes the bottleneck).

### Throughput expectations

Per-compound wallclock, ~15 ms including featurize + forward:
- Single GPU (V100/A5000+): ~40-60 compounds/sec effective
- 2 GPUs: ~70-90 compounds/sec effective

So for a 500k-compound library on 2 GPUs, budget ~2 hours end-to-end (prep ~10-30 min + predict ~1.5-2 hr).

For very large libraries (>100k compounds), the mol2 reader streams internally so memory stays bounded regardless of file size.

## Multi-target screening

Just call `prepare` per target, writing into the same parent directory, then run `predict` once over the whole tree:

```bash
uv run python motifscreen.py prepare --protein t1_h.pdb --ligands lib.sdf \
    --center=10,20,30 --output prepared/ --target-id t1

uv run python motifscreen.py prepare --protein t2_h.pdb --ligands lib.sdf \
    --center=15,25,35 --output prepared/ --target-id t2

uv run python motifscreen.py predict \
    --datapath prepared/ \
    --checkpoint models/epoch70.pkl \
    --base-config configs/training/endtoend.yaml \
    --gpus 0,1 \
    --output scores.csv
```

For 3+ targets, the batched helper is easier: see `scripts/prepare_batch.sh` and `scripts/gen_manifest.py` (documented in [BENCHMARKS.md](BENCHMARKS.md)).

## Troubleshooting

**"No grid points generated"**: `--center` coordinates aren't near any protein atoms. Sanity-check the coords, or use `--crystal-ligand` with a known bound ligand.

**"No results produced"**: usually a keyatom-lookup mismatch. Confirm `all_ligands.keyatom.def.npz` exists in the target dir and its compound IDs match the mol2 records. If you split the mol2 into `batch_mol2s/`, keep the keyatom file at the target root — predict searches there.

**GPU OOM**: lower `--batch-size` (16, then 8). Very large targets (>3000 receptor atoms in pocket) may need `--batch-size 5`.

**Negative coordinate parsed as flag**: use `=` syntax: `--center=-3.7,-3.6,-8.0`.
