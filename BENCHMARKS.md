# Benchmark Reproduction

Reproduces the published numbers on three standard virtual-screening benchmarks
plus MotifScreen-Aff's proposed ChEMBL-LR-107 set. All three use the same
`prepare_batch.sh` + `motifscreen predict` pattern; download + manifest details
differ per benchmark.

## Prerequisites

- Model checkpoint downloaded from Zenodo (see `CHECKPOINTS.md`)
- `reduce` binary on PATH for protein protonation (optional; falls back to
  passing the raw PDB through if missing)
- `obabel` binary on PATH (comes with `uv sync`)

## Timings (nova-class cluster, 1 GPU + 8-way parallel prepare)

| Benchmark | Targets | ~Compounds | Prep wall | Predict wall | Total |
|---|---|---|---|---|---|
| ChEMBL-LR-107 | 107 | ~200k | ~15 min | ~30 min | ~45 min |
| DUD-E | 96 | ~1.5M | ~15-30 min | ~2 h | ~2.5 h |
| LIT-PCBA | 14 | ~2.3M | ~20-30 min | ~3 h | ~3.5 h |

Prep time is amortized across `--parallel N` concurrent targets. Predict is
GPU-bound; scales with GPU count.

## ChEMBL-LR-107 (our contribution)

Tercile-balanced by AVE bias (36 top / 35 mid / 36 bottom) to test whether the
model gains generalize across bias regimes.

```bash
bash scripts/download_and_prepare_chembl_bench.sh \
    --checkpoint models/best.pkl \
    --mode fresh \
    --parallel 8
```

Two modes:

- `--mode fresh` (default): run full prepare from raw. Uses `reduce` for
  protonation and public obabel MMFF94 charges. Reproduces what a public user
  gets on their own data.
- `--mode reproduce`: overwrites grid.npz + prop.npz with baked training-time
  files after prepare. Recovers exact paper numbers (Rosetta-relaxed features).
  Small numerical improvement on top-tercile targets with tight pockets.

Outputs land in `results/chembl_bench_<mode>/` with per-target metrics + a
tercile summary printed to stdout.

## DUD-E-96 (external)

DUD-E raw data lives at http://dud.docking.org — we don't redistribute. Our
`data/dude_bench_manifest.tsv` ships the exact pocket centers we used.

```bash
# Download DUD-E once (~15 GB, ~30 min)
bash scripts/download_dude_raw.sh --output data/dude_raw/

# Run
bash scripts/run_dude_bench.sh \
    --checkpoint models/best.pkl \
    --raw-root data/dude_raw/ \
    --parallel 8
```

Output: per-target AUROC/EF@1%/BEDROC, comparable to numbers from Chen et al.
2019, Ragoza et al. 2017, etc. See `docs/dude_targets.md` for the target list.

## LIT-PCBA-14 (external)

Source: https://drugdesign.unistra.fr/LIT-PCBA/. We ship
`data/litpcba_bench_manifest.tsv`. Full benchmark has 15 targets; we run 14.
FEN1 is excluded because its LIT-PCBA-supplied mol2 conversion is incomplete
(no combined actives mol2 in the redistribution). All other 14 targets have
complete raw data.

```bash
# Download LIT-PCBA once (~40 GB, ~1 h)
bash scripts/download_litpcba_raw.sh --output data/litpcba_raw/

# Run
bash scripts/run_litpcba_bench.sh \
    --checkpoint models/best.pkl \
    --raw-root data/litpcba_raw/ \
    --parallel 8
```

## Custom targets

For your own screen (any set of receptor / ligand-library combinations):

```bash
# 1. Organize raw data
my_targets/
├── target1/
│   ├── receptor.pdb
│   ├── ligands.mol2 or ligands.sdf
│   └── crystal_ligand.mol2   (optional; enables auto center detection)
├── target2/
│   ...

# 2. Auto-generate manifest (or write manually)
python scripts/gen_manifest.py \
    --root my_targets/ \
    --output my_manifest.tsv \
    --center-mode crystal

# 3. Batch prepare
bash scripts/prepare_batch.sh \
    --manifest my_manifest.tsv \
    --raw-root my_targets/ \
    --output my_prepared/ \
    --parallel 8

# 4. Predict
uv run python motifscreen.py predict \
    --datapath my_prepared/ \
    --checkpoint models/best.pkl \
    --base-config configs/training/endtoend.yaml \
    --gpus 0 \
    --output my_scores.csv
```

`--center-mode` options:

- `crystal` (default): expects `<target>/crystal_ligand.mol2`, computes COM of
  heavy atoms.
- `ligand`: computes COM from the ligands file itself. Only sensible if the
  ligands are docked poses (already positioned in the pocket).
- `manual`: pass `--centers-file centers.tsv` with columns
  `target_id  center_x  center_y  center_z`.

## Notes on multi-GPU

`motifscreen.py predict --gpus 0,1,...` uses one model replica per GPU with
target-level parallelism. Sub-linear scaling above 2 GPUs (CPU featurization
becomes shared bottleneck). For most benchmarks 2 GPUs is a sweet spot.

Larger batch size helps but has memory ceiling. On A6000 (48 GB) `--batch-size
64` is safe; on V100 (32 GB) stay at `--batch-size 32`.
