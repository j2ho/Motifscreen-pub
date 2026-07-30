# Benchmark Reproduction

Three benchmarks: **ChEMBL-LR-107** (our contribution, tercile-balanced by AVE bias), **DUD-E** (100 targets), **LIT-PCBA** (13 targets).

Two paths to reproduce:
- **A. Download the pre-featurized tarball** — fastest, paper-matching numbers, no user-side prep needed
- **B. Run prepare + predict from scratch** — validates the full public pipeline on your machine

Use path A if you just want to reproduce numbers. Use path B if you want to see what a fresh user run looks like end-to-end.

---

## Path A: pre-featurized tarball

Download `motifscreen_aff_benchmarks_v2.tar.gz` (~3.9 GB) from Zenodo, DOI [`10.5281/zenodo.<BENCH_DOI>`](https://doi.org/10.5281/zenodo.<BENCH_DOI>) (TODO: fill at publish time).

```bash
wget https://zenodo.org/record/<BENCH_DOI>/files/motifscreen_aff_benchmarks_v2.tar.gz
echo "df2d57c7c2b4942af24216e6ed993d23c42ba3403cae9fdc42c41cae49d9f753  motifscreen_aff_benchmarks_v2.tar.gz" | sha256sum -c
tar xzf motifscreen_aff_benchmarks_v2.tar.gz
```

Contents:
```
motifscreen_aff_benchmarks/
├── chembl_bench/chembl_bench_raw/<target>/     # 107 ChEMBL-LR-107 targets (see note below)
├── dude_bench/prepared/<target>/                # 100 DUD-E targets
├── dude_bench/labels/<target>.actives.txt
├── litpcba_bench/prepared/<target>/             # 13 LIT-PCBA targets
├── litpcba_bench/labels/<target>.actives.txt
└── README.md
```

### DUD-E and LIT-PCBA — predict directly

Both ship predict-ready (baked grid + prop + all_ligands.mol2 + keyatom.def):

```bash
# DUD-E (100 targets, ~1.5M compounds; ~2-3 hr on 2 GPUs)
uv run python motifscreen.py predict \
    --datapath motifscreen_aff_benchmarks/dude_bench/prepared \
    --checkpoint models/epoch70.pkl \
    --base-config configs/training/endtoend.yaml \
    --gpus 0,1 \
    --output dude_scores.csv

# LIT-PCBA (13 targets, ~2.3M compounds; ~3-4 hr on 2 GPUs)
uv run python motifscreen.py predict \
    --datapath motifscreen_aff_benchmarks/litpcba_bench/prepared \
    --checkpoint models/epoch70.pkl \
    --base-config configs/training/endtoend.yaml \
    --gpus 0,1 \
    --output litpcba_scores.csv
```

Per-target labels for AUROC / EF@1% computation are in `<benchmark>/labels/<target>.actives.txt` (one active compound ID per line).

### ChEMBL-LR-107 — one prep step then predict

ChEMBL-LR ships "raw" (per-compound batch mol2s under `batch_mol2s/`, no `all_ligands.keyatom.def.npz`). Users run one prepare step to compute BRICS keyatoms, then predict:

```bash
bash scripts/download_and_prepare_chembl_bench.sh \
    --mode reproduce \
    --input-dir motifscreen_aff_benchmarks/chembl_bench/chembl_bench_raw \
    --output motifscreen_aff_benchmarks/chembl_bench/prepared \
    --parallel 8

uv run python motifscreen.py predict \
    --datapath motifscreen_aff_benchmarks/chembl_bench/prepared \
    --checkpoint models/epoch70.pkl \
    --base-config configs/training/endtoend.yaml \
    --gpus 0,1 \
    --output chembl_scores.csv
```

Tercile labels (AVE-bias top/mid/bottom): `data/chembl_lr_tercile_labels.tsv`.

### Metrics

`results/analysis/compare_benchmarks.py` (in the repo) takes a scores CSV + a labels file and prints per-target AUROC + EF@1% + BEDROC. See `--help`.

---

## Path B: from-scratch reproduction

Runs the same `reduce → prepare → predict` pipeline a first-time user would run on their own targets. Useful for validating the pipeline end-to-end, or for benchmarking with a checkpoint you trained yourself.

### Prerequisites

- `reduce` binary on `$PATH` (`apt install reduce` on Debian/Ubuntu, or download from kinemage.biochem.duke.edu)
- `obabel` binary — installed via `uv sync --extra preprocessing`
- Model checkpoint from Zenodo

### DUD-E-100 from scratch

```bash
# 1. Download DUD-E raw (~15 GB, ~30 min)
bash scripts/download_dude_raw.sh --output data/dude_raw/

# 2. Prep + predict (single script)
bash scripts/run_dude_bench.sh \
    --checkpoint models/epoch70.pkl \
    --raw-root data/dude_raw/ \
    --parallel 8
```

Output at `results/dude_bench/`. Two of the canonical 102 DUD-E targets (`fgfr1`, `kif11`) are excluded due to incomplete raw data in the DUD-E distribution.

### LIT-PCBA-14 from scratch

```bash
# 1. Download LIT-PCBA raw (~40 GB, ~1 h)
bash scripts/download_litpcba_raw.sh --output data/litpcba_raw/

# 2. Prep + predict
bash scripts/run_litpcba_bench.sh \
    --checkpoint models/epoch70.pkl \
    --raw-root data/litpcba_raw/ \
    --parallel 8
```

FEN1 is excluded because its LIT-PCBA-supplied mol2 conversion is incomplete. From-scratch reproduction gives all 14 remaining targets (MAPK1 works here — the v2 tarball only drops MAPK1 because its baked grid was unavailable at packaging time).

### ChEMBL-LR-107 from scratch

```bash
bash scripts/download_and_prepare_chembl_bench.sh \
    --checkpoint models/epoch70.pkl \
    --mode fresh \
    --parallel 8
```

Two modes:
- `--mode fresh` (default): full reduce → featurize → predict. Reproduces the pipeline a public user gets on their own data.
- `--mode reproduce`: overwrites the freshly-computed grid.npz + prop.npz with baked training-time files after prepare. Recovers exact paper numbers (matches tarball path A).

### From-scratch vs tarball delta

Running from scratch with `reduce` protonation gives slightly lower numbers than the baked tarball on some benchmarks:

| Benchmark | Reduce-vs-baked mean ΔAUROC | Cause |
|---|---|---|
| ChEMBL-LR-107 | ~-0.00 to -0.02 | Same-pipeline reproduction |
| DUD-E | ~-0.03 | Baked features used `score_jd2 -no_optH false` which assigns His tautomer states (HIE/HID/HIP) that reduce doesn't; slightly different pocket-atom charge assignment |
| LIT-PCBA | ~-0.007 | Baked features also used reduce-like protonation; delta is essentially numerical noise |

If you need exact paper-matching numbers on DUD-E specifically, use the tarball (path A).

---

## Custom targets

For your own screen (arbitrary receptor + ligand-library combinations):

```
my_targets/
├── target1/
│   ├── receptor.pdb            # already protonated
│   ├── ligands.mol2 or .sdf
│   └── crystal_ligand.mol2     # optional; used for auto-center detection
├── target2/
│   ...
```

Generate a manifest, batch-prepare, predict:

```bash
python scripts/gen_manifest.py \
    --root my_targets/ \
    --output my_manifest.tsv \
    --center-mode crystal

bash scripts/prepare_batch.sh \
    --manifest my_manifest.tsv \
    --raw-root my_targets/ \
    --output my_prepared/ \
    --parallel 8

uv run python motifscreen.py predict \
    --datapath my_prepared/ \
    --checkpoint models/epoch70.pkl \
    --base-config configs/training/endtoend.yaml \
    --gpus 0,1 \
    --output my_scores.csv
```

`--center-mode` options:
- `crystal` (default): expects `<target>/crystal_ligand.mol2`, uses COM
- `ligand`: uses COM of the ligand library file itself (only meaningful if compounds are already docked poses)
- `manual`: pass `--centers-file centers.tsv` with columns `target_id  center_x  center_y  center_z`

---

## Multi-GPU notes

Predict uses one model replica per GPU with target-level parallelism. Empirical scaling:

| GPUs | Aggregate throughput | Note |
|---|---|---|
| 1 | 40-60 compounds/sec | baseline |
| 2 | 70-90 c/s | sweet spot, ~1.7× |
| 3+ | ~100-120 c/s | diminishing returns (CPU featurization saturates) |

Batch size caps by VRAM:
- A6000 (48 GB): `--batch-size 64`
- V100 (32 GB): `--batch-size 32`
- T4 / smaller (16 GB): `--batch-size 16`

For libraries >~100k compounds per target, the mol2 reader streams internally so memory stays bounded regardless of file size.
