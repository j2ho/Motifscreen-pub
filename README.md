<p align="center">
  <img src="assets/motifscreen-logo.png" alt="MotifScreen" width="450"/>
</p>


SE(3)-equivariant structure-based virtual screening model. Given a protein pocket + a compound library, it ranks compounds by predicted binding likelihood. Combines a receptor grid encoder (SE(3)-equivariant transformer) with a ligand GAT encoder, linked by trigonal attention.

## Install

Requires Python 3.9, CUDA 11.7+ compatible GPU (or CPU fallback), and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/j2ho/Motifscreen-pub.git
cd Motifscreen-pub
uv sync --extra preprocessing
```

The `preprocessing` extra pulls in RDKit + openbabel-wheel, both required for the `prepare` step. See [INSTALL.md](INSTALL.md) for CUDA-version alternatives and troubleshooting.

## Quick inference

Two commands: `prepare` featurizes a protein + a compound library once; `predict` scores compounds against the prepared receptor.

```bash
# 1. Add hydrogens to your protein PDB (reduce is easiest)
reduce -BUILD receptor.pdb > receptor_h.pdb

# 2. Prepare: protein + compound library -> model input
uv run python motifscreen.py prepare \
    --protein receptor_h.pdb \
    --ligands compounds.sdf \
    --center=12.5,34.2,8.7 \
    --output prepared/

# 3. Predict: score compounds
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

Score is in range [0, 1] where higher score is more likely to be a binder. Score should be evaluated as relative likeliness to be a binder within the same target. (e.g. target A could have active score 0.3 and decoy score 0.1 where target B could have active score of 0.7 and decoy score of 0.5) 

Model checkpoint + `endtoend.yaml` config: checkpoint to be uploaded. 


## Benchmark reproduction

Download the pre-featurized benchmark tarball (~3.9 GB, ChEMBL-LR-107 + DUD-E-100 + LIT-PCBA-13) from Zenodo and run `predict` directly — no user-side prep needed:

```bash
# Fetch bench + verify
wget https://zenodo.org/records/21371299/files/motifscreen_benchmarks.tar.gz
echo "df2d57c7c2b4942af24216e6ed993d23c42ba3403cae9fdc42c41cae49d9f753  motifscreen_benchmarks.tar.gz" | sha256sum -c
tar xzf motifscreen_benchmarks.tar.gz

# Score DUD-E
uv run python motifscreen.py predict \
    --datapath motifscreen_benchmarks/dude_bench/prepared \
    --checkpoint {path_to_model_chkpt} \
    --base-config configs/training/endtoend.yaml \
    --gpus 0,1 \
    --output dude_scores.csv
```

Full reproduction scripts + benchmark-specific notes: see [BENCHMARKS.md](BENCHMARKS.md).

## Model

- SE(3)-equivariant transformer for the receptor pocket grid (~800 grid points)
- GAT for ligand molecular graph
- Trigonal attention crossover between the two
- Outputs: motif labels on the grid, predicted key-atom positions, and a binding score for ranking

Architecture details, loss composition, and training-time inputs: see `docs/`.

## ChEMBL data curation

The model was trained on a curated ChEMBL34-derived dataset along with PDBbind and BioLip data. If you want to use the ChEMBL dataset for training, but with the newer ChEMBL releases, the curation pipeline is a separate repository: [j2ho/chembl-q](https://github.com/j2ho/chembl-q). It filters ChEMBL activity data, selects artificial decoys with reasonable compound/target criteria, and provides a **leakage-resistant train/test split by receptor sequence similarity** against a supplied training-set FASTA (PDBbind + BioLip by default, easily swappable). Running chembl-q with the default settings reproduces the ChEMBL-LR benchmark this repo uses (may differ slightly from the original due to updates in ChEMBL).

## Repository layout

```
Motifscreen-pub/
├── motifscreen.py                # CLI entry (prepare + predict subcommands)
├── src/
│   ├── cli/                      # prepare.py, predict.py
│   ├── data/                     # dataset, mol2 parsing, SASA, graph building
│   ├── model/                    # SE(3) + GAT + trigon attention
│   └── io/                       # protein/ligand file utilities
├── scripts/
│   ├── prepare_batch.sh          # multi-target parallel prepare
│   ├── gen_manifest.py           # auto-manifest from a directory of targets
│   ├── run_dude_bench.sh         # DUD-E-100 reproduction
│   ├── run_litpcba_bench.sh      # LIT-PCBA-14 reproduction
│   └── download_and_prepare_chembl_bench.sh
├── configs/
│   ├── training/endtoend.yaml    # architecture config (also needed by predict)
│   └── inference/                # benchmark configs
├── data/
│   ├── rosetta_params/           # bundled residue-type params (no Rosetta install needed)
│   ├── generic_potential/        # ligand atom typing
│   ├── dude_bench_manifest.tsv
│   ├── litpcba_bench_manifest.tsv
│   └── chembl_bench_manifest.tsv
├── docs/
│   ├── model-architecture.md
│   ├── input-features.md
│   ├── outputs.md
│   └── losses.md
├── INSTALL.md
├── INFERENCE.md
└── BENCHMARKS.md
```

