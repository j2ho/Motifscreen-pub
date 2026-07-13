#!/bin/bash
# End-to-end MotifScreen-Aff ChEMBL-LR benchmark (107 targets, tercile-balanced by AVE bias).
#
# Two modes:
#   --mode reproduce   Use baked grid+prop npz files from training. Reproduces
#                      published paper numbers exactly. Skips reduce+featurize.
#   --mode fresh       Run the full public prepare pipeline (reduce + featurize)
#                      from raw PDB. Shows what a user on their own target gets.
#                      ~0.03 lower mean AUROC on average; larger drops on tight-
#                      pocket / metalloenzyme targets. Default.
#
# Usage:
#   bash scripts/download_and_prepare_chembl_bench.sh --checkpoint models/epoch70.pkl
#   bash scripts/download_and_prepare_chembl_bench.sh --checkpoint models/epoch70.pkl --mode reproduce
#
# Options:
#   --checkpoint PATH    Model checkpoint (.pkl). Required.
#   --mode MODE          reproduce | fresh (default: fresh)
#   --gpu ID             GPU id (default: 0)
#   --skip-download      Assume data/chembl_bench_raw/ already unpacked
#   --skip-prep          Assume data/chembl_bench_prepared/ already produced
#   --batch-size N       Predict batch size (default: 32)

set -e

CHECKPOINT=""
MODE=fresh
GPU=0
SKIP_DOWNLOAD=0
SKIP_PREP=0
BATCH_SIZE=32

while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        --skip-prep) SKIP_PREP=1; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 2 ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: --checkpoint is required"
    exit 2
fi

case "$MODE" in
    reproduce|fresh) ;;
    *) echo "ERROR: --mode must be 'reproduce' or 'fresh', got '$MODE'"; exit 2 ;;
esac

# Fill in when the Zenodo record is created
ZENODO_URL="{{ZENODO_URL_PLACEHOLDER}}"
TARBALL="chembl_bench_v1.tar.gz"
SHA256="{{SHA256_PLACEHOLDER}}"

RAW=data/chembl_bench_raw
PREPARED=data/chembl_bench_prepared_${MODE}
RESULTS=results/chembl_bench_${MODE}
MANIFEST=data/chembl_bench_manifest.tsv
SCORES=$RESULTS/scores.csv
METRICS=$RESULTS/metrics.csv

mkdir -p "$RESULTS"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found at $MANIFEST (should be in the public repo)"
    exit 2
fi

# ---- 1. Download + verify ------------------------------------------------
if [ "$SKIP_DOWNLOAD" -eq 0 ]; then
    echo "===== 1. Download ChEMBL benchmark tarball ====="
    if [ ! -d "$RAW" ]; then
        mkdir -p data
        if [ ! -f "data/$TARBALL" ]; then
            curl -L "$ZENODO_URL/$TARBALL" -o "data/$TARBALL"
        fi
        if [ "$SHA256" != "{{SHA256_PLACEHOLDER}}" ]; then
            echo "$SHA256  data/$TARBALL" | sha256sum -c -
        fi
        tar xzf "data/$TARBALL" -C data/
        # Tarball unpacks into:
        #   data/chembl_bench_raw/<target>/
        #     receptor.pdb                       (Rosetta-relaxed, already protonated)
        #     <target>.grid.npz                  (baked from training pipeline)
        #     <target>.prop.npz                  (baked from training pipeline)
        #     active_smiles_clu.csv              (ground-truth actives, for metrics)
        #     batch_mol2s/*.mol2                 (compound batches, 1 active + 30 decoys each)
    fi
    echo "raw data: $(find $RAW -maxdepth 1 -mindepth 1 -type d | wc -l) target dirs"
    echo ""
fi

# ---- 2. Prepare per target ----------------------------------------------
if [ "$SKIP_PREP" -eq 0 ]; then
    echo "===== 2. Prepare per target (mode=$MODE) ====="
    prep_ok=0
    prep_fail=0
    : > "$RESULTS/failures.log"
    while IFS=$'\t' read -r target tercile pdb ligands cx cy cz n_mol2s; do
        [ "$target" = "target_id" ] && continue
        tgt_raw=$RAW/$target

        # Concatenate the manifested mol2 batches
        all_mol2=$tgt_raw/all_ligands.mol2
        : > "$all_mol2"
        IFS=',' read -ra M2S <<< "$ligands"
        for m in "${M2S[@]}"; do
            src=$tgt_raw/batch_mol2s/$m
            [ -f "$src" ] && cat "$src" >> "$all_mol2"
        done
        if [ ! -s "$all_mol2" ]; then
            echo "$target NO_MOL2S" >> "$RESULTS/failures.log"
            prep_fail=$((prep_fail+1))
            continue
        fi

        prep_dir=$PREPARED/$target
        mkdir -p "$prep_dir"
        cp "$all_mol2" "$prep_dir/all_ligands.mol2"

        if [ "$MODE" = "reproduce" ]; then
            # Run full prepare (produces mol2 copy + keyatom.def.npz + fresh grid/prop),
            # then overwrite the fresh grid/prop with baked versions from training.
            if [ ! -f "$tgt_raw/$target.grid.npz" ] || [ ! -f "$tgt_raw/$target.prop.npz" ]; then
                echo "$target MISSING_BAKED_NPZ" >> "$RESULTS/failures.log"
                prep_fail=$((prep_fail+1))
                continue
            fi
            uv run python motifscreen.py prepare \
                --protein "$tgt_raw/receptor.pdb" \
                --ligands "$all_mol2" \
                --center="$cx,$cy,$cz" \
                --target-id "$target" \
                --output "$PREPARED" \
                --skip-ligand-prep --workers 4 > /dev/null 2>&1
            cp "$tgt_raw/$target.grid.npz" "$prep_dir/$target.grid.npz"
            cp "$tgt_raw/$target.prop.npz" "$prep_dir/$target.prop.npz"
        else
            # fresh mode: reduce + featurize from scratch
            if command -v reduce >/dev/null; then
                reduce -BUILD -Quiet "$tgt_raw/receptor.pdb" > "$tgt_raw/receptor_h.pdb" 2>/dev/null
            else
                cp "$tgt_raw/receptor.pdb" "$tgt_raw/receptor_h.pdb"
            fi
            uv run python motifscreen.py prepare \
                --protein "$tgt_raw/receptor_h.pdb" \
                --ligands "$all_mol2" \
                --center="$cx,$cy,$cz" \
                --target-id "$target" \
                --output "$PREPARED" \
                --skip-ligand-prep --workers 4 > /dev/null 2>&1
        fi

        if [ -f "$prep_dir/$target.grid.npz" ] && ls "$prep_dir/"*.keyatom.def.npz > /dev/null 2>&1; then
            prep_ok=$((prep_ok+1))
        else
            echo "$target PREPARE_FAIL($MODE)" >> "$RESULTS/failures.log"
            prep_fail=$((prep_fail+1))
        fi
    done < "$MANIFEST"
    echo "prepare done: ok=$prep_ok fail=$prep_fail"
    echo ""
fi

# ---- 3. Predict ----------------------------------------------------------
echo "===== 3. Predict across all prepared targets ====="
TARGETS=$(ls "$PREPARED")
uv run python motifscreen.py predict \
    --datapath "$PREPARED" \
    --targets $TARGETS \
    --checkpoint "$CHECKPOINT" \
    --base-config configs/training/endtoend.yaml \
    --gpus "$GPU" \
    --batch-size "$BATCH_SIZE" \
    --output "$SCORES"
echo ""

# ---- 4. Metrics ----------------------------------------------------------
echo "===== 4. Per-tercile metrics (mode=$MODE) ====="
uv run python scripts/inference/compute_tercile_metrics.py \
    --scores "$SCORES" \
    --labels "$RAW" \
    --manifest "$MANIFEST" \
    --metrics-out "$METRICS"

echo ""
echo "Mode: $MODE"
echo "Scores:  $SCORES"
echo "Metrics: $METRICS"
