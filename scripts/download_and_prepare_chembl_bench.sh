#!/bin/bash
# End-to-end MotifScreen-Aff ChEMBL-LR benchmark (107 targets, tercile-balanced by AVE bias).
#
# Downloads the raw benchmark tarball from Zenodo, runs the SAME prepare + predict CLI
# documented in INFERENCE.md across all 107 targets, then reports per-tercile AUROC.
#
# Usage:
#   bash scripts/download_and_prepare_chembl_bench.sh --checkpoint models/epoch70.pkl
#
# Options:
#   --checkpoint PATH   Model checkpoint (.pkl). Required.
#   --gpu ID            GPU id (default: 0)
#   --skip-download     Assume data/chembl_bench_raw/ already unpacked
#   --skip-prep         Assume data/chembl_bench_prepared/ already produced
#   --batch-size N      Predict batch size (default: 32)

set -e

CHECKPOINT=""
GPU=0
SKIP_DOWNLOAD=0
SKIP_PREP=0
BATCH_SIZE=32

while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
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

# Fill in when the Zenodo record is created
ZENODO_URL="{{ZENODO_URL_PLACEHOLDER}}"
TARBALL="chembl_bench_v1.tar.gz"
SHA256="{{SHA256_PLACEHOLDER}}"

RAW=data/chembl_bench_raw
PREPARED=data/chembl_bench_prepared
RESULTS=results/chembl_bench
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
        # Optional integrity check
        if [ "$SHA256" != "{{SHA256_PLACEHOLDER}}" ]; then
            echo "$SHA256  data/$TARBALL" | sha256sum -c -
        fi
        tar xzf "data/$TARBALL" -C data/
        # tarball unpacks into data/chembl_bench_raw/<target>/{receptor.pdb,batch_mol2s/...,active_smiles_clu.csv}
    fi
    echo "raw data: $(find $RAW -maxdepth 1 -mindepth 1 -type d | wc -l) target dirs"
    echo ""
fi

# ---- 2. Prepare per target ----------------------------------------------
if [ "$SKIP_PREP" -eq 0 ]; then
    echo "===== 2. Prepare per target ====="
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
        # Protonate receptor (users can substitute their own protonation tool)
        if command -v reduce >/dev/null; then
            reduce -BUILD -Quiet "$tgt_raw/$pdb" > "$tgt_raw/receptor_h.pdb" 2>/dev/null
        else
            cp "$tgt_raw/$pdb" "$tgt_raw/receptor_h.pdb"
        fi
        # Prepare - identical to the CLI a public user runs on their own target
        uv run python motifscreen.py prepare \
            --protein "$tgt_raw/receptor_h.pdb" \
            --ligands "$all_mol2" \
            --center="$cx,$cy,$cz" \
            --target-id "$target" \
            --output "$PREPARED" \
            --skip-ligand-prep \
            --workers 4 > /dev/null 2>&1
        if [ -f "$PREPARED/$target/$target.grid.npz" ]; then
            prep_ok=$((prep_ok+1))
        else
            echo "$target PREPARE_FAIL" >> "$RESULTS/failures.log"
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
echo "===== 4. Per-tercile metrics ====="
uv run python scripts/inference/compute_tercile_metrics.py \
    --scores "$SCORES" \
    --labels "$RAW" \
    --manifest "$MANIFEST" \
    --metrics-out "$METRICS"

echo ""
echo "Done. scores=$SCORES metrics=$METRICS"
