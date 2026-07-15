#!/bin/bash
# End-to-end MotifScreen-Aff ChEMBL-LR benchmark (107 targets, tercile-balanced by AVE bias).
#
# Two modes:
#   --mode reproduce   Use baked grid+prop npz files from training (exact paper numbers).
#   --mode fresh       Full public prepare pipeline from scratch (default).
#                      ~0.03 lower mean AUROC on average with reduce vs Rosetta.
#
# Uses the portable scripts/prepare_batch.sh under the hood, so target-level
# parallelism comes with --parallel N (default 8).
#
# Usage:
#   bash scripts/download_and_prepare_chembl_bench.sh --checkpoint models/epoch70.pkl
#   bash scripts/download_and_prepare_chembl_bench.sh --checkpoint models/epoch70.pkl \
#        --mode reproduce --parallel 8

set -e

CHECKPOINT=""
MODE=fresh
GPU=0
SKIP_DOWNLOAD=0
SKIP_PREP=0
BATCH_SIZE=32
PARALLEL=8

while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        --skip-prep) SKIP_PREP=1; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 2 ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: --checkpoint is required"
    exit 2
fi

case "$MODE" in
    reproduce|fresh) ;;
    *) echo "ERROR: --mode must be 'reproduce' or 'fresh'"; exit 2 ;;
esac

ZENODO_URL="{{ZENODO_URL_PLACEHOLDER}}"
TARBALL="chembl_bench_v1.tar.gz"
SHA256="{{SHA256_PLACEHOLDER}}"

RAW=data/chembl_bench_raw
PREPARED=data/chembl_bench_prepared_${MODE}
RESULTS=results/chembl_bench_${MODE}
MANIFEST=data/chembl_bench_manifest.tsv
BATCH_MANIFEST=$RESULTS/batch_manifest.tsv
SCORES=$RESULTS/scores.csv
METRICS=$RESULTS/metrics.csv

mkdir -p "$RESULTS"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found at $MANIFEST"
    exit 2
fi

# ---- 1. Download + verify -----------------------------------------------
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
    fi
    echo "raw data: $(find $RAW -maxdepth 1 -mindepth 1 -type d | wc -l) target dirs"
    echo ""
fi

# ---- 2a. Preprocess: concat mol2 batches + reduce protonation ------------
if [ "$SKIP_PREP" -eq 0 ]; then
    echo "===== 2a. Preprocess per target (concat batch mol2s + reduce) ====="
    echo -e "target_id\treceptor_pdb\tligands\tcenter_x\tcenter_y\tcenter_z" > "$BATCH_MANIFEST"
    while IFS=$'\t' read -r target tercile pdb ligands cx cy cz n_mol2s; do
        [ "$target" = "target_id" ] && continue
        tgt_raw=$RAW/$target
        all_mol2=$tgt_raw/all_ligands.mol2
        : > "$all_mol2"
        IFS=',' read -ra M2S <<< "$ligands"
        for m in "${M2S[@]}"; do
            src=$tgt_raw/batch_mol2s/$m
            [ -f "$src" ] && cat "$src" >> "$all_mol2"
        done
        [ ! -s "$all_mol2" ] && continue
        # Reduce (protonate) receptor
        if command -v reduce >/dev/null; then
            reduce -BUILD -Quiet "$tgt_raw/$pdb" > "$tgt_raw/receptor_h.pdb" 2>/dev/null
            recpdb=receptor_h.pdb
        else
            cp "$tgt_raw/$pdb" "$tgt_raw/receptor_h.pdb"
            recpdb=receptor_h.pdb
        fi
        echo -e "$target\t$recpdb\tall_ligands.mol2\t$cx\t$cy\t$cz" >> "$BATCH_MANIFEST"
    done < "$MANIFEST"
    echo "batch manifest: $BATCH_MANIFEST ($(($(wc -l < $BATCH_MANIFEST) - 1)) targets)"
    echo ""

    # ---- 2b. Batch prepare (parallel via prepare_batch.sh) -----------------
    echo "===== 2b. prepare_batch.sh (parallel=$PARALLEL) ====="
    bash scripts/prepare_batch.sh \
        --manifest "$BATCH_MANIFEST" \
        --raw-root "$RAW" \
        --output "$PREPARED" \
        --parallel "$PARALLEL" \
        --extra-args "--skip-ligand-prep"
    echo ""

    # ---- 2c. Reproduce mode: overwrite grid/prop with baked training-time npz -
    if [ "$MODE" = "reproduce" ]; then
        echo "===== 2c. Overwrite grid+prop with baked training-time npz ====="
        while IFS=$'\t' read -r target tercile pdb ligands cx cy cz n_mol2s; do
            [ "$target" = "target_id" ] && continue
            [ ! -d "$PREPARED/$target" ] && continue
            if [ -f "$RAW/$target/$target.grid.npz" ]; then
                cp "$RAW/$target/$target.grid.npz" "$PREPARED/$target/"
                cp "$RAW/$target/$target.prop.npz" "$PREPARED/$target/"
            fi
        done < "$MANIFEST"
        echo ""
    fi
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
