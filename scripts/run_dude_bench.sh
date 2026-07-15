#!/bin/bash
# Run the DUD-E benchmark: prepare + predict + per-target enrichment.
#
# Assumes DUD-E raw data was downloaded via scripts/download_dude_raw.sh.
#
# Usage:
#   bash scripts/run_dude_bench.sh \
#       --checkpoint models/best.pkl \
#       --raw-root data/dude_raw/ \
#       --parallel 8

set -e

CHECKPOINT=""
RAW=""
GPU=0
BATCH_SIZE=32
PARALLEL=8

while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --raw-root) RAW="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 2 ;;
    esac
done

[ -z "$CHECKPOINT" ] && { echo "ERROR: --checkpoint is required"; exit 2; }
[ -z "$RAW" ] && { echo "ERROR: --raw-root is required"; exit 2; }

MANIFEST=data/dude_bench_manifest.tsv
[ ! -f "$MANIFEST" ] && { echo "ERROR: $MANIFEST not found"; exit 2; }

PREPARED=data/dude_bench_prepared
RESULTS=results/dude_bench
BATCH_MANIFEST=$RESULTS/batch_manifest.tsv
SCORES=$RESULTS/scores.csv
METRICS=$RESULTS/metrics.csv

mkdir -p "$RESULTS"

# ---- 1. Preprocess: gunzip mol2s, concat actives+decoys, reduce protonation ---
echo "===== 1. Preprocess per target ====="
echo -e "target_id\treceptor_pdb\tligands\tcenter_x\tcenter_y\tcenter_z" > "$BATCH_MANIFEST"
n_targets=0
while IFS=$'\t' read -r target cx cy cz; do
    [ "$target" = "target_id" ] && continue
    tdir=$RAW/$target
    [ ! -d "$tdir" ] && { echo "  $target: no raw dir, skip"; continue; }
    # Gunzip if not already
    for f in actives_final decoys_final; do
        [ ! -f "$tdir/$f.mol2" ] && [ -f "$tdir/$f.mol2.gz" ] && \
            gunzip -c "$tdir/$f.mol2.gz" > "$tdir/$f.mol2"
    done
    # Concat
    cat "$tdir/actives_final.mol2" "$tdir/decoys_final.mol2" > "$tdir/all_ligands.mol2"
    # Reduce
    if command -v reduce >/dev/null; then
        reduce -BUILD -Quiet "$tdir/receptor.pdb" > "$tdir/receptor_h.pdb" 2>/dev/null
    else
        cp "$tdir/receptor.pdb" "$tdir/receptor_h.pdb"
    fi
    echo -e "$target\treceptor_h.pdb\tall_ligands.mol2\t$cx\t$cy\t$cz" >> "$BATCH_MANIFEST"
    n_targets=$((n_targets+1))
done < "$MANIFEST"
echo "  $n_targets targets staged"
echo ""

# ---- 2. Batch prepare ----------------------------------------------------
echo "===== 2. prepare_batch.sh (parallel=$PARALLEL) ====="
bash scripts/prepare_batch.sh \
    --manifest "$BATCH_MANIFEST" \
    --raw-root "$RAW" \
    --output "$PREPARED" \
    --parallel "$PARALLEL" \
    --extra-args "--skip-ligand-prep"
echo ""

# ---- 3. Predict ----------------------------------------------------------
echo "===== 3. Predict ====="
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

# ---- 4. Metrics (per-target AUROC + EF1) ---------------------------------
echo "===== 4. Metrics ====="
uv run python - <<EOF
import csv, gzip, os, numpy as np
from sklearn.metrics import roc_auc_score

def read_actives(target):
    p = f"$RAW/{target}/actives_final.mol2"
    if not os.path.exists(p):
        p = f"$RAW/{target}/actives_final.mol2.gz"
        opener = lambda x: gzip.open(x, 'rt')
    else:
        opener = open
    ids = set()
    prev = ''
    with opener(p) as f:
        for line in f:
            if prev.startswith('@<TRIPOS>MOLECULE'):
                ids.add(line.strip())
            prev = line
    return ids

per_target = {}
with open("$SCORES") as f:
    for r in csv.DictReader(f):
        per_target.setdefault(r['target_id'], []).append((float(r['score']), r['compound_id']))

with open("$METRICS", 'w') as out:
    out.write("target_id,n,n_actives,AUROC,EF1\n")
    aurocs, ef1s = [], []
    for t, entries in sorted(per_target.items()):
        actives = read_actives(t)
        scores = np.array([e[0] for e in entries])
        labels = np.array([1 if e[1] in actives else 0 for e in entries])
        n_act = int(labels.sum())
        if n_act == 0 or n_act == len(labels):
            continue
        a = roc_auc_score(labels, scores)
        k = max(1, int(0.01 * len(labels)))
        topk = np.argsort(-scores)[:k]
        ef1 = (labels[topk].sum() / k) / (n_act / len(labels))
        out.write(f"{t},{len(labels)},{n_act},{a:.4f},{ef1:.4f}\n")
        aurocs.append(a); ef1s.append(ef1)
        print(f"  {t:10} n={len(labels):>6} act={n_act:>4} AUROC={a:.3f} EF1={ef1:.2f}")

if aurocs:
    print()
    print(f"Mean AUROC across {len(aurocs)} targets: {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}")
    print(f"Mean EF@1%: {np.mean(ef1s):.2f}")
    print(f"AUROC > 0.7: {sum(1 for a in aurocs if a > 0.7)}/{len(aurocs)}")
EOF
echo ""
echo "Done. metrics=$METRICS"
