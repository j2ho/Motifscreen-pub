#!/bin/bash
# Run the LIT-PCBA benchmark: prepare + predict + per-target enrichment.
#
# Assumes raw data was downloaded via scripts/download_litpcba_raw.sh, OR the
# user has an equivalent local mirror (SMILES + PDB templates per target).
#
# This script converts SMILES -> 3D mol2 via obabel and stages the per-target
# receptor.pdb + all_ligands.mol2 + crystal_ligand.mol2 layout, then calls
# scripts/prepare_batch.sh + predict.

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

MANIFEST=data/litpcba_bench_manifest.tsv
[ ! -f "$MANIFEST" ] && { echo "ERROR: $MANIFEST not found"; exit 2; }

PREPARED=data/litpcba_bench_prepared
RESULTS=results/litpcba_bench
STAGE=data/litpcba_bench_stage
BATCH_MANIFEST=$RESULTS/batch_manifest.tsv
SCORES=$RESULTS/scores.csv
METRICS=$RESULTS/metrics.csv

mkdir -p "$RESULTS" "$STAGE"

# ---- 1. Stage per-target files -------------------------------------------
# LIT-PCBA raw layout varies. Our manifest lists (target_id, pdb_name, cx, cy, cz).
# For each target we expect <RAW>/<target>/{<pdb_name>, actives.smi, inactives.smi}
# and produce a canonical staged layout with mol2s + receptor.
echo "===== 1. Stage LIT-PCBA raw -> mol2 per target ====="
echo -e "target_id\treceptor_pdb\tligands\tcenter_x\tcenter_y\tcenter_z" > "$BATCH_MANIFEST"

while IFS=$'\t' read -r target pdb cx cy cz n_act n_dec; do
    [ "$target" = "target_id" ] && continue
    src="$RAW/$target"
    stg="$STAGE/$target"
    mkdir -p "$stg"

    # Receptor
    if [ ! -f "$stg/receptor.pdb" ]; then
        if [ -f "$src/$pdb" ]; then
            cp "$src/$pdb" "$stg/receptor.pdb"
        elif [ -f "$src/protein.pdb" ]; then
            cp "$src/protein.pdb" "$stg/receptor.pdb"
        else
            echo "  $target: no receptor PDB, skip"
            continue
        fi
    fi

    # Convert SMILES to 3D mol2 (once per target)
    if [ ! -f "$stg/all_ligands.mol2" ]; then
        actives_smi="$src/actives.smi"
        inact_smi="$src/inactives.smi"
        if [ -f "$actives_smi" ] && [ -f "$inact_smi" ]; then
            obabel "$actives_smi" -O "$stg/actives.mol2" --gen3D -h 2>/dev/null
            obabel "$inact_smi" -O "$stg/inactives.mol2" --gen3D -h 2>/dev/null
            cat "$stg/actives.mol2" "$stg/inactives.mol2" > "$stg/all_ligands.mol2"
        elif [ -f "$src/all_ligands.mol2" ]; then
            cp "$src/all_ligands.mol2" "$stg/all_ligands.mol2"
        else
            echo "  $target: no ligand SMILES/mol2 sources, skip"
            continue
        fi
    fi

    # Reduce protonation
    if command -v reduce >/dev/null; then
        reduce -BUILD -Quiet "$stg/receptor.pdb" > "$stg/receptor_h.pdb" 2>/dev/null
        recpdb=receptor_h.pdb
    else
        cp "$stg/receptor.pdb" "$stg/receptor_h.pdb"
        recpdb=receptor_h.pdb
    fi

    echo -e "$target\t$recpdb\tall_ligands.mol2\t$cx\t$cy\t$cz" >> "$BATCH_MANIFEST"
done < "$MANIFEST"

n_staged=$(($(wc -l < "$BATCH_MANIFEST") - 1))
echo "  staged $n_staged targets"
echo ""

# ---- 2. Batch prepare ----------------------------------------------------
echo "===== 2. prepare_batch.sh (parallel=$PARALLEL) ====="
bash scripts/prepare_batch.sh \
    --manifest "$BATCH_MANIFEST" \
    --raw-root "$STAGE" \
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

# ---- 4. Metrics ----------------------------------------------------------
echo "===== 4. Metrics ====="
uv run python - <<EOF
import csv, os, numpy as np
from sklearn.metrics import roc_auc_score

def read_actives(target):
    ids = set()
    # LIT-PCBA staged: $STAGE/{target}/actives.smi (first column is name after conversion)
    p = f"$STAGE/{target}/actives.mol2"
    if not os.path.exists(p):
        return ids
    prev = ''
    with open(p) as f:
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
        if n_act == 0 or n_act == len(labels): continue
        a = roc_auc_score(labels, scores)
        k = max(1, int(0.01 * len(labels)))
        topk = np.argsort(-scores)[:k]
        ef1 = (labels[topk].sum() / k) / (n_act / len(labels))
        out.write(f"{t},{len(labels)},{n_act},{a:.4f},{ef1:.4f}\n")
        aurocs.append(a); ef1s.append(ef1)
        print(f"  {t:10} n={len(labels):>7} act={n_act:>5} AUROC={a:.3f} EF1={ef1:.2f}")

if aurocs:
    print()
    print(f"Mean AUROC across {len(aurocs)} targets: {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}")
    print(f"Mean EF@1%: {np.mean(ef1s):.2f}")
EOF
echo ""
echo "Done. metrics=$METRICS"
