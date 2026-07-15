#!/bin/bash
# Build the LIT-PCBA benchmark tarball for the combined benchmarks bundle.
#
# Similar to build_dude_bench_tarball.sh but with the LIT-PCBA raw layout.
#
# Expected input layout:
#   <PREPARED>/<target>/                       -- output of prepare
#   <RAW>/                                     -- LIT-PCBA prepared mol2s + receptor PDBs
#     <target>.active.mol2                     -- ground-truth actives
#     <target>.decoy.mol2                      -- decoys
#     (LIT-PCBA distributes SMILES; we assume the user staged mol2s in advance)
#
# Usage:
#   bash scripts/build_litpcba_bench_tarball.sh \
#       --prepared /home/j2ho/tmp/msk_litpcba_run/prepared \
#       --raw /home/j2ho/DB/LIT_PCBA/motifscreen \
#       --manifest data/litpcba_bench_manifest.tsv \
#       --out data/staging/litpcba_bench

set -e

PREPARED=""
RAW=""
MANIFEST=""
OUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --prepared) PREPARED="$2"; shift 2 ;;
        --raw) RAW="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 2 ;;
    esac
done

[ -z "$PREPARED" ] && { echo "ERROR: --prepared required"; exit 2; }
[ -z "$RAW" ] && { echo "ERROR: --raw required"; exit 2; }
[ -z "$MANIFEST" ] && { echo "ERROR: --manifest required"; exit 2; }
[ -z "$OUT" ] && { echo "ERROR: --out required"; exit 2; }

mkdir -p "$OUT/prepared"
cp "$MANIFEST" "$OUT/manifest.tsv"

n_ok=0
n_missing=0
: > "$OUT/build.log"

while IFS=$'\t' read -r target pdb cx cy cz n_act n_dec; do
    [ "$target" = "target_id" ] && continue
    src_prep="$PREPARED/$target"
    dst="$OUT/prepared/$target"

    if [ ! -d "$src_prep" ]; then
        echo "$target: missing prepared dir" >> "$OUT/build.log"
        n_missing=$((n_missing+1)); continue
    fi
    mkdir -p "$dst"

    for f in "$target.grid.npz" "$target.prop.npz" \
             all_ligands.mol2 all_ligands.keyatom.def.npz; do
        if [ -f "$src_prep/$f" ]; then
            cp "$src_prep/$f" "$dst/"
        else
            echo "$target: missing $f" >> "$OUT/build.log"
            n_missing=$((n_missing+1))
            continue 2
        fi
    done

    # Reference receptor
    if [ -f "$src_prep/receptor_h.pdb" ]; then
        cp "$src_prep/receptor_h.pdb" "$dst/receptor.pdb"
    fi

    # Labels: LIT-PCBA active mol2 (ship for convenience)
    for src_lbl in "$RAW/$target.active.mol2" "$RAW/$target/actives.mol2"; do
        if [ -f "$src_lbl" ]; then
            cp "$src_lbl" "$dst/actives.mol2"
            break
        fi
    done

    n_ok=$((n_ok+1))
done < "$MANIFEST"

cat > "$OUT/README.md" <<EOF
# LIT-PCBA benchmark, prepared for MotifScreen-Aff

$n_ok targets from LIT-PCBA (Tran-Nguyen et al. 2020, J Chem Inf Model).

## What's here

Per target under \`prepared/<target>/\`:
- \`receptor.pdb\`               Rosetta-relaxed receptor (protonated)
- \`<target>.grid.npz\`          Receptor grid
- \`<target>.prop.npz\`          Receptor properties
- \`all_ligands.mol2\`           Canonicalized MMFF94-charged mol2, actives + decoys
- \`all_ligands.keyatom.def.npz\` BRICS keyatoms
- \`actives.mol2\`               LIT-PCBA-supplied actives (labels)

Two targets from the original LIT-PCBA (FEN1, KAT2A) are excluded due to
incomplete raw data. See \`manifest.tsv\` for the exact 13 targets.

## Usage

\`\`\`bash
uv run python motifscreen.py predict \\
    --datapath data/litpcba_bench/prepared \\
    --checkpoint models/best.pkl \\
    --base-config configs/training/endtoend.yaml \\
    --gpus 0,1 \\
    --output results/litpcba_scores.csv
\`\`\`

## Original data source

LIT-PCBA: https://drugdesign.unistra.fr/LIT-PCBA/
Tran-Nguyen, Jacquemard, Rognan (2020). LIT-PCBA: An Unbiased Data Set for
Machine Learning and Virtual Screening. J Chem Inf Model 60(9):4263-4273.
https://doi.org/10.1021/acs.jcim.0c00155

License: Please refer to the original LIT-PCBA distribution terms.
Redistribution here is for the purpose of reproducing MotifScreen-Aff
benchmark numbers.
EOF

echo ""
echo "Staged $n_ok / $((n_ok + n_missing)) LIT-PCBA targets in $OUT"
echo "Size: $(du -sh $OUT | cut -f1)"
if [ "$n_missing" -gt 0 ]; then
    echo "$n_missing targets missing (see $OUT/build.log)"
fi
