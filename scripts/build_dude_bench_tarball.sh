#!/bin/bash
# Build the DUD-E benchmark tarball for the combined benchmarks bundle.
#
# Takes an already-prepared DUD-E directory + raw source + manifest, packages
# into a `dude_bench/` subtree suitable for the combined benchmarks tarball.
#
# Expected input layout:
#   <PREPARED>/<target>/                      -- output of prepare (grid+prop+keyatom+mol2)
#   <RAW>/<target>/                           -- DUD-E raw (for labels + reference PDB)
#     actives_final.mol2.gz
#     decoys_final.mol2.gz (optional, not shipped)
#     receptor.pdb
#     crystal_ligand.mol2
#
# Usage:
#   bash scripts/build_dude_bench_tarball.sh \
#       --prepared /home/j2ho/tmp/msk_dude_full_run/prepared \
#       --raw /home/j2ho/DB/dud-e \
#       --manifest data/dude_bench_manifest.tsv \
#       --out data/staging/dude_bench

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

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST"; exit 2
fi

mkdir -p "$OUT/prepared"

cp "$MANIFEST" "$OUT/manifest.tsv"

n_ok=0
n_missing=0
: > "$OUT/build.log"

while IFS=$'\t' read -r target cx cy cz; do
    [ "$target" = "target_id" ] && continue
    src_prep="$PREPARED/$target"
    src_raw="$RAW/$target"
    dst="$OUT/prepared/$target"

    if [ ! -d "$src_prep" ]; then
        echo "$target: missing prepared dir" >> "$OUT/build.log"
        n_missing=$((n_missing+1)); continue
    fi
    mkdir -p "$dst"

    # Prepared files (required)
    for f in "$target.grid.npz" "$target.prop.npz"; do
        if [ -f "$src_prep/$f" ]; then
            cp "$src_prep/$f" "$dst/"
        else
            echo "$target: missing $f" >> "$OUT/build.log"
            n_missing=$((n_missing+1))
            continue 2
        fi
    done
    for f in all_ligands.mol2 all_ligands.keyatom.def.npz; do
        if [ -f "$src_prep/$f" ]; then
            cp "$src_prep/$f" "$dst/"
        else
            echo "$target: missing $f" >> "$OUT/build.log"
            n_missing=$((n_missing+1))
            continue 2
        fi
    done

    # Reference receptor.pdb (Rosetta-relaxed if available, else raw)
    if [ -f "$src_prep/receptor_h.pdb" ]; then
        cp "$src_prep/receptor_h.pdb" "$dst/receptor.pdb"
    elif [ -f "$src_raw/receptor.pdb" ]; then
        cp "$src_raw/receptor.pdb" "$dst/receptor.pdb"
    fi

    # Labels: unpack actives_final.mol2 (public DUD-E, redistribute for convenience)
    if [ -f "$src_raw/actives_final.mol2.gz" ]; then
        gunzip -c "$src_raw/actives_final.mol2.gz" > "$dst/actives_final.mol2"
    elif [ -f "$src_raw/actives_final.mol2" ]; then
        cp "$src_raw/actives_final.mol2" "$dst/actives_final.mol2"
    fi

    n_ok=$((n_ok+1))
done < "$MANIFEST"

# Top-level README
cat > "$OUT/README.md" <<EOF
# DUD-E benchmark, prepared for MotifScreen-Aff

$n_ok targets from DUD-E (Mysinger et al. 2012, J Med Chem).

## What's here

Per target under \`prepared/<target>/\`:
- \`receptor.pdb\`               Rosetta-relaxed receptor (protonated)
- \`<target>.grid.npz\`          Baked receptor grid (from public-code prepare)
- \`<target>.prop.npz\`          Baked receptor properties
- \`all_ligands.mol2\`           Canonicalized, MMFF94-charged mol2 with all compounds
- \`all_ligands.keyatom.def.npz\` BRICS keyatoms
- \`actives_final.mol2\`         DUD-E-supplied actives (labels for metrics)

## Usage

Assuming this dir is at \`data/dude_bench/\`:

\`\`\`bash
uv run python motifscreen.py predict \\
    --datapath data/dude_bench/prepared \\
    --checkpoint models/best.pkl \\
    --base-config configs/training/endtoend.yaml \\
    --gpus 0,1 \\
    --output results/dude_scores.csv
bash scripts/run_dude_bench.sh --checkpoint models/best.pkl ... # for one-shot re-run
\`\`\`

## Original data source

DUD-E: http://dud.docking.org
Mysinger et al. (2012). Directory of useful decoys, enhanced (DUD-E).
J Med Chem 55(14):6582-94. https://doi.org/10.1021/jm300687e

License: DUD-E raw data is freely available for academic and commercial use.
This preparation is redistributed under the same terms.
EOF

echo ""
echo "Staged $n_ok / $((n_ok + n_missing)) DUD-E targets in $OUT"
echo "Size: $(du -sh $OUT | cut -f1)"
if [ "$n_missing" -gt 0 ]; then
    echo "$n_missing targets missing (see $OUT/build.log)"
fi
