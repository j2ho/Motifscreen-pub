#!/bin/bash
# Repackage a flat training-time bundle into the per-target subdir layout
# expected by motifscreen predict. Ships the exact features used to generate
# the paper numbers (baked Rosetta-relaxed grid/prop + training-time keyatoms).
#
# Input layout (flat):
#   <FLAT>/<target>.grid.npz
#   <FLAT>/<target>.prop.npz
#   <FLAT>/<target>.keyatom.def.npz
#   <FLAT>/<target>.active.mol2
#   <FLAT>/<target>.decoy_N.mol2 ...
#   dude:    <FLAT>/<target>.receptor.pdb
#   litpcba: <FLAT>/<manifest.pdb column>       (e.g. 3p0g_protein.pdb)
#
# Output layout:
#   <OUT>/prepared/<target>/receptor.pdb
#   <OUT>/prepared/<target>/<target>.grid.npz
#   <OUT>/prepared/<target>/<target>.prop.npz
#   <OUT>/prepared/<target>/<target>.keyatom.def.npz
#   <OUT>/prepared/<target>/all_ligands.mol2     (cat active + decoy_*)
#   <OUT>/labels/<target>.actives.txt            (compound IDs, one per line)
#   <OUT>/manifest.tsv
#   <OUT>/README.md
#
# Usage:
#   bash scripts/build_bench_from_training_bundle.sh \
#       --benchmark dude \
#       --flat-dir /home/j2ho/DB/dud-e/motifscreen \
#       --manifest data/dude_bench_manifest.tsv \
#       --out /home/j2ho/tmp/bench_staging/dude_bench

set -e

BENCH=""
FLAT=""
MANIFEST=""
OUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --benchmark) BENCH="$2"; shift 2 ;;
        --flat-dir)  FLAT="$2";  shift 2 ;;
        --manifest)  MANIFEST="$2"; shift 2 ;;
        --out)       OUT="$2";   shift 2 ;;
        *) echo "Unknown flag: $1"; exit 2 ;;
    esac
done

[ -z "$BENCH" ]    && { echo "ERROR: --benchmark {dude|litpcba} required"; exit 2; }
[ -z "$FLAT" ]     && { echo "ERROR: --flat-dir required"; exit 2; }
[ -z "$MANIFEST" ] && { echo "ERROR: --manifest required"; exit 2; }
[ -z "$OUT" ]      && { echo "ERROR: --out required"; exit 2; }

[ ! -d "$FLAT" ]     && { echo "ERROR: flat dir not found: $FLAT"; exit 2; }
[ ! -f "$MANIFEST" ] && { echo "ERROR: manifest not found: $MANIFEST"; exit 2; }

case "$BENCH" in
    dude|litpcba) ;;
    *) echo "ERROR: --benchmark must be 'dude' or 'litpcba'"; exit 2 ;;
esac

mkdir -p "$OUT/prepared" "$OUT/labels"
cp "$MANIFEST" "$OUT/manifest.tsv"

: > "$OUT/build.log"
n_ok=0
n_missing=0
n_no_decoys=0

extract_ids() {
    # Extract compound IDs from a mol2 (the line immediately after @<TRIPOS>MOLECULE).
    awk '/^@<TRIPOS>MOLECULE/ {getline id; print id}' "$1" | tr -d '\r'
}

while IFS=$'\t' read -r col1 col2 rest; do
    [ "$col1" = "target_id" ] && continue
    [ -z "$col1" ] && continue

    target="$col1"
    if [ "$BENCH" = "litpcba" ]; then
        pdb_file="$col2"
    fi

    dst="$OUT/prepared/$target"

    # Required baked features
    grid="$FLAT/${target}.grid.npz"
    prop="$FLAT/${target}.prop.npz"
    keya="$FLAT/${target}.keyatom.def.npz"
    activ="$FLAT/${target}.active.mol2"

    missing=0
    for f in "$grid" "$prop" "$keya" "$activ"; do
        if [ ! -f "$f" ]; then
            echo "$target: missing $(basename $f)" >> "$OUT/build.log"
            missing=1
        fi
    done

    # Receptor pdb
    if [ "$BENCH" = "dude" ]; then
        rec="$FLAT/${target}.receptor.pdb"
    else
        rec="$FLAT/${pdb_file}"
    fi
    if [ ! -f "$rec" ]; then
        echo "$target: missing receptor $(basename $rec)" >> "$OUT/build.log"
        missing=1
    fi

    if [ "$missing" = "1" ]; then
        n_missing=$((n_missing + 1))
        continue
    fi

    mkdir -p "$dst"
    cp "$grid" "$dst/${target}.grid.npz"
    cp "$prop" "$dst/${target}.prop.npz"
    cp "$keya" "$dst/${target}.keyatom.def.npz"
    cp "$rec"  "$dst/receptor.pdb"

    # Concat active + all numbered decoys into all_ligands.mol2 (streaming, no RAM).
    # Sort decoy files numerically to keep order stable.
    decoy_list=$(ls -v "$FLAT/${target}.decoy_"*.mol2 2>/dev/null || true)
    if [ -z "$decoy_list" ]; then
        n_no_decoys=$((n_no_decoys + 1))
        echo "$target: no decoy_*.mol2 files found (actives-only ship)" >> "$OUT/build.log"
        cat "$activ" > "$dst/all_ligands.mol2"
    else
        cat "$activ" $decoy_list > "$dst/all_ligands.mol2"
    fi

    # Labels: compound IDs from actives
    extract_ids "$activ" > "$OUT/labels/${target}.actives.txt"

    n_ok=$((n_ok + 1))
done < "$MANIFEST"

# README
if [ "$BENCH" = "dude" ]; then
    src_label="DUD-E (Mysinger et al. 2012, J Med Chem 55:6582, doi:10.1021/jm300687e)"
    src_url="http://dud.docking.org"
    lic="Freely available for academic and commercial use. Redistribution under same terms."
else
    src_label="LIT-PCBA (Tran-Nguyen, Jacquemard, Rognan 2020, JCIM 60:4263, doi:10.1021/acs.jcim.0c00155)"
    src_url="https://drugdesign.unistra.fr/LIT-PCBA/"
    lic="See original LIT-PCBA distribution terms."
fi

cat > "$OUT/README.md" <<EOF
# ${BENCH} benchmark, prepared for MotifScreen-Aff

$n_ok targets from $src_label.

These are the exact baked features used to generate the paper's benchmark
numbers: Rosetta score_jd2 relaxed receptors, training-era grid/prop/keyatom
npzs, and MMFF94-charged ligand mol2s parsed through the training-time
mol2 -> obabel PDB -> RDKit round-trip.

## Per target under \`prepared/<target>/\`

- \`receptor.pdb\`                    Rosetta-relaxed protein
- \`<target>.grid.npz\`               Baked receptor grid
- \`<target>.prop.npz\`               Baked receptor properties
- \`<target>.keyatom.def.npz\`        BRICS keyatoms (dict keyed by compound ID)
- \`all_ligands.mol2\`                Concat of actives + all decoys (single file)

## Labels

\`labels/<target>.actives.txt\` -- one compound ID per line. Used by the
metric scripts to compute AUROC / EF@1% against predict output.

## Usage

    uv run python motifscreen.py predict \\
        --datapath ${BENCH}_bench/prepared \\
        --checkpoint <path>/epoch70.pkl \\
        --base-config configs/training/endtoend.yaml \\
        --gpus 0,1 \\
        --output results/${BENCH}_scores.csv

For metrics see \`BENCHMARKS.md\` in the code repo.

## Original data source

$src_label
Source: $src_url

## License

$lic
EOF

echo ""
echo "Staged $n_ok / $((n_ok + n_missing)) $BENCH targets in $OUT"
echo "Size:  $(du -sh $OUT | cut -f1)"
if [ "$n_missing" -gt 0 ]; then
    echo "$n_missing targets skipped (see $OUT/build.log)"
fi
if [ "$n_no_decoys" -gt 0 ]; then
    echo "$n_no_decoys targets shipped actives-only (see $OUT/build.log)"
fi
