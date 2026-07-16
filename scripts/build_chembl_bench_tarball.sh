#!/bin/bash
# Build the ChEMBL-LR benchmark tarball for Zenodo upload.
#
# Reads data/chembl_bench_manifest.tsv, pulls raw inputs + baked npz + labels
# from the training data mirror, packages into a single tar.gz for public
# distribution, and emits a sha256 for the release notes.
#
# Usage:
#   bash scripts/build_chembl_bench_tarball.sh \
#       --input-dir /home/j2ho/DB/motifscreen_a/chembl \
#       --out chembl_bench_v1.tar.gz
#
# Expected structure of the source dir (per target):
#   <SOURCE>/<target>/
#     <pdb_name>.pdb                    (Rosetta-relaxed, already protonated)
#     <target>.grid.npz                 (baked at training time)
#     <target>.prop.npz                 (baked at training time)
#     active_smiles_clu.csv             (ground-truth actives)
#     batch_mol2s/*.mol2                (compound batches)
#
# The tarball layout matches what scripts/download_and_prepare_chembl_bench.sh expects:
#   chembl_bench_raw/<target>/
#     receptor.pdb                      (renamed from <pdb_name>.pdb)
#     <target>.grid.npz
#     <target>.prop.npz
#     active_smiles_clu.csv
#     batch_mol2s/*.mol2
#
# Output:
#   <OUT>                                the tarball
#   <OUT>.sha256                         checksum for release notes / verification

set -e

SOURCE=""
OUT=""
MANIFEST=data/chembl_bench_manifest.tsv
STAGING=""

while [ $# -gt 0 ]; do
    case "$1" in
        --input-dir) SOURCE="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --staging) STAGING="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 2 ;;
    esac
done

if [ -z "$SOURCE" ] || [ -z "$OUT" ]; then
    echo "Usage: $0 --input-dir <chembl_dir> --out <chembl_bench_v1.tar.gz>"
    exit 2
fi

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found at $MANIFEST"
    exit 2
fi

if [ -z "$STAGING" ]; then
    STAGING=$(mktemp -d)
    CLEANUP_STAGING=1
else
    mkdir -p "$STAGING"
    CLEANUP_STAGING=0
fi

TOP=$STAGING/chembl_bench_raw
mkdir -p "$TOP"

echo "Staging under: $TOP"
echo "Reading manifest: $MANIFEST"
echo ""

ok=0
missing=0
: > "$STAGING/build.log"

while IFS=$'\t' read -r target tercile pdb ligands cx cy cz n_mol2s; do
    [ "$target" = "target_id" ] && continue

    src=$SOURCE/$target
    dst=$TOP/$target
    mkdir -p "$dst/batch_mol2s"

    # Receptor PDB (rename to canonical receptor.pdb)
    if [ ! -f "$src/$pdb" ]; then
        echo "$target: missing $pdb" >> "$STAGING/build.log"
        missing=$((missing+1))
        continue
    fi
    cp "$src/$pdb" "$dst/receptor.pdb"

    # Baked npz files (required for reproduce mode)
    for f in "$target.grid.npz" "$target.prop.npz"; do
        if [ ! -f "$src/$f" ]; then
            echo "$target: missing $f" >> "$STAGING/build.log"
            missing=$((missing+1))
            continue 2
        fi
        cp "$src/$f" "$dst/$f"
    done

    # Active labels
    if [ -f "$src/active_smiles_clu.csv" ]; then
        cp "$src/active_smiles_clu.csv" "$dst/active_smiles_clu.csv"
    else
        echo "$target: missing active_smiles_clu.csv" >> "$STAGING/build.log"
    fi

    # Compound batches (only the manifested ones - keeps tarball small)
    IFS=',' read -ra M2S <<< "$ligands"
    for m in "${M2S[@]}"; do
        if [ -f "$src/batch_mol2s/$m" ]; then
            cp "$src/batch_mol2s/$m" "$dst/batch_mol2s/$m"
        fi
    done

    ok=$((ok+1))
done < "$MANIFEST"

echo "Staged: ok=$ok, missing=$missing"
if [ "$missing" -gt 0 ]; then
    echo "See $STAGING/build.log for missing files"
fi

echo ""
echo "Directory tree summary:"
echo "  target dirs: $(find $TOP -maxdepth 1 -mindepth 1 -type d | wc -l)"
echo "  total files: $(find $TOP -type f | wc -l)"
echo "  size:        $(du -sh $TOP | cut -f1)"
echo ""

echo "Building tarball: $OUT"
tar czf "$OUT" -C "$STAGING" chembl_bench_raw
sha256sum "$OUT" | tee "$OUT.sha256"
echo ""
echo "Tarball size: $(du -sh $OUT | cut -f1)"

if [ "$CLEANUP_STAGING" = "1" ]; then
    echo "Cleaning staging: $STAGING"
    # Guardrails prohibit rm -rf; leave staging in place for manual cleanup
    echo "  (staging dir left in place - remove manually if not needed)"
fi

echo ""
echo "Next steps:"
echo "  1. Verify: tar tzf $OUT | head"
echo "  2. Upload $OUT to Zenodo"
echo "  3. Update ZENODO_URL and SHA256 in scripts/download_and_prepare_chembl_bench.sh"
echo "  4. Add DOI to README.md and manuscript"
