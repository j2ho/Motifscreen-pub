#!/bin/bash
# Download raw LIT-PCBA data (~40 GB) from Unistra's LIT-PCBA distribution.
# One-shot per user; cache locally.
#
# Usage:
#   bash scripts/download_litpcba_raw.sh --output data/litpcba_raw/

set -e

OUTPUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --output) OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 2 ;;
    esac
done

if [ -z "$OUTPUT" ]; then
    echo "Usage: $0 --output data/litpcba_raw/"
    exit 2
fi

mkdir -p "$OUTPUT"

MANIFEST=data/litpcba_bench_manifest.tsv
[ ! -f "$MANIFEST" ] && { echo "ERROR: $MANIFEST not found"; exit 2; }

BASE_URL="https://drugdesign.unistra.fr/LIT-PCBA/Data"

TARGETS=$(tail -n +2 "$MANIFEST" | cut -f1)

for t in $TARGETS; do
    tdir="$OUTPUT/$t"
    mkdir -p "$tdir"
    if [ -f "$tdir/actives.smi" ] && [ -f "$tdir/inactives.smi" ]; then
        echo "$t: cached"
        continue
    fi
    echo "downloading $t ..."
    # LIT-PCBA distributes per-target zip archives
    curl -sf -L "$BASE_URL/$t.zip" -o "$OUTPUT/$t.zip" && \
        unzip -q -o "$OUTPUT/$t.zip" -d "$OUTPUT/" && \
        rm -f "$OUTPUT/$t.zip"
done

echo ""
echo "downloaded to $OUTPUT"
echo "NOTE: LIT-PCBA distributes SMILES + PDB. See scripts/run_litpcba_bench.sh for how"
echo "we convert to the mol2 format used by prepare."
