#!/bin/bash
# Download raw DUD-E data (~15 GB) from dud.docking.org.
# One-shot: users run this once, then re-run bench scripts against the cache.
#
# Usage:
#   bash scripts/download_dude_raw.sh --output data/dude_raw/
#
# By default downloads all 102 DUD-E targets. Use --targets to pick a subset.

set -e

OUTPUT=""
TARGETS_FILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --output) OUTPUT="$2"; shift 2 ;;
        --targets-file) TARGETS_FILE="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 2 ;;
    esac
done

if [ -z "$OUTPUT" ]; then
    echo "Usage: $0 --output data/dude_raw/ [--targets-file targets.list]"
    exit 2
fi

mkdir -p "$OUTPUT"

# Default target list: everything in our manifest
if [ -z "$TARGETS_FILE" ]; then
    MANIFEST=data/dude_bench_manifest.tsv
    if [ ! -f "$MANIFEST" ]; then
        echo "ERROR: manifest not found at $MANIFEST"
        exit 2
    fi
    TARGETS=$(tail -n +2 "$MANIFEST" | cut -f1)
else
    TARGETS=$(cat "$TARGETS_FILE")
fi

BASE_URL="http://dud.docking.org/targets"

for t in $TARGETS; do
    tdir="$OUTPUT/$t"
    if [ -f "$tdir/receptor.pdb" ] && \
       [ -f "$tdir/actives_final.mol2.gz" ] && \
       [ -f "$tdir/decoys_final.mol2.gz" ] && \
       [ -f "$tdir/crystal_ligand.mol2" ]; then
        echo "$t: cached, skip"
        continue
    fi
    mkdir -p "$tdir"
    echo "downloading $t ..."
    for f in receptor.pdb actives_final.mol2.gz decoys_final.mol2.gz crystal_ligand.mol2; do
        if [ ! -f "$tdir/$f" ]; then
            curl -sf -L "$BASE_URL/$t/$f" -o "$tdir/$f" || echo "  WARN: could not fetch $t/$f"
        fi
    done
done

n_ok=0
for t in $TARGETS; do
    [ -f "$OUTPUT/$t/receptor.pdb" ] && n_ok=$((n_ok+1))
done
echo ""
echo "downloaded $n_ok targets to $OUTPUT"
