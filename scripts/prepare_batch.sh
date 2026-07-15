#!/bin/bash
# Batch prepare across targets, parallelized.
#
# Given a manifest.tsv describing multiple targets, runs `motifscreen prepare`
# on each. Multiple targets processed concurrently via xargs -P; each target's
# obabel step is already chunked internally (workers=8 by default).
#
# Manifest.tsv columns (tab-separated, header required):
#   target_id  receptor_pdb  ligands  center_x  center_y  center_z
#
# receptor_pdb, ligands: paths relative to --raw-root (or absolute).
#
# Usage:
#   scripts/prepare_batch.sh \
#       --manifest data/chembl_bench_manifest.tsv \
#       --raw-root data/chembl_bench_raw \
#       --output prepared/ \
#       --parallel 8
#
# Options:
#   --manifest PATH       tab-separated file (required)
#   --raw-root DIR        base dir for relative receptor_pdb + ligands paths
#                         (default: same dir as manifest)
#   --output DIR          output prepared/ dir (required)
#   --parallel N          concurrent targets (default: 4)
#   --workers-per-target N chunks within a target's obabel (default: 8)
#   --extra-args "..."    extra flags passed to motifscreen prepare
#                         (e.g. "--skip-ligand-prep")

set -e

MANIFEST=""
RAW_ROOT=""
OUTPUT=""
PARALLEL=4
WORKERS_PER_TARGET=8
EXTRA_ARGS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --raw-root) RAW_ROOT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --workers-per-target) WORKERS_PER_TARGET="$2"; shift 2 ;;
        --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 2 ;;
    esac
done

if [ -z "$MANIFEST" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 --manifest tsv --output prepared_dir [--parallel N]"
    exit 2
fi

if [ -z "$RAW_ROOT" ]; then
    RAW_ROOT=$(dirname "$MANIFEST")
fi

mkdir -p "$OUTPUT"

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Emit prep commands, one per target, then feed to xargs -P for parallel exec.
tmp_cmds=$(mktemp)
trap "rm -f $tmp_cmds" EXIT

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r target pdb ligands cx cy cz rest; do
    [ -z "$target" ] && continue
    # Resolve receptor + ligands paths (absolute or relative to RAW_ROOT/<target>/)
    if [[ "$pdb" != /* ]]; then
        pdb_full="$RAW_ROOT/$target/$pdb"
    else
        pdb_full="$pdb"
    fi
    if [[ "$ligands" != /* ]]; then
        lig_full="$RAW_ROOT/$target/$ligands"
    else
        lig_full="$ligands"
    fi
    # Fall back if per-target subdir doesn't hold the files
    [ ! -f "$pdb_full" ] && pdb_full="$RAW_ROOT/$pdb"
    [ ! -f "$lig_full" ] && lig_full="$RAW_ROOT/$ligands"

    echo "cd $REPO_ROOT && PYTHONPATH=. python motifscreen.py prepare \
        --protein $pdb_full \
        --ligands $lig_full \
        --center=$cx,$cy,$cz \
        --target-id $target \
        --output $OUTPUT \
        --workers $WORKERS_PER_TARGET \
        $EXTRA_ARGS > $OUTPUT/prep.$target.log 2>&1 && echo '  ok: $target' || echo '  FAIL: $target'" >> "$tmp_cmds"
done

n_targets=$(wc -l < "$tmp_cmds")
echo "batch prepare: $n_targets targets, parallel=$PARALLEL, workers-per-target=$WORKERS_PER_TARGET"
echo ""

t0=$SECONDS
cat "$tmp_cmds" | xargs -I {} -P "$PARALLEL" bash -c "{}"
echo ""
echo "batch wallclock: $((SECONDS - t0))s"
n_ok=$(ls "$OUTPUT" 2>/dev/null | grep -v '\.log$' | wc -l)
echo "prepared: $n_ok / $n_targets"
