#!/bin/bash
# Build the combined MotifScreen-Aff benchmarks tarball for Zenodo upload.
#
# Combines ChEMBL-LR, DUD-E, and LIT-PCBA prepared bundles into a single
# tar.gz. Expects each sub-benchmark's staging dir to be built beforehand
# (via build_chembl_bench_tarball.sh / build_dude_bench_tarball.sh /
# build_litpcba_bench_tarball.sh).
#
# Usage:
#   # 1. Build each sub-bundle first (see per-script args):
#   bash scripts/build_chembl_bench_tarball.sh --source ... --out data/staging/chembl_bench.tar.gz
#   bash scripts/build_dude_bench_tarball.sh --prepared ... --raw ... --manifest ... --out data/staging/dude_bench
#   bash scripts/build_litpcba_bench_tarball.sh --prepared ... --raw ... --manifest ... --out data/staging/litpcba_bench
#
#   # 2. Combine:
#   bash scripts/build_all_benchmarks_tarball.sh \
#       --chembl data/staging/chembl_bench \
#       --dude data/staging/dude_bench \
#       --litpcba data/staging/litpcba_bench \
#       --out motifscreen_aff_benchmarks_v1.tar.gz

set -e

CHEMBL_STAGE=""
DUDE_STAGE=""
LITPCBA_STAGE=""
OUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --chembl) CHEMBL_STAGE="$2"; shift 2 ;;
        --dude) DUDE_STAGE="$2"; shift 2 ;;
        --litpcba) LITPCBA_STAGE="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 2 ;;
    esac
done

if [ -z "$OUT" ]; then
    echo "Usage: $0 --chembl DIR --dude DIR --litpcba DIR --out tarball.tar.gz"
    exit 2
fi

# Assemble a top-level staging dir
STAGING=$(mktemp -d)
mkdir -p "$STAGING/motifscreen_aff_benchmarks"

if [ -n "$CHEMBL_STAGE" ] && [ -d "$CHEMBL_STAGE" ]; then
    cp -r "$CHEMBL_STAGE" "$STAGING/motifscreen_aff_benchmarks/chembl_bench"
    echo "included chembl_bench: $(du -sh $CHEMBL_STAGE | cut -f1)"
fi
if [ -n "$DUDE_STAGE" ] && [ -d "$DUDE_STAGE" ]; then
    cp -r "$DUDE_STAGE" "$STAGING/motifscreen_aff_benchmarks/dude_bench"
    echo "included dude_bench: $(du -sh $DUDE_STAGE | cut -f1)"
fi
if [ -n "$LITPCBA_STAGE" ] && [ -d "$LITPCBA_STAGE" ]; then
    cp -r "$LITPCBA_STAGE" "$STAGING/motifscreen_aff_benchmarks/litpcba_bench"
    echo "included litpcba_bench: $(du -sh $LITPCBA_STAGE | cut -f1)"
fi

# Top-level README
cat > "$STAGING/motifscreen_aff_benchmarks/README.md" <<'EOF'
# MotifScreen-Aff benchmark bundle (v1)

Three virtual-screening benchmarks, prepared with the public MotifScreen-Aff
pipeline. Predict directly without running prepare from scratch.

## Contents

- `chembl_bench/` -- 107-target AVE-tercile-balanced ChEMBL-LR set (our contribution)
- `dude_bench/`   -- 96 targets from DUD-E (Mysinger et al. 2012)
- `litpcba_bench/` -- 13 targets from LIT-PCBA (Tran-Nguyen et al. 2020)

## Quick start

```bash
tar xzf motifscreen_aff_benchmarks_v1.tar.gz
uv run python motifscreen.py predict \
    --datapath motifscreen_aff_benchmarks/dude_bench/prepared \
    --checkpoint <model>.pkl \
    --base-config configs/training/endtoend.yaml \
    --gpus 0,1 \
    --output results/dude_scores.csv
```

Then compute metrics per benchmark. See `BENCHMARKS.md` in the code repo
for full instructions and expected numbers.

## Data provenance and licensing

- **ChEMBL-LR-107** -- assembled from ChEMBL v34 by the MotifScreen-Aff
  authors. CC-BY 4.0. Cite the MotifScreen-Aff paper.
- **DUD-E** -- Mysinger, Carchia, Irwin, Shoichet (2012). J Med Chem 55:6582.
  http://dud.docking.org. Freely available for academic + commercial use.
- **LIT-PCBA** -- Tran-Nguyen, Jacquemard, Rognan (2020). J Chem Inf Model
  60:4263. https://drugdesign.unistra.fr/LIT-PCBA/. Refer to Unistra's
  distribution terms.

Each sub-benchmark's `README.md` has its own provenance notes and citation.

## What's in each `prepared/<target>/`

- `receptor.pdb`                Rosetta-relaxed protein (protonated)
- `<target>.grid.npz`           Baked receptor pocket grid
- `<target>.prop.npz`           Baked receptor per-atom properties
- `all_ligands.mol2`            Canonicalized + MMFF94-charged mol2, ready for predict
- `all_ligands.keyatom.def.npz` BRICS keyatoms per compound
- `actives*.mol2`               Ground-truth actives for metrics computation

Skip 3-10 hours of prep time by using this bundle directly with `predict`.
EOF

echo ""
echo "Total staged size: $(du -sh $STAGING/motifscreen_aff_benchmarks | cut -f1)"
echo "Creating tarball: $OUT"

tar czf "$OUT" -C "$STAGING" motifscreen_aff_benchmarks

# sha256
sha256sum "$OUT" | tee "$OUT.sha256"

# Guardrails prohibit rm -rf so leave staging dir in place
echo ""
echo "Tarball ready: $OUT ($(du -sh $OUT | cut -f1))"
echo "Staging preserved at: $STAGING (remove manually if not needed)"
echo ""
echo "Next: upload $OUT and $OUT.sha256 to Zenodo as motifscreen-aff-benchmarks-v1"
