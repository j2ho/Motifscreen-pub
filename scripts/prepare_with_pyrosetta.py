"""Drop-in replacement for `reduce -BUILD -Quiet` using PyRosetta.

Matches the exact protein-prep procedure the training pipeline used:
Rosetta `score_jd2 -no_optH false` (score + hydrogen optimization). This
adds hydrogens AND renames histidines to their protonation-state-specific
type names (HIE / HID / HIP) and marks disulfide cysteines as CYX -- the
same residue-name-encoded protonation the training-time features were
built from.

Compared to `reduce -BUILD`, this closes the reduce-vs-baked DUD-E delta
(~0.03 AUROC on non-GPCR targets) at the cost of ~15-30 s per target
instead of ~1-5 s, plus a ~1 GB PyRosetta wheel install.

Usage:
    python scripts/prepare_with_pyrosetta.py INPUT.pdb OUTPUT.pdb
    python scripts/prepare_with_pyrosetta.py --help

Install PyRosetta (only needed if you want to use this script):
    uv sync --extra rosetta

Falls back to a non-zero exit code with a clear message if PyRosetta
is not installed; the calling shell script can then fall back to reduce.
"""
import argparse
import os
import sys
import time


def prepare_receptor(input_pdb: str, output_pdb: str, quiet: bool = True) -> int:
    """Run Rosetta H-opt on input_pdb, write result to output_pdb.

    Returns 0 on success, non-zero on failure (matching CLI convention so
    callers can use standard shell exit-code branching).
    """
    try:
        import pyrosetta
    except ImportError:
        print(
            "ERROR: pyrosetta is not installed. Install with:\n"
            "  uv sync --extra rosetta\n"
            "Or fall back to reduce (much faster to install, ~0.03 AUROC lower "
            "on DUD-E non-GPCR targets).",
            file=sys.stderr,
        )
        return 2

    if not os.path.exists(input_pdb):
        print(f"ERROR: input PDB not found: {input_pdb}", file=sys.stderr)
        return 2

    # Rosetta init flags. -no_optH false matches the training-time
    # score_jd2 invocation. -mute all keeps the log quiet by default.
    init_flags = ["-no_optH false", "-ex1", "-ex2aro", "-ignore_zero_occupancy false"]
    if quiet:
        init_flags.append("-mute all")

    t0 = time.time()
    pyrosetta.init(options=" ".join(init_flags), silent=quiet)

    # Load the pose. Rosetta assigns per-residue types during this step
    # (HIS -> HIE / HID / HIP based on H-bond environment; CYS -> CYX
    # when a disulfide is detected). With -no_optH false, H rotamers are
    # optimized here.
    try:
        pose = pyrosetta.pose_from_pdb(input_pdb)
    except Exception as exc:
        print(f"ERROR: pyrosetta failed to load {input_pdb}: {exc}", file=sys.stderr)
        return 1

    if pose.size() == 0:
        print(f"ERROR: pyrosetta loaded 0 residues from {input_pdb}", file=sys.stderr)
        return 1

    out_dir = os.path.dirname(os.path.abspath(output_pdb))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pose.dump_pdb(output_pdb)

    if not (os.path.exists(output_pdb) and os.path.getsize(output_pdb) > 0):
        print(f"ERROR: output PDB is empty: {output_pdb}", file=sys.stderr)
        return 1

    elapsed = time.time() - t0
    if not quiet:
        print(
            f"pyrosetta prep: {pose.size()} residues -> {output_pdb} in {elapsed:.1f}s",
            file=sys.stderr,
        )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=("Rosetta score_jd2 -no_optH false via PyRosetta. "
                     "Drop-in for `reduce -BUILD -Quiet`."))
    parser.add_argument("input_pdb", help="Input protein PDB")
    parser.add_argument("output_pdb", help="Output PDB path")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print progress + timing info to stderr")
    args = parser.parse_args()

    rc = prepare_receptor(args.input_pdb, args.output_pdb, quiet=not args.verbose)
    sys.exit(rc)


if __name__ == "__main__":
    main()
