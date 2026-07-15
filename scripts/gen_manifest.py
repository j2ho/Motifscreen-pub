#!/usr/bin/env python
"""Auto-generate a manifest.tsv from a directory of targets.

Expected layout under --root:
    <target1>/
        receptor.pdb          (or *_protein.pdb / *_receptor.pdb)
        ligands.mol2          (or actives.mol2 + decoys.mol2, or *.sdf)
        crystal_ligand.mol2   (optional: used to compute center)

Writes manifest.tsv with columns: target_id, receptor_pdb, ligands, center_x, center_y, center_z

Center handling:
  --center-mode crystal   (default) compute COM from <target>/crystal_ligand.mol2
  --center-mode ligand    compute COM from the ligands file
  --center-mode manual    require a centers.tsv (target_id x y z) via --centers-file

Usage:
    python scripts/gen_manifest.py --root my_targets/ --output manifest.tsv
    python scripts/gen_manifest.py --root my_targets/ --output manifest.tsv \\
        --center-mode manual --centers-file centers.tsv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np


def find_receptor(target_dir: Path) -> str:
    for name in ('receptor.pdb', 'receptor_h.pdb'):
        p = target_dir / name
        if p.exists():
            return name
    # look for *_protein.pdb or *_receptor.pdb
    for pat in ('*_protein.pdb', '*_receptor.pdb', '*.pdb'):
        hits = sorted(target_dir.glob(pat))
        if hits:
            return hits[0].name
    return None


def find_ligands(target_dir: Path) -> str:
    # Prefer combined file if present, else concat actives + decoys is user's responsibility
    for name in ('ligands.mol2', 'all_ligands.mol2', 'compounds.mol2', 'ligands.sdf', 'compounds.sdf'):
        p = target_dir / name
        if p.exists():
            return name
    # First mol2 or sdf in dir (excluding crystal_ligand)
    for pat in ('*.mol2', '*.sdf'):
        hits = sorted(p.name for p in target_dir.glob(pat) if 'crystal' not in p.name.lower())
        if hits:
            return hits[0]
    return None


def read_ligand_com(mol2_or_pdb: Path):
    """Compute COM of heavy atoms in a mol2 or pdb file."""
    xs, ys, zs = [], [], []
    suffix = mol2_or_pdb.suffix.lower()
    with open(mol2_or_pdb) as f:
        if suffix == '.mol2':
            in_atoms = False
            for line in f:
                if line.startswith('@<TRIPOS>ATOM'):
                    in_atoms = True; continue
                if line.startswith('@<TRIPOS>'):
                    in_atoms = False; continue
                if in_atoms and line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                            atype = parts[5].split('.')[0] if len(parts) >= 6 else 'C'
                            if atype != 'H':
                                xs.append(x); ys.append(y); zs.append(z)
                        except ValueError:
                            pass
        else:  # PDB
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    try:
                        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                        elem = line[76:78].strip()
                        name = line[12:16].strip()
                        if elem and elem == 'H': continue
                        if not elem and name.startswith('H'): continue
                        xs.append(x); ys.append(y); zs.append(z)
                    except ValueError:
                        pass
    if not xs:
        return None
    return (float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='directory of per-target subdirs')
    ap.add_argument('--output', required=True, help='manifest.tsv output path')
    ap.add_argument('--center-mode', choices=['crystal', 'ligand', 'manual'],
                    default='crystal')
    ap.add_argument('--centers-file',
                    help='TSV file with target_id, x, y, z (for --center-mode manual)')
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(2)

    manual_centers = {}
    if args.center_mode == 'manual':
        if not args.centers_file:
            print("ERROR: --center-mode manual requires --centers-file", file=sys.stderr)
            sys.exit(2)
        with open(args.centers_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        manual_centers[parts[0]] = (
                            float(parts[1]), float(parts[2]), float(parts[3]))
                    except ValueError:
                        pass

    rows = []
    skipped = []
    for tdir in sorted(root.iterdir()):
        if not tdir.is_dir():
            continue
        target = tdir.name
        pdb = find_receptor(tdir)
        if not pdb:
            skipped.append((target, 'no receptor PDB'))
            continue
        ligands = find_ligands(tdir)
        if not ligands:
            skipped.append((target, 'no ligands file'))
            continue

        if args.center_mode == 'manual':
            center = manual_centers.get(target)
            if center is None:
                skipped.append((target, 'no manual center'))
                continue
        elif args.center_mode == 'ligand':
            center = read_ligand_com(tdir / ligands)
            if center is None:
                skipped.append((target, 'ligand COM failed'))
                continue
        else:  # crystal
            for crys in ('crystal_ligand.mol2', 'crystal.mol2', 'crystal_ligand.pdb'):
                p = tdir / crys
                if p.exists():
                    center = read_ligand_com(p)
                    if center is not None:
                        break
            else:
                skipped.append((target, 'no crystal_ligand.mol2 (try --center-mode ligand)'))
                continue

        rows.append((target, pdb, ligands, center))

    with open(args.output, 'w') as f:
        f.write("target_id\treceptor_pdb\tligands\tcenter_x\tcenter_y\tcenter_z\n")
        for target, pdb, ligands, (cx, cy, cz) in rows:
            f.write(f"{target}\t{pdb}\t{ligands}\t{cx:.3f}\t{cy:.3f}\t{cz:.3f}\n")

    print(f"wrote {args.output}: {len(rows)} targets")
    if skipped:
        print(f"skipped ({len(skipped)}):")
        for t, reason in skipped[:10]:
            print(f"  {t}: {reason}")
        if len(skipped) > 10:
            print(f"  ...and {len(skipped) - 10} more")


if __name__ == '__main__':
    main()
