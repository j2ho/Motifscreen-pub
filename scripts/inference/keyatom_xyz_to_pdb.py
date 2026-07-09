#!/usr/bin/env python3
"""
Convert keyatom_xyz.npz files to PDB format.

The NPZ file contains predicted and original key atom coordinates from inference.
Each compound's key atoms are written as a separate MODEL in the PDB.

Usage:
    python keyatom_xyz_to_pdb.py <input.npz> <output.pdb> [--pred|--orig|--both]

Examples:
    # Write predicted coordinates (default)
    python keyatom_xyz_to_pdb.py 3jq7_keyatom_xyz.npz 3jq7_keyatoms_pred.pdb

    # Write original coordinates
    python keyatom_xyz_to_pdb.py 3jq7_keyatom_xyz.npz 3jq7_keyatoms_orig.pdb --orig

    # Write both as separate models (pred first, then orig)
    python keyatom_xyz_to_pdb.py 3jq7_keyatom_xyz.npz 3jq7_keyatoms.pdb --both
"""

import numpy as np
import sys
import os
import argparse
from datetime import datetime


def get_element_from_atom_name(atom_name: str) -> str:
    """Extract element symbol from atom name (e.g., 'C4' -> 'C', 'N1' -> 'N')."""
    # Remove digits and get first character(s)
    element = ''.join(c for c in atom_name if not c.isdigit())
    if len(element) == 0:
        return "X"
    # Common two-letter elements
    if element.upper() in ['CL', 'BR', 'FE', 'ZN', 'MG', 'CA', 'NA']:
        return element.upper()[:2]
    return element[0].upper()


def write_pdb_header(f, title: str = "KEYATOM COORDINATES"):
    """Write PDB header."""
    current_date = datetime.now().strftime("%d-%b-%y").upper()
    f.write(f"HEADER    {title:<40}{current_date}   KEYATOM\n")
    f.write("EXPDTA    THEORETICAL MODEL\n")
    f.write("REMARK 220\n")
    f.write("REMARK 220 EXPERIMENTAL DETAILS\n")
    f.write("REMARK 220  EXPERIMENT TYPE                : MOTIFSCREEN-AFF INFERENCE\n")
    f.write(f"REMARK 220  DATE OF DATA COLLECTION        : {current_date}\n")
    f.write("REMARK 220\n")
    f.write("REMARK 220 REMARK: KEY ATOM COORDINATES FROM INFERENCE\n")
    f.write("REMARK 220 REMARK: EACH MODEL = ONE COMPOUND\n")
    f.write("REMARK 220\n")


def write_compound_atoms(f, compound_id: str, atom_names: list, xyz: np.ndarray,
                         model_num: int, coord_type: str = "PRED", score: float = None):
    """
    Write atoms for one compound as a MODEL.

    Args:
        f: File handle
        compound_id: Compound identifier
        atom_names: List of atom names
        xyz: Coordinates array, shape [K, 3] or [K, 1, 3]
        model_num: MODEL number
        coord_type: "PRED" or "ORIG" for remarks
        score: Optional binding score
    """
    # Handle different xyz shapes
    if xyz.ndim == 3:
        xyz = xyz.squeeze(1)  # [K, 1, 3] -> [K, 3]

    f.write(f"MODEL     {model_num:4d}\n")
    f.write(f"REMARK 250 COMPOUND: {compound_id}\n")
    f.write(f"REMARK 250 COORD_TYPE: {coord_type}\n")
    if score is not None:
        f.write(f"REMARK 250 SCORE: {score:.4f}\n")

    for i, (atom_name, coords) in enumerate(zip(atom_names, xyz)):
        atom_num = i + 1
        x, y, z = coords
        element = get_element_from_atom_name(atom_name)

        # PDB HETATM format
        # HETATM    1  C4  LIG A   1       0.000   0.000   0.000  1.00  0.00           C
        f.write(
            f"HETATM{atom_num:5d}  {atom_name:<3s} LIG A{model_num:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
        )

    f.write("ENDMDL\n")


def keyatom_xyz_to_pdb(input_file: str, output_file: str, coord_type: str = "pred"):
    """
    Convert keyatom xyz NPZ to PDB format.

    Args:
        input_file: Path to keyatom_xyz.npz file
        output_file: Path to output PDB file
        coord_type: "pred", "orig", or "both"
    """
    # Load the NPZ file
    try:
        data = np.load(input_file, allow_pickle=True)
        keyatom_xyz = data['keyatom_xyz'].item()
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return False
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return False

    if not keyatom_xyz:
        print("Error: No keyatom data found in file.")
        return False

    try:
        with open(output_file, 'w') as f:
            write_pdb_header(f, f"KEYATOM {coord_type.upper()} COORDINATES")

            model_num = 0
            compounds_written = 0

            for compound_id, cdata in keyatom_xyz.items():
                atom_names = cdata['atom_names']
                score = cdata.get('score')

                if coord_type in ["pred", "both"]:
                    if 'pred_xyz' in cdata and cdata['pred_xyz'] is not None:
                        model_num += 1
                        write_compound_atoms(
                            f, compound_id, atom_names, cdata['pred_xyz'],
                            model_num, "PRED", score
                        )
                        compounds_written += 1

                if coord_type in ["orig", "both"]:
                    if 'orig_xyz' in cdata and cdata['orig_xyz'] is not None:
                        model_num += 1
                        write_compound_atoms(
                            f, compound_id, atom_names, cdata['orig_xyz'],
                            model_num, "ORIG", score
                        )
                        if coord_type == "orig":
                            compounds_written += 1

            f.write("END\n")

        print(f"Successfully wrote {compounds_written} compounds ({model_num} models) to {output_file}")
        return True

    except Exception as e:
        print(f"Error writing to {output_file}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert keyatom_xyz.npz to PDB format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Write predicted coordinates (default)
    python keyatom_xyz_to_pdb.py results/3jq7_keyatom_xyz.npz output_pred.pdb

    # Write original coordinates
    python keyatom_xyz_to_pdb.py results/3jq7_keyatom_xyz.npz output_orig.pdb --orig

    # Write both predicted and original as separate models
    python keyatom_xyz_to_pdb.py results/3jq7_keyatom_xyz.npz output_both.pdb --both
        """
    )

    parser.add_argument('input', help='Input keyatom_xyz.npz file')
    parser.add_argument('output', help='Output PDB file')

    coord_group = parser.add_mutually_exclusive_group()
    coord_group.add_argument('--pred', action='store_true', default=True,
                             help='Write predicted coordinates (default)')
    coord_group.add_argument('--orig', action='store_true',
                             help='Write original coordinates')
    coord_group.add_argument('--both', action='store_true',
                             help='Write both pred and orig as separate models')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)

    # Determine coordinate type
    if args.both:
        coord_type = "both"
    elif args.orig:
        coord_type = "orig"
    else:
        coord_type = "pred"

    success = keyatom_xyz_to_pdb(args.input, args.output, coord_type)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
