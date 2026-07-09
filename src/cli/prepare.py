#!/usr/bin/env python
"""Prepare input files for MotifScreen-Aff inference.

Takes a protein PDB and ligand file (SDF or mol2), produces all npz files
needed by the predict step.

The protein PDB should already have hydrogens added. If not, use
--protonate-rosetta to add them via Rosetta score_jd2.

Usage:
    # Protein already has H (or user is OK without):
    uv run python motifscreen.py prepare \
        --protein receptor.pdb \
        --ligands compounds.sdf \
        --center 12.5,34.2,8.7 \
        --output prepared/

    # Add H via Rosetta first:
    uv run python motifscreen.py prepare \
        --protein receptor.pdb \
        --ligands compounds.sdf \
        --center 12.5,34.2,8.7 \
        --output prepared/ \
        --protonate-rosetta /path/to/rosetta/source/bin/score_jd2.linuxgccrelease

Output:
    {output}/{target_id}/
        {target_id}.grid.npz
        {target_id}.prop.npz
        {ligand_stem}.mol2
        {ligand_stem}.keyatom.def.npz
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.io.protein_featurizer import main as featurize_protein
from src.io.protein_featurizer import calculate_ligand_com
from src.io.ligand_processer import launch_batched_ligand

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def check_obabel():
    try:
        result = subprocess.run(
            ['obabel', '-V'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    logger.error("obabel not found. Install: uv pip install openbabel-wheel")
    return False


def protonate_rosetta(input_pdb: str, output_pdb: str, score_jd2_bin: str) -> bool:
    """Add hydrogens to protein using Rosetta score_jd2."""
    tmpdir = tempfile.mkdtemp(prefix='msk_rosetta_')
    try:
        result = subprocess.run(
            [score_jd2_bin, '-s', input_pdb, '-no_optH', 'false', '-out:pdb'],
            capture_output=True, text=True, cwd=tmpdir)
        # Rosetta outputs {stem}_0001.pdb
        stem = Path(input_pdb).stem
        rosetta_out = os.path.join(tmpdir, f'{stem}_0001.pdb')
        if result.returncode != 0 or not _file_ok(rosetta_out):
            logger.error(f"Rosetta protonation failed (rc={result.returncode})")
            if result.stderr:
                logger.error(result.stderr[-500:])
            return False
        shutil.copy2(rosetta_out, output_pdb)
        logger.info(f"Rosetta protonation -> {output_pdb}")
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def prepare_ligand_mol2(input_file: str, output_mol2: str) -> bool:
    """Convert SDF/mol2 to prepared mol2 with hydrogens and charges.

    Pipeline: input -> mol2 -> add H (pH 7) -> add MMFF94 charges
    """
    tmpdir = tempfile.mkdtemp(prefix='msk_ligprep_')
    try:
        suffix = Path(input_file).suffix.lower()

        # Step 1: ensure mol2 format
        if suffix == '.sdf':
            raw_mol2 = os.path.join(tmpdir, 'raw.mol2')
            result = subprocess.run(
                ['obabel', '-isdf', input_file, '-omol2', '-O', raw_mol2],
                capture_output=True, text=True)
            if not _file_ok(raw_mol2):
                logger.error(f"SDF -> mol2 conversion failed: {result.stderr}")
                return False
        elif suffix == '.mol2':
            raw_mol2 = input_file
        else:
            logger.error(f"Unsupported ligand format: {suffix}. Use .sdf or .mol2")
            return False

        # Step 2: add hydrogens at pH 7
        h_mol2 = os.path.join(tmpdir, 'h.mol2')
        result = subprocess.run(
            ['obabel', '-imol2', raw_mol2, '-omol2', '-O', h_mol2, '-p', '7'],
            capture_output=True, text=True)
        if not _file_ok(h_mol2):
            logger.error(f"H-addition failed: {result.stderr}")
            return False

        # Step 3: add MMFF94 charges
        result = subprocess.run(
            ['obabel', '-imol2', h_mol2, '-omol2', '-O', output_mol2,
             '--partialcharge', 'mmff94'],
            capture_output=True, text=True)
        if not _file_ok(output_mol2):
            logger.error(f"Charge assignment failed: {result.stderr}")
            return False

        logger.info(f"Prepared ligand mol2 -> {output_mol2}")
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _file_ok(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def count_mol2_compounds(mol2_path: str) -> int:
    count = 0
    with open(mol2_path) as f:
        for line in f:
            if line.startswith('@<TRIPOS>MOLECULE'):
                count += 1
    return count


def prepare(protein_pdb: str, ligands_file: str,
            center: np.ndarray,
            output_dir: str,
            target_id: str = None,
            protonate_rosetta_bin: str = None,
            skip_ligand_prep: bool = False,
            workers: int = 4,
            gridsize: float = 1.5,
            padding: float = 10.0,
            clash: float = 1.1):
    """Run full preparation pipeline."""

    if target_id is None:
        target_id = Path(protein_pdb).stem

    target_dir = Path(output_dir) / target_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Protein protonation (optional)
    if protonate_rosetta_bin:
        prot_pdb = str(target_dir / f"{target_id}_H.pdb")
        if not protonate_rosetta(protein_pdb, prot_pdb, protonate_rosetta_bin):
            sys.exit(1)
    else:
        prot_pdb = protein_pdb
        logger.info(f"Using protein PDB as-is: {prot_pdb}")

    # 2. Featurize protein -> grid.npz + prop.npz
    outprefix = str(target_dir / target_id)
    logger.info("Featurizing protein...")
    featurize_protein(
        pdb=prot_pdb,
        outprefix=outprefix,
        gridsize=gridsize,
        com=center,
        padding=padding,
        clash=clash,
        gridoption='com',
    )

    grid_npz = f"{outprefix}.grid.npz"
    prop_npz = f"{outprefix}.prop.npz"
    if not _file_ok(grid_npz) or not _file_ok(prop_npz):
        logger.error("Protein featurization failed (no output)")
        sys.exit(1)
    logger.info(f"  -> {grid_npz}")
    logger.info(f"  -> {prop_npz}")

    # 3. Prepare ligand mol2
    ligand_stem = Path(ligands_file).stem
    prepared_mol2 = str(target_dir / f"{ligand_stem}.mol2")

    if skip_ligand_prep:
        if Path(ligands_file).suffix.lower() == '.mol2':
            shutil.copy2(ligands_file, prepared_mol2)
        else:
            logger.error("--skip-ligand-prep requires mol2 input")
            sys.exit(1)
        logger.info(f"Copied ligand mol2 as-is -> {prepared_mol2}")
    else:
        if not prepare_ligand_mol2(ligands_file, prepared_mol2):
            sys.exit(1)

    n_compounds = count_mol2_compounds(prepared_mol2)
    logger.info(f"  {n_compounds} compounds in {prepared_mol2}")

    # 4. Compute key atoms -> keyatom.def.npz
    keyatom_npz = str(target_dir / f"{ligand_stem}.keyatom.def.npz")
    logger.info(f"Computing key atoms (BRICS decomposition, {workers} workers)...")
    launch_batched_ligand(
        prepared_mol2,
        N=workers,
        collated_npz=keyatom_npz,
    )

    if not _file_ok(keyatom_npz):
        logger.error("Key atom computation failed (no output)")
        sys.exit(1)

    # Verify keyatom coverage
    data = np.load(keyatom_npz, allow_pickle=True)
    keyatms = data['keyatms'].item()
    n_keyatoms = len(keyatms)
    logger.info(f"  -> {keyatom_npz} ({n_keyatoms}/{n_compounds} compounds with key atoms)")

    if n_keyatoms == 0:
        logger.error("No key atoms computed. Check ligand file format.")
        sys.exit(1)

    # Summary
    logger.info("")
    logger.info("Preparation complete.")
    logger.info(f"  Target:     {target_id}")
    logger.info(f"  Output:     {target_dir}")
    logger.info(f"  Compounds:  {n_compounds} ({n_keyatoms} with key atoms)")


def parse_center(center_str: str) -> np.ndarray:
    parts = center_str.split(',')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Center must be x,y,z (3 comma-separated floats), got: {center_str}")
    return np.array([float(x) for x in parts])


def main():
    parser = argparse.ArgumentParser(
        description='Prepare inputs for MotifScreen-Aff inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--protein', required=True,
                        help='Protein PDB file (ideally with hydrogens added)')
    parser.add_argument('--ligands', required=True,
                        help='Ligand file (SDF or mol2, can contain multiple compounds)')
    parser.add_argument('--output', required=True,
                        help='Output directory')

    center_group = parser.add_mutually_exclusive_group(required=True)
    center_group.add_argument('--center', type=parse_center,
                              help='Binding site center as x,y,z')
    center_group.add_argument('--crystal-ligand',
                              help='Crystal ligand file (PDB or mol2) to compute center from')

    parser.add_argument('--target-id',
                        help='Target identifier (default: protein filename stem)')
    parser.add_argument('--protonate-rosetta', metavar='SCORE_JD2',
                        help='Path to Rosetta score_jd2 binary to add hydrogens')
    parser.add_argument('--skip-ligand-prep', action='store_true',
                        help='Skip ligand preparation (mol2 input assumed ready)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Workers for key atom computation (default: 4)')
    parser.add_argument('--gridsize', type=float, default=1.5,
                        help='Grid spacing in Angstroms (default: 1.5)')
    parser.add_argument('--padding', type=float, default=10.0,
                        help='Grid padding around binding site (default: 10.0)')
    parser.add_argument('--clash', type=float, default=1.1,
                        help='Clash distance for grid filtering (default: 1.1)')

    args = parser.parse_args()

    if not check_obabel():
        sys.exit(1)

    if not os.path.exists(args.protein):
        logger.error(f"Protein file not found: {args.protein}")
        sys.exit(1)
    if not os.path.exists(args.ligands):
        logger.error(f"Ligand file not found: {args.ligands}")
        sys.exit(1)
    if args.protonate_rosetta and not os.path.exists(args.protonate_rosetta):
        logger.error(f"Rosetta binary not found: {args.protonate_rosetta}")
        sys.exit(1)

    # Resolve center
    if args.center is not None:
        center = args.center
    else:
        center = calculate_ligand_com(args.crystal_ligand)
        logger.info(f"Center from crystal ligand: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

    prepare(
        protein_pdb=args.protein,
        ligands_file=args.ligands,
        center=center,
        output_dir=args.output,
        target_id=args.target_id,
        protonate_rosetta_bin=args.protonate_rosetta,
        skip_ligand_prep=args.skip_ligand_prep,
        workers=args.workers,
        gridsize=args.gridsize,
        padding=args.padding,
        clash=args.clash,
    )


if __name__ == '__main__':
    main()
