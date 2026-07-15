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
from src.io.ligand_processer import launch_batched_ligand, build_ligand_graphs_batch

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


def _split_mol2_by_molecule(input_mol2: str, n_chunks: int, chunk_dir: str) -> list:
    """Split a multi-mol mol2 file into ~n_chunks files at MOLECULE boundaries.

    Returns list of chunk file paths in order. Empty chunks are omitted.
    """
    # Read all lines once, find MOLECULE line indices
    with open(input_mol2) as f:
        lines = f.readlines()
    mol_starts = [i for i, l in enumerate(lines) if l.startswith('@<TRIPOS>MOLECULE')]
    if not mol_starts:
        return []
    n_mol = len(mol_starts)
    if n_chunks > n_mol:
        n_chunks = n_mol

    chunk_paths = []
    per_chunk = (n_mol + n_chunks - 1) // n_chunks
    for chunk_idx in range(n_chunks):
        start_mol = chunk_idx * per_chunk
        end_mol = min((chunk_idx + 1) * per_chunk, n_mol)
        if start_mol >= n_mol:
            break
        start_line = mol_starts[start_mol]
        end_line = mol_starts[end_mol] if end_mol < n_mol else len(lines)
        path = os.path.join(chunk_dir, f'chunk_{chunk_idx:03d}.mol2')
        with open(path, 'w') as f:
            f.writelines(lines[start_line:end_line])
        chunk_paths.append(path)
    return chunk_paths


def _obabel_parallel(input_mol2: str, output_mol2: str, extra_args: list,
                     n_chunks: int = 8) -> bool:
    """Run obabel with `extra_args` in parallel over N chunks of the input mol2.

    Splits by molecule boundary, spawns N concurrent obabel subprocesses,
    concatenates results in original order.

    A single-target obabel MMFF94 step on ~30k compounds drops from ~40s to
    ~5-8s with n_chunks=8. Same numerical result (each chunk is independent).
    """
    tmpdir = tempfile.mkdtemp(prefix='msk_obabel_par_')
    try:
        in_chunks = _split_mol2_by_molecule(input_mol2, n_chunks, tmpdir)
        if not in_chunks:
            logger.error(f"No molecules found in {input_mol2}")
            return False
        out_chunks = [p + '.out' for p in in_chunks]
        procs = []
        for cin, cout in zip(in_chunks, out_chunks):
            cmd = ['obabel', '-imol2', cin, '-omol2', '-O', cout] + list(extra_args)
            procs.append(subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
        errs = []
        for p in procs:
            _, err = p.communicate()
            if p.returncode != 0:
                errs.append(err.decode('utf-8', errors='ignore')[-200:])
        if errs:
            logger.error(f"obabel chunk errors: {errs[:2]}")
            return False
        # Concatenate in order
        with open(output_mol2, 'w') as out:
            for cout in out_chunks:
                if not _file_ok(cout):
                    logger.error(f"obabel produced empty chunk: {cout}")
                    return False
                with open(cout) as f:
                    out.write(f.read())
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def prepare_ligand_mol2(input_file: str, output_mol2: str, workers: int = 8) -> bool:
    """Convert SDF/mol2 to prepared mol2 with hydrogens and charges.

    Pipeline: input -> mol2 -> add H (pH 7) -> add MMFF94 charges.
    Steps 2 and 3 parallelize obabel across `workers` chunks.
    """
    tmpdir = tempfile.mkdtemp(prefix='msk_ligprep_')
    try:
        suffix = Path(input_file).suffix.lower()

        # Step 1: ensure mol2 format (fast, single obabel call)
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

        # Step 2: add hydrogens at pH 7 (parallel across chunks)
        h_mol2 = os.path.join(tmpdir, 'h.mol2')
        if not _obabel_parallel(raw_mol2, h_mol2, ['-p', '7'], n_chunks=workers):
            logger.error("H-addition failed")
            return False

        # Step 3: add MMFF94 charges (parallel across chunks)
        if not _obabel_parallel(
                h_mol2, output_mol2, ['--partialcharge', 'mmff94'], n_chunks=workers):
            logger.error("Charge assignment failed")
            return False

        logger.info(f"Prepared ligand mol2 -> {output_mol2}")
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _file_ok(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def canonicalize_mol2_atom_names(input_mol2: str, output_mol2: str) -> None:
    """Rewrite mol2 atom names to <element><per-element-index> (C1, C2, ..., N1, N2, ...).

    Two mol2 files run through this function will end up with matching atom
    names on every heavy atom, provided obabel preserved heavy-atom ordering
    (it does, for MMFF94 charge assignment on already-hydrogenated input).

    Hydrogens get numbered independently (H1, H2, ...) - they never appear as
    BRICS keyatoms so their naming doesn't affect keyatom lookup.
    """
    with open(input_mol2) as f:
        lines = f.readlines()

    per_elem_counter = {}
    in_atom = False
    out_lines = []
    for line in lines:
        # Reset counter per compound so names are stable across files that
        # differ in inter-compound atom counts (e.g. after obabel adds H).
        if line.startswith('@<TRIPOS>MOLECULE'):
            per_elem_counter = {}
            in_atom = False
            out_lines.append(line)
            continue
        if line.startswith('@<TRIPOS>ATOM'):
            in_atom = True
            out_lines.append(line)
            continue
        if line.startswith('@<TRIPOS>'):
            in_atom = False
            out_lines.append(line)
            continue
        if in_atom and line.strip():
            parts = line.split()
            if len(parts) >= 6:
                elem_atomtype = parts[5]  # e.g. "C.ar", "N.pl3", "H"
                elem = elem_atomtype.split('.')[0]
                per_elem_counter[elem] = per_elem_counter.get(elem, 0) + 1
                parts[1] = f'{elem}{per_elem_counter[elem]}'
                # Rebuild the line preserving the original whitespace layout
                # (mol2 is fixed-width-ish; use consistent spacing)
                out_lines.append(
                    '{:>7s} {:<8s}{:>10s}{:>10s}{:>10s} {:<7s}{:>3s}  {:<8s}{:>10s}\n'.format(
                        parts[0], parts[1], parts[2], parts[3], parts[4],
                        parts[5], parts[6], parts[7],
                        parts[8] if len(parts) > 8 else '0.0000'))
                continue
        out_lines.append(line)

    with open(output_mol2, 'w') as f:
        f.writelines(out_lines)


# Common cofactors + metal ions that should be kept as part of the receptor.
# Drug-like HETATMs are stripped by default.
DEFAULT_KEEP_HETATMS = frozenset([
    # Metals
    'ZN', 'MG', 'CA', 'MN', 'FE', 'CU', 'K', 'NA', 'NI', 'CO', 'CD',
    'CS', 'HG', 'PB', 'SR', 'PT', 'AU', 'AG', 'LI', 'BA', 'RB',
    # Iron-sulfur clusters, hemes
    'FES', 'FE2', 'SF4', 'F3S', 'HEM', 'HEC', 'HEA', 'HEB', 'HED',
    # Nucleotide cofactors
    'NAD', 'NAI', 'NAP', 'NDP', 'FAD', 'FMN', 'FMR',
    'ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'GMP', 'CTP', 'UTP',
    # Common cofactors
    'PLP', 'PMP', 'COA', 'ACO', 'SAM', 'SAH', 'BTN', 'MDO', 'TPP',
    'CLA', 'BCL', 'CLR',
    # Waters (kept - scoring benefits from bridging waters)
    'HOH', 'WAT', 'H2O', 'DOD',
])


def strip_protein_hetatms(input_pdb: str, output_pdb: str,
                          extra_keep: set = None) -> tuple:
    """Copy input_pdb to output_pdb, keeping ATOM records and whitelisted HETATMs.

    Drug-like HETATMs (crystal ligands, buffer components, non-standard residues
    that are not in the cofactor whitelist) are dropped. This is important
    because the featurizer includes HETATMs in receptor xyz for grid clash
    filtering - a crystal ligand would carve out grid points in the pocket
    itself, degrading downstream scoring.

    Returns (n_atom, n_hetatm_kept, dropped_resnames_dict).
    """
    keep_hetatms = set(DEFAULT_KEEP_HETATMS)
    if extra_keep:
        keep_hetatms |= set(r.strip().upper() for r in extra_keep)

    n_atom = 0
    n_hetatm_kept = 0
    dropped = {}  # resname -> count
    with open(input_pdb) as fin, open(output_pdb, 'w') as fout:
        for line in fin:
            if line.startswith('ATOM'):
                n_atom += 1
                fout.write(line)
            elif line.startswith('HETATM'):
                resname = line[17:20].strip().upper()
                if resname in keep_hetatms:
                    n_hetatm_kept += 1
                    fout.write(line)
                else:
                    dropped[resname] = dropped.get(resname, 0) + 1
            elif line.startswith(('TER', 'END', 'CONECT', 'MODEL', 'ENDMDL', 'HEADER')):
                fout.write(line)
    return n_atom, n_hetatm_kept, dropped


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
            clash: float = 1.1,
            keep_hetatms: list = None,
            precompute_graphs: bool = False):
    """Run full preparation pipeline."""

    if target_id is None:
        target_id = Path(protein_pdb).stem

    target_dir = Path(output_dir) / target_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1a. Strip drug-like HETATMs from protein PDB (keep cofactors and metals).
    # If left in, the featurizer treats them as receptor atoms, and their
    # clash volumes carve holes in the pocket grid.
    stripped_pdb = str(target_dir / f"{target_id}_stripped.pdb")
    n_atom, n_het, dropped = strip_protein_hetatms(
        protein_pdb, stripped_pdb, extra_keep=keep_hetatms)
    if dropped:
        drop_summary = ", ".join(f"{r}({c})" for r, c in sorted(dropped.items()))
        logger.info(f"Dropped {sum(dropped.values())} non-cofactor HETATM lines: {drop_summary}")
        logger.info(f"  (to keep any of these, use --keep-hetatms RES1,RES2,...)")
    logger.info(f"Protein: {n_atom} ATOM + {n_het} whitelisted HETATM lines")

    # 1b. Protein protonation (optional)
    if protonate_rosetta_bin:
        prot_pdb = str(target_dir / f"{target_id}_H.pdb")
        if not protonate_rosetta(stripped_pdb, prot_pdb, protonate_rosetta_bin):
            sys.exit(1)
    else:
        prot_pdb = stripped_pdb
        logger.info(f"Using stripped protein PDB as-is: {prot_pdb}")

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
    #
    # NOTE ON ORDERING: BRICS runs on the RAW input mol2 (before obabel prep).
    # This is because obabel's MMFF94 output uses Sybyl types that RDKit's mol2
    # parser refuses (~75% drop rate observed on DUD-E). obabel preserves heavy
    # atom names during charge assignment, so keyatom names from the raw mol2
    # still match the final obabel-processed mol2 that predict reads.
    ligand_stem = Path(ligands_file).stem
    prepared_mol2 = str(target_dir / f"{ligand_stem}.mol2")

    # 3a. Convert to mol2 if needed and canonicalize atom names (unique per-element).
    #     Both the BRICS input and the final inference mol2 go through the same
    #     canonicalization -- keyatom names then match by construction.
    input_suffix = Path(ligands_file).suffix.lower()
    if input_suffix == '.sdf':
        raw_mol2 = str(target_dir / f"{ligand_stem}_raw.mol2")
        result = subprocess.run(
            ['obabel', '-isdf', ligands_file, '-omol2', '-O', raw_mol2],
            capture_output=True, text=True)
        if not _file_ok(raw_mol2):
            logger.error(f"SDF -> mol2 for BRICS input failed: {result.stderr}")
            sys.exit(1)
    else:
        raw_mol2 = ligands_file

    canonical_input = str(target_dir / f"{ligand_stem}_canonical.mol2")
    canonicalize_mol2_atom_names(raw_mol2, canonical_input)

    # 3b. BRICS on the canonicalized input.
    keyatom_npz = str(target_dir / f"{ligand_stem}.keyatom.def.npz")
    logger.info(f"Computing key atoms on canonical input (BRICS, {workers} workers)...")
    launch_batched_ligand(
        canonical_input,
        N=workers,
        collated_npz=keyatom_npz,
    )
    if not _file_ok(keyatom_npz):
        logger.error("Key atom computation failed (no output)")
        sys.exit(1)

    # 3c. Ligand prep for inference-ready mol2. Uses the canonicalized input
    # so obabel preserves the same heavy-atom ordering; we re-canonicalize
    # afterward to restore names in case obabel's -p 7 step stripped them.
    if skip_ligand_prep:
        shutil.copy2(canonical_input, prepared_mol2)
        logger.info(f"Copied canonicalized mol2 as-is -> {prepared_mol2}")
    else:
        tmp_prepared = str(target_dir / f"{ligand_stem}_obabel.mol2")
        if not prepare_ligand_mol2(canonical_input, tmp_prepared, workers=workers):
            sys.exit(1)
        canonicalize_mol2_atom_names(tmp_prepared, prepared_mol2)
        os.remove(tmp_prepared)

    n_compounds = count_mol2_compounds(prepared_mol2)
    logger.info(f"  {n_compounds} compounds in {prepared_mol2}")

    # 3d. Optional: precompute DGL ligand graphs so predict can skip all CPU
    # featurization. Only worth doing if the same prepared dir will be scored
    # against multiple checkpoints or multi-GPU predict is CPU-bound. Adds
    # ~60-90 sec/target at prep time; saves ~15-25 sec per predict pass.
    if precompute_graphs:
        try:
            from configs.config_loader import load_config as _load_cfg
            _cfg_path = str(Path(__file__).resolve().parent.parent.parent /
                            "configs" / "training" / "endtoend.yaml")
            graph_config = _load_cfg(_cfg_path)
            graphs_out_prefix = str(target_dir / ligand_stem)
            logger.info(f"Precomputing DGL ligand graphs ({workers} workers)...")
            n_graphs = build_ligand_graphs_batch(
                prepared_mol2, keyatom_npz, graphs_out_prefix,
                config=graph_config, N=workers,
            )
            logger.info(f"  -> {graphs_out_prefix}.graphs.bin ({n_graphs} graphs)")
        except Exception as e:
            logger.warning(f"Graph precomputation failed ({e}); predict will fall "
                           f"back to on-the-fly featurization")

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
    parser.add_argument('--keep-hetatms',
                        help='Comma-separated extra HETATM residue names to keep '
                             '(e.g. "MYR,SUC"). Metals and standard cofactors are '
                             'kept automatically.')
    parser.add_argument('--precompute-graphs', action='store_true',
                        help='Also save DGL ligand graphs to <stem>.graphs.bin so '
                             'predict skips CPU featurization. Adds ~60-90s/target '
                             'to prep; saves ~15-25s per predict pass. Worth it '
                             'when scoring the same library against multiple '
                             'checkpoints or if predict is CPU-bound.')

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

    keep_hetatms = None
    if args.keep_hetatms:
        keep_hetatms = [s.strip() for s in args.keep_hetatms.split(',') if s.strip()]

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
        keep_hetatms=keep_hetatms,
        precompute_graphs=args.precompute_graphs,
    )


if __name__ == '__main__':
    main()
