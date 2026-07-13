import os
import numpy as np

from rdkit import Chem
from rdkit.Chem.BRICS import BRICSDecompose, BreakBRICSBonds
from rdkit import RDLogger

import multiprocessing as mp
import logging

from src.io.ligand_utils import xyz_from_mol2, xyz_from_batch_pdb

RDLogger.DisableLog('rdApp.*')
logger = logging.getLogger(__name__)


def _read_mol2_blocks(mol2_path):
    """Yield (tag, block_text) for each compound in a multi-molecule mol2."""
    buf, tag = [], None
    with open(mol2_path) as f:
        for line in f:
            if line.startswith('@<TRIPOS>MOLECULE'):
                if buf:
                    yield tag, ''.join(buf)
                buf = [line]
                tag = None
            else:
                buf.append(line)
                if tag is None and len(buf) == 2:
                    tag = line.strip()
    if buf:
        yield tag, ''.join(buf)


def _parse_atom_names_from_block(mol2_block):
    """Extract ordered atom names from a mol2 block's TRIPOS ATOM section."""
    names = []
    in_atoms = False
    for line in mol2_block.splitlines():
        if line.startswith('@<TRIPOS>ATOM'):
            in_atoms = True
            continue
        if line.startswith('@<TRIPOS>'):
            in_atoms = False
            continue
        if in_atoms and line.strip():
            parts = line.split()
            if len(parts) >= 2:
                names.append(parts[1])
    return names


def _select_closest_to_com(atmxyz, frag_atm_names):
    """Given atom coords dict + a list of atom names in a fragment, pick the
    fragment atom closest to the fragment centroid."""
    valid = [a for a in frag_atm_names if a in atmxyz]
    if not valid:
        return None
    xyz_f = np.array([atmxyz[a] for a in valid])
    com_f = np.mean(xyz_f, axis=0)
    d2 = np.sum((xyz_f - com_f) ** 2, axis=1)
    return valid[int(np.argmin(d2))]


def _brics_keyatoms_from_mol2_block(mol2_block, mol_atom_names, atms, mol2xyz_tag):
    """RDKit-native BRICS: mol2 block -> fragments -> keyatom names.

    Returns list of atom names (one per BRICS fragment, closest-to-COM), padded
    to at least 4 keyatoms with random non-H heavy atoms if BRICS produces fewer.
    Returns None if the mol2 block cannot be parsed by RDKit.
    """
    m = Chem.MolFromMol2Block(mol2_block, sanitize=True, removeHs=True)
    if m is None:
        return None

    # Fallback: MolFromMol2Block strips H, so atom count may not match mol2's
    # original atom_names list. Build a heavy-atom-only name list aligned with
    # the RDKit mol's atom order. Assumption: RDKit reads mol2 atoms in order
    # and only heavy atoms survive removeHs=True.
    heavy_names = [name for name in mol_atom_names if not name.startswith('H')]
    if len(heavy_names) != m.GetNumAtoms():
        # Rare mismatch (e.g. bare H atom without a leading 'H' name, or
        # RDKit reordered). Fall back to all names in order and hope for the
        # best; downstream _select_closest_to_com filters unknown names.
        heavy_names = mol_atom_names[:m.GetNumAtoms()]

    for i, atm in enumerate(m.GetAtoms()):
        atm.SetAtomMapNum(i)

    try:
        list(BRICSDecompose(m))
        m2 = BreakBRICSBonds(m)
        frags = Chem.GetMolFrags(m2, asMols=True)
    except Exception:
        return None

    atmxyz = {a: x for a, x in zip(atms, mol2xyz_tag)}

    key_atm_list = []
    for f in frags:
        frag_atom_names = []
        for atm in f.GetAtoms():
            if atm.GetSymbol() == '*':
                continue
            i = atm.GetAtomMapNum()
            if i < len(heavy_names):
                frag_atom_names.append(heavy_names[i].strip())
        picked = _select_closest_to_com(atmxyz, frag_atom_names)
        if picked is not None:
            key_atm_list.append(picked)

    if len(key_atm_list) < 4:
        npick = 4 - len(key_atm_list)
        candidates = [a for a in atms if a not in key_atm_list and not a.startswith('H')]
        if candidates:
            toadd = list(np.random.choice(
                candidates, size=min(npick, len(candidates)), replace=False))
            key_atm_list += toadd

    return key_atm_list if key_atm_list else None


def _worker(args):
    """Multiprocessing worker: BRICS on one mol2 block."""
    tag, block, atms, mol2xyz_tag = args
    try:
        atom_names = _parse_atom_names_from_block(block)
        keys = _brics_keyatoms_from_mol2_block(block, atom_names, atms, mol2xyz_tag)
        return (tag, keys) if keys else None
    except Exception:
        return None


def launch_batched_ligand(ligand_f, N=10, collated_npz='keyatom.def.npz'):
    """Compute BRICS keyatoms for every compound in a multi-mol mol2 file.

    Direct mol2 -> RDKit -> BRICS. No PDB intermediate, no obabel dependency
    at this stage. Coordinates and atom names come from the input mol2.

    Args:
        ligand_f: path to multi-molecule .mol2 (or a .pdb multi-model file
                  for legacy compatibility)
        N: number of parallel workers
        collated_npz: output path for the {tag: [key_atom_names]} npz
    """
    if ligand_f.endswith('.mol2'):
        mol2xyz, atms = xyz_from_mol2(ligand_f)
        blocks = list(_read_mol2_blocks(ligand_f))
        args_list = []
        for tag, block in blocks:
            if tag in mol2xyz:
                args_list.append((tag, block, atms[tag], mol2xyz[tag]))
    elif ligand_f.endswith('.pdb'):
        # Legacy path preserved: users may still ship a multi-model PDB where
        # coords came from an already-hydrogen'd mol2 upstream. Fall back to
        # the PDB->RDKit route in this branch.
        return _launch_batched_pdb_legacy(ligand_f, N, collated_npz)
    else:
        raise ValueError(f"Unsupported ligand format: {ligand_f}")

    if N > 1:
        with mp.Pool(processes=N) as pool:
            results = pool.map(_worker, args_list)
    else:
        results = [_worker(a) for a in args_list]

    keyatms = {}
    for r in results:
        if r is None:
            continue
        tag, keys = r
        keyatms[tag] = keys

    np.savez(collated_npz, keyatms=keyatms)


# ---- Legacy PDB path (kept minimal for backwards compat) ------------------

def _pdb_worker_legacy(args):
    pdb, xyz, atms, trg = args
    try:
        m = Chem.MolFromPDBFile(pdb)
        if m is None:
            return None
        orgnames = []
        for i, atm in enumerate(m.GetAtoms()):
            ri = atm.GetPDBResidueInfo()
            orgnames.append(ri.GetName() if ri else atm.GetSymbol() + str(i))
            atm.SetAtomMapNum(i)
        list(BRICSDecompose(m))
        m2 = BreakBRICSBonds(m)
        frags = Chem.GetMolFrags(m2, asMols=True)
        atmxyz = {a: x for a, x in zip(atms, xyz)}
        key_atm_list = []
        for f in frags:
            frag_atm_names = []
            for atm in f.GetAtoms():
                if atm.GetSymbol() == '*':
                    continue
                i = atm.GetAtomMapNum()
                frag_atm_names.append(orgnames[i].strip())
            picked = _select_closest_to_com(atmxyz, frag_atm_names)
            if picked is not None:
                key_atm_list.append(picked)
        if len(key_atm_list) < 4:
            npick = 4 - len(key_atm_list)
            candidates = [a for a in atms if a not in key_atm_list and not a.startswith('H')]
            if candidates:
                key_atm_list += list(np.random.choice(
                    candidates, size=min(npick, len(candidates)), replace=False))
        return {trg: key_atm_list} if key_atm_list else None
    except Exception:
        return None


def _split_pdb_legacy(pdb, workpath):
    ligpdbs = []
    out = None
    for l in open(pdb):
        if l.startswith('COMPND'):
            tag = l[:-1].split()[-1].replace('.pdb', '')
            ligpdb = os.path.join(workpath, f'{tag}.pdb')
            ligpdbs.append([tag, ligpdb])
            out = open(ligpdb, 'w')
        elif l.startswith('ENDMDL') and out is not None:
            out.close()
            out = None
        elif l.startswith('ATOM') or l.startswith('CONECT'):
            if out is not None:
                out.write(l)
    return ligpdbs


def _launch_batched_pdb_legacy(pdb_file, N, collated_npz):
    import tempfile
    mol2xyz, atms = xyz_from_batch_pdb(pdb_file)
    workpath = tempfile.mkdtemp()
    try:
        ligpdbs = _split_pdb_legacy(pdb_file, workpath)
        args_list = [(pdb, mol2xyz[trg], atms[trg], trg) for trg, pdb in ligpdbs if trg in mol2xyz]
        if N > 1:
            with mp.Pool(processes=N) as pool:
                results = pool.map(_pdb_worker_legacy, args_list)
        else:
            results = [_pdb_worker_legacy(a) for a in args_list]
        keyatms = {}
        for r in results:
            if r is None:
                continue
            keyatms.update(r)
        np.savez(collated_npz, keyatms=keyatms)
    finally:
        for f in os.listdir(workpath):
            try:
                os.remove(os.path.join(workpath, f))
            except OSError:
                pass
        try:
            os.rmdir(workpath)
        except OSError:
            pass


if __name__ == "__main__":
    import sys
    lig_file = sys.argv[1] if len(sys.argv) > 1 else "data/example/ligand.mol2"
    launch_batched_ligand(lig_file, N=1, collated_npz='keyatom.def.npz')
