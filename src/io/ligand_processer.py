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


def _rdkit_mol_from_mol2_block(mol2_block):
    """Try progressively looser strategies to get an RDKit Mol from a mol2 block.

    LIT-PCBA and other Unistra/PubChem-derived mol2s have ~35% failure rate on
    strict `MolFromMol2Block`. Falling back to an obabel-mediated SDF round-trip
    rescues ~99.5% of the failures (verified on KAT2A, N=200 sample).

    Strategy order:
      1. MolFromMol2Block strict (fast, ~5ms)
      2. Obabel subprocess: mol2 -> SDF -> RDKit MolFromMolBlock (~50-80ms)

    Returns (mol, source_tag) where source_tag is 'direct' or 'obabel_sdf'.
    Returns (None, None) if all strategies fail.
    """
    m = Chem.MolFromMol2Block(mol2_block, sanitize=True, removeHs=True)
    if m is not None:
        return m, 'direct'

    # Fallback: obabel converts mol2 -> SDF, RDKit reads the SDF cleanly.
    # Preserves atom ordering so downstream name lookup by index still works.
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.mol2', mode='w', delete=False) as f:
        f.write(mol2_block)
        mol2_path = f.name
    sdf_path = mol2_path + '.sdf'
    try:
        result = subprocess.run(
            ['obabel', mol2_path, '-O', sdf_path],
            capture_output=True, timeout=10)
        if not os.path.exists(sdf_path) or os.path.getsize(sdf_path) == 0:
            return None, None
        with open(sdf_path) as f:
            sdf_block = f.read()
        m = Chem.MolFromMolBlock(sdf_block, sanitize=True, removeHs=True)
        if m is None:
            return None, None
        return m, 'obabel_sdf'
    except (subprocess.TimeoutExpired, Exception):
        return None, None
    finally:
        for p in (mol2_path, sdf_path):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


def _brics_keyatoms_from_mol2_block(mol2_block, mol_atom_names, atms, mol2xyz_tag):
    """RDKit-native BRICS: mol2 block -> fragments -> keyatom names.

    Returns list of atom names (one per BRICS fragment, closest-to-COM), padded
    to at least 4 keyatoms with random non-H heavy atoms if BRICS produces fewer.
    Returns None if the mol2 block cannot be parsed by RDKit even via fallback.
    """
    m, _ = _rdkit_mol_from_mol2_block(mol2_block)
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


def _prescreen_worker(args):
    """Worker: try direct MolFromMol2Block; return None if OK, tag if failed."""
    tag, block = args
    m = Chem.MolFromMol2Block(block, sanitize=True, removeHs=True)
    return None if m is not None else tag


def _brics_worker(args):
    """Worker: parse text block (mol2 or sdf), then BRICS on the mol."""
    tag, block, fmt, atms, mol2xyz_tag = args
    try:
        atom_names = _parse_atom_names_from_block(block) if fmt == 'mol2' else None
        if fmt == 'mol2':
            m = Chem.MolFromMol2Block(block, sanitize=True, removeHs=True)
        else:
            m = Chem.MolFromMolBlock(block, sanitize=True, removeHs=True)
        if m is None:
            return None
        if atom_names is None:
            atom_names = [a.GetSymbol() + str(i) for i, a in enumerate(m.GetAtoms())]
        keys = _brics_keyatoms_from_mol(m, atom_names, atms, mol2xyz_tag)
        return (tag, keys) if keys else None
    except Exception:
        return None


def _brics_keyatoms_from_mol(m, mol_atom_names, atms, mol2xyz_tag):
    """BRICS + keyatom picking on an already-parsed RDKit mol."""
    heavy_names = [name for name in mol_atom_names if not name.startswith('H')]
    if len(heavy_names) != m.GetNumAtoms():
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


def _batch_obabel_fallback(fail_blocks, timeout=1200):
    """One obabel subprocess converts all failed mol2 blocks to SDF at once.

    Amortizes fork+exec across the whole batch (~30-80ms per subprocess
    dropping to ~1ms per compound in aggregate).

    Returns dict {tag: sdf_block} for compounds obabel could convert.
    """
    if not fail_blocks:
        return {}
    import subprocess
    import tempfile
    import shutil as _shutil
    tmpdir = tempfile.mkdtemp(prefix='msk_batch_fallback_')
    try:
        mol2_path = os.path.join(tmpdir, 'fails.mol2')
        sdf_path = os.path.join(tmpdir, 'fails.sdf')
        with open(mol2_path, 'w') as f:
            for tag, block in fail_blocks:
                f.write(block)
        try:
            subprocess.run(
                ['obabel', mol2_path, '-O', sdf_path],
                capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            logger.warning(f"obabel batch fallback timed out (>{timeout}s)")
            return {}
        if not os.path.exists(sdf_path) or os.path.getsize(sdf_path) == 0:
            return {}
        # SDF blocks are $$$$-terminated. obabel writes each mol's title (== mol2 name)
        # on the first line of its SDF block. Match sdf blocks back to tags by title.
        recovered = {}
        buf = []
        with open(sdf_path) as f:
            for line in f:
                buf.append(line)
                if line.startswith('$$$$'):
                    sdf_block = ''.join(buf)
                    # First non-empty line is the title == compound tag
                    title = None
                    for L in sdf_block.splitlines():
                        s = L.strip()
                        if s:
                            title = s
                            break
                    if title is not None:
                        recovered[title] = sdf_block
                    buf = []
        return recovered
    finally:
        _shutil.rmtree(tmpdir, ignore_errors=True)


def launch_batched_ligand(ligand_f, N=10, collated_npz='keyatom.def.npz'):
    """Compute BRICS keyatoms for every compound in a multi-mol mol2 file.

    Two-pass batched design:
      1. Parallel prescreen: identify mol2 blocks RDKit's strict parser rejects.
      2. Single obabel subprocess: convert all failures to SDF at once.
      3. Parallel BRICS: parse (mol2 for direct-OK, SDF for recovered) + BRICS.

    Args:
        ligand_f: path to multi-molecule .mol2 (or a legacy multi-model .pdb)
        N: number of parallel workers
        collated_npz: output path for the {tag: [key_atom_names]} npz
    """
    if ligand_f.endswith('.pdb'):
        return _launch_batched_pdb_legacy(ligand_f, N, collated_npz)
    if not ligand_f.endswith('.mol2'):
        raise ValueError(f"Unsupported ligand format: {ligand_f}")

    mol2xyz, atms = xyz_from_mol2(ligand_f)
    blocks = [(tag, block) for tag, block in _read_mol2_blocks(ligand_f)
              if tag in mol2xyz]

    # Pass 1: parallel prescreen for failures
    prescreen_args = [(tag, block) for tag, block in blocks]
    if N > 1:
        with mp.Pool(processes=N) as pool:
            fail_tags = pool.map(_prescreen_worker, prescreen_args)
    else:
        fail_tags = [_prescreen_worker(a) for a in prescreen_args]
    fail_tag_set = {t for t in fail_tags if t is not None}
    n_fail = len(fail_tag_set)
    logger.info(f"  prescreen: {len(blocks) - n_fail}/{len(blocks)} direct-parse OK, "
                f"{n_fail} need SDF fallback")

    # Pass 2: batched obabel SDF conversion for the failures
    fail_blocks = [(tag, block) for tag, block in blocks if tag in fail_tag_set]
    recovered_sdf = _batch_obabel_fallback(fail_blocks) if fail_blocks else {}
    logger.info(f"  batched-fallback: {len(recovered_sdf)}/{n_fail} recovered as SDF")

    # Pass 3: parallel BRICS with pre-selected block format
    brics_args = []
    for tag, block in blocks:
        if tag in fail_tag_set:
            if tag in recovered_sdf:
                brics_args.append((tag, recovered_sdf[tag], 'sdf', atms[tag], mol2xyz[tag]))
            # else: dropped, no rescue
        else:
            brics_args.append((tag, block, 'mol2', atms[tag], mol2xyz[tag]))

    if N > 1:
        with mp.Pool(processes=N) as pool:
            results = pool.map(_brics_worker, brics_args)
    else:
        results = [_brics_worker(a) for a in brics_args]

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


## -- Precompute DGL ligand graphs (Tier 3 perf) --

_graph_worker_state = {}


def _init_graph_worker(gb_args):
    """Init per-worker GraphBuilder for parallel graph building.

    gb_args is a picklable tuple. GraphBuilder is instantiated inside each
    worker; config dataclasses pickle fine.
    """
    from src.data.dataset_jiho import GraphBuilder
    config_graph, config_augmentation, config_processing, static = gb_args
    _graph_worker_state['gb'] = GraphBuilder(
        config_graph=config_graph,
        config_augmentation=config_augmentation,
        config_processing=config_processing,
        static=static,
    )


def _graph_worker(args):
    """Build DGL ligand graph for one compound. Returns dict or None."""
    import torch
    (elems, qs, bonds, borders, xyz, nneighs, atms, atypes,
     ka_names, drop_H, tag) = args
    try:
        if not ka_names:
            return None
        gb = _graph_worker_state['gb']
        mol_tuple = (elems, qs, bonds, borders, xyz, nneighs, atypes)
        graph = gb.build_ligand_graph(mol_tuple, name=tag)
        if graph is None:
            return None
        com = torch.mean(graph.ndata['x'], axis=0).float()
        graph.ndata['x'] = (graph.ndata['x'] - com).float()
        # IMPORTANT: .copy() to break the memory view. Torch->numpy views can
        # become stale after the source tensor is GC'd in the worker.
        gdata_np = graph.gdata.detach().cpu().numpy().copy()
        filtered = [a for a, e in zip(atms, elems) if e != 'H'] if drop_H else atms
        ka_in_mol = [a for a in ka_names if a in filtered]
        if not ka_in_mol:
            return None
        indices = [filtered.index(a) for a in ka_in_mol]
        if len(indices) > 10:
            selected = np.random.choice(len(indices), 10, replace=False)
            indices = [indices[i] for i in selected]
            ka_in_mol = [ka_in_mol[i] for i in selected]
        return {
            'tag': tag,
            'graph': graph,
            'gdata': gdata_np,
            'key_indices': indices,
            'key_atom_names': ka_in_mol,
            'key_atom_orig_xyz': graph.ndata['x'][indices].detach().cpu().numpy().copy(),
        }
    except Exception:
        return None


def build_ligand_graphs_batch(mol2_path, keyatom_npz, out_prefix, config, N=8):
    """Precompute DGL ligand graphs + metadata from an already-prepped mol2.

    Reads existing keyatom.def.npz (from launch_batched_ligand) and builds
    DGL graphs in parallel worker processes. Writes:

      <out_prefix>.graphs.bin      dgl.save_graphs output for N compounds
      <out_prefix>.graphs.meta.npz per-graph metadata:
        - tags (N,)             compound_id per graph
        - gdata (N, 19)         global features per graph
        - key_indices (N,)      list per graph, filtered-atom indices
        - key_atom_names (N,)   list per graph, atom names
        - key_atom_orig_xyz (N,) list per graph, centered coords

    The graphs.bin + meta.npz pair lets predict skip all CPU featurization.
    """
    import dgl
    if not os.path.exists(keyatom_npz):
        raise RuntimeError(f"keyatom.def.npz not found: {keyatom_npz}")

    keyatoms_dict = np.load(keyatom_npz, allow_pickle=True)['keyatms'].item()

    # Parse mol2 once via the standard loader
    from src.data.dataset_jiho import MolecularLoader
    loader = MolecularLoader(config.paths, config.processing, config.augmentation)
    mol_data = loader.read_mol2_batch(mol2_path, tags=None)
    if mol_data is None:
        raise RuntimeError(f"Failed to read {mol2_path}")
    elems_l, qs_l, bonds_l, borders_l, xyz_l, nneighs_l, atms_l, atypes_l, tags_l = mol_data
    drop_H = config.processing.drop_H

    args_list = []
    for i, tag in enumerate(tags_l):
        args_list.append((
            elems_l[i], qs_l[i], bonds_l[i], borders_l[i], xyz_l[i],
            nneighs_l[i], atms_l[i], atypes_l[i],
            keyatoms_dict.get(tag), drop_H, tag,
        ))

    gb_args = (config.graph, config.augmentation, config.processing, True)

    if N > 1:
        with mp.Pool(processes=N, initializer=_init_graph_worker,
                     initargs=(gb_args,)) as pool:
            results = pool.map(_graph_worker, args_list)
    else:
        _init_graph_worker(gb_args)
        results = [_graph_worker(a) for a in args_list]

    graphs, tags_out, gdatas, key_indices, key_atom_names, key_atom_orig_xyz = \
        [], [], [], [], [], []
    for r in results:
        if r is None:
            continue
        graphs.append(r['graph'])
        tags_out.append(r['tag'])
        gdatas.append(r['gdata'])
        key_indices.append(r['key_indices'])
        key_atom_names.append(r['key_atom_names'])
        key_atom_orig_xyz.append(r['key_atom_orig_xyz'])

    if not graphs:
        raise RuntimeError(f"No graphs built for {mol2_path}")

    dgl.save_graphs(f"{out_prefix}.graphs.bin", graphs)
    np.savez(
        f"{out_prefix}.graphs.meta.npz",
        tags=np.array(tags_out, dtype=object),
        gdata=np.stack(gdatas, axis=0),
        key_indices=np.array(key_indices, dtype=object),
        key_atom_names=np.array(key_atom_names, dtype=object),
        key_atom_orig_xyz=np.array(key_atom_orig_xyz, dtype=object),
    )
    return len(graphs)


if __name__ == "__main__":
    import sys
    lig_file = sys.argv[1] if len(sys.argv) > 1 else "data/example/ligand.mol2"
    launch_batched_ligand(lig_file, N=1, collated_npz='keyatom.def.npz')
