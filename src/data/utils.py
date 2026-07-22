import numpy as np
import scipy.spatial
from .types import ELEMS

def sasa_from_xyz(xyz, elems, probe_radius=1.4, n_samples=50):

    atomic_radii = {"C":  2.0,"N": 1.5,"O": 1.4,"S": 1.85,"H": 0.0, #ignore hydrogen for consistency
                    "F": 1.47,"Cl":1.75,"Br":1.85,"I": 2.0,'P': 1.8, 'Null':0.0}

    radii = np.array([atomic_radii[e] for e in elems])
    n_atoms = len(elems)

    # Fibonacci sphere probe points (vectorized)
    inc = np.pi * (3 - np.sqrt(5))
    off = 2.0 / n_samples
    ks = np.arange(n_samples)
    phi = ks * inc
    y = ks * off - 1 + (off / 2)
    r = np.sqrt(1 - y * y)
    pts0 = np.stack([np.cos(phi) * r, y, np.sin(phi) * r], axis=1)  # (n_samples, 3)

    kd = scipy.spatial.cKDTree(xyz)
    neighs_raw = kd.query_ball_tree(kd, 8.0)

    # Remove self from neighbor lists, find max neighbor count
    neighs = []
    for i in range(n_atoms):
        n = [x for x in neighs_raw[i] if x != i]
        neighs.append(n)

    n_neigh_counts = np.array([len(n) for n in neighs])
    max_neigh = int(n_neigh_counts.max()) if n_atoms > 0 else 0

    if max_neigh == 0:
        areas = list(4 * np.pi * (radii + probe_radius) ** 2)
        normareas = np.ones(n_atoms)
        occls = np.full(n_atoms, -2.0)  # (0 - 6) / 3
        return areas, normareas, occls

    # Pad neighbor index arrays for vectorized ops
    neigh_idx = np.zeros((n_atoms, max_neigh), dtype=int)
    neigh_mask = np.zeros((n_atoms, max_neigh), dtype=bool)
    for i, n in enumerate(neighs):
        nn = len(n)
        if nn > 0:
            neigh_idx[i, :nn] = n
            neigh_mask[i, :nn] = True

    neigh_xyz = xyz[neigh_idx]  # (n_atoms, max_neigh, 3)

    # --- Occlusion (vectorized) ---
    d2cen = np.sum((xyz[:, None, :] - neigh_xyz) ** 2, axis=2)  # (n_atoms, max_neigh)
    exp_d2 = np.exp(-d2cen / 6.0) * neigh_mask
    occls = np.sum(exp_d2, axis=1)
    occls = (occls - 6.0) / 3.0

    # --- SASA (vectorized) ---
    # Probe points per atom: pts0 * (radius_i + probe) + center_i
    expanded_radii = (radii + probe_radius)[:, None, None]  # (n_atoms, 1, 1)
    all_pts = pts0[None, :, :] * expanded_radii + xyz[:, None, :]  # (n_atoms, n_samples, 3)

    # Distance^2 from each probe to each neighbor
    # (n_atoms, n_samples, 1, 3) - (n_atoms, 1, max_neigh, 3) -> (n_atoms, n_samples, max_neigh)
    d2 = np.sum(
        (all_pts[:, :, None, :] - neigh_xyz[:, None, :, :]) ** 2,
        axis=3
    )

    # Neighbor radii thresholds
    r2 = (radii[neigh_idx] + probe_radius) ** 2 * 0.99  # (n_atoms, max_neigh)

    # Probe is inside if d2 < r2 for ANY valid neighbor
    inside = (d2 < r2[:, None, :]) & neigh_mask[:, None, :]  # (n_atoms, n_samples, max_neigh)
    any_inside = np.any(inside, axis=2)  # (n_atoms, n_samples)
    n_outsiders = np.sum(~any_inside, axis=1)  # (n_atoms,)

    areas = 4 * np.pi * (radii + probe_radius) ** 2 * n_outsiders / n_samples
    norm = 4 * np.pi * (radii + probe_radius)
    normareas = np.minimum(1.0, areas / norm)

    return list(areas), normareas, occls

def read_mol2(mol2,drop_H=False):
    read_cont = 0
    qs = []
    elems = []
    xyzs = []
    bonds = []
    borders = []
    atms = []

    for l in open(mol2):
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            continue
        if l.startswith('@<TRIPOS>BOND'):
            read_cont = 2
            continue
        if l.startswith('@<TRIPOS>SUBSTRUCTURE'):
            break
        if l.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            read_cont = 0
            continue

        words = l[:-1].split()
        if read_cont == 1:

            idx = words[0]
            # if words[1].startswith('BR'): words[1] = 'Br'
            # if words[1].startswith('CL'): words[1] = 'Cl'
            # if words[1].startswith('Br') or  words[1].startswith('Cl') :
            #     # elem = words[1][:2]
            #     elem = words[5].split('.')[0]
            # else:
            #     elem = words[1][0]

            # if elem == 'A' or elem == 'B' :
            #     elem = words[5].split('.')[0]
            try:
                elem = words[5].split('.')[0]
            except IndexError:
                elem = words[1][0]  # Fallback to first character of the atom name

            if elem not in ELEMS: elem = 'Null'

            atms.append(words[1])
            elems.append(elem)
            qs.append(float(words[-1]))
            xyzs.append([float(words[2]),float(words[3]),float(words[4])])

        elif read_cont == 2:
            # if words[3] == 'du' or 'un': rint(mol2)
            bonds.append([int(words[1])-1,int(words[2])-1]) #make 0-index
            bondtypes = {'0':0,'1':1,'2':2,'3':3,'ar':3,'am':2, 'du':0, 'un':0}
            borders.append(bondtypes[words[3]])

    nneighs = [[0,0,0,0] for _ in qs]
    for i,j in bonds:
        if elems[i] in ['H','C','N','O']:
            k = ['H','C','N','O'].index(elems[i])
            nneighs[j][k] += 1.0
        if elems[j] in ['H','C','N','O']:
            l = ['H','C','N','O'].index(elems[j])
            nneighs[i][l] += 1.0

    # drop hydrogens
    if drop_H:
        nonHid = [i for i,a in enumerate(elems) if a != 'H']
    else:
        nonHid = [i for i,a in enumerate(elems)]

    nonH_set = set(nonHid)
    nonH_map = {old: new for new, old in enumerate(nonHid)}
    borders = [b for b,ij in zip(borders,bonds) if ij[0] in nonH_set and ij[1] in nonH_set]
    bonds = [[nonH_map[i],nonH_map[j]] for i,j in bonds if i in nonH_set and j in nonH_set]

    return np.array(elems)[nonHid], np.array(qs)[nonHid], bonds, borders, np.array(xyzs)[nonHid], np.array(nneighs,dtype=float)[nonHid], list(np.array(atms)[nonHid])


def _finalize_mol2_molecule(elems, qs, bonds, borders, xyzs, atms, atypes, tag, drop_H):
    """Compute nneighs and (optionally) strip hydrogens for one molecule.

    Returns (elem, q, bond, border, coord, nneigh, atm, atype, tag) or None.
    Extracted from read_mol2_batch's inner save_current_molecule so both the
    streaming iterator and the list-returning batch reader can share it.
    """
    if tag is None or not qs:
        return None
    nneighs = [[0, 0, 0, 0] for _ in qs]
    for i, j in bonds:
        if elems[i] in ['H', 'C', 'N', 'O']:
            k = ['H', 'C', 'N', 'O'].index(elems[i])
            nneighs[j][k] += 1.0
        if elems[j] in ['H', 'C', 'N', 'O']:
            l = ['H', 'C', 'N', 'O'].index(elems[j])
            nneighs[i][l] += 1.0
    if drop_H:
        nonHid = [i for i, a in enumerate(elems) if a != 'H']
    else:
        nonHid = list(range(len(elems)))
    nonH_set = set(nonHid)
    nonH_map = {old: new for new, old in enumerate(nonHid)}
    bonds_filt = [[nonH_map[i], nonH_map[j]]
                  for i, j in bonds if i in nonH_set and j in nonH_set]
    borders_filt = [b for b, ij in zip(borders, bonds)
                    if ij[0] in nonH_set and ij[1] in nonH_set]
    return (
        np.array(elems)[nonHid],
        np.array(qs)[nonHid],
        bonds_filt,
        borders_filt,
        np.array(xyzs)[nonHid],
        np.array(nneighs, dtype=float)[nonHid],
        list(np.array(atms)[nonHid]),
        np.array(atypes)[nonHid],
        tag,
    )


def iter_mol2_batch(mol2, drop_H=True, tags_read=None):
    """Stream a multi-molecule mol2, yielding one compound tuple at a time.

    Yields (elem, q, bond, border, coord, nneigh, atm, atype, tag) per compound.

    Never holds more than one compound in memory at a time. Use this instead
    of read_mol2_batch when a file is large (>~500 MB) or the caller can
    consume incrementally -- the batch reader loads all lines into memory and
    stalls on multi-hundred-thousand-compound files.

    tags_read: if provided, only yield compounds whose tag is in this set.
    """
    tags_set = set(tags_read) if tags_read is not None else None
    STATE_IDLE, STATE_TAG, STATE_WAIT, STATE_ATOM, STATE_BOND, STATE_SKIP = range(6)

    qs, elems, xyzs, bonds, borders, atms, atypes = [], [], [], [], [], [], []
    tag = None
    state = STATE_IDLE

    def _reset():
        return [], [], [], [], [], [], []

    with open(mol2) as f:
        for l in f:
            if l.startswith('#'):
                continue

            if l.startswith('@<TRIPOS>MOLECULE'):
                if state not in (STATE_IDLE, STATE_SKIP):
                    out = _finalize_mol2_molecule(
                        elems, qs, bonds, borders, xyzs, atms, atypes, tag, drop_H)
                    if out is not None:
                        yield out
                qs, elems, xyzs, bonds, borders, atms, atypes = _reset()
                tag = None
                state = STATE_TAG
                continue

            if state == STATE_TAG:
                tag = l.strip()
                if tags_set is not None and tag not in tags_set:
                    state = STATE_SKIP
                else:
                    state = STATE_WAIT
                continue

            if state == STATE_SKIP:
                continue

            if l.startswith('@<TRIPOS>ATOM'):
                state = STATE_ATOM
                continue
            if l.startswith('@<TRIPOS>BOND'):
                state = STATE_BOND
                continue
            if l.startswith('@<TRIPOS>'):
                state = STATE_WAIT
                continue

            if state == STATE_ATOM:
                words = l.strip().split()
                if len(words) < 6:
                    continue
                name = words[1]
                if name.startswith('BR'):
                    name = 'Br'
                try:
                    elem = words[5].split('.')[0]
                except IndexError:
                    if name.startswith('Br') or name.startswith('Cl'):
                        elem = name[:2]
                    else:
                        elem = name[0]
                if elem not in ELEMS:
                    elem = 'Null'
                atms.append(words[1])
                atypes.append(words[5])
                elems.append(elem)
                xyzs.append([float(words[2]), float(words[3]), float(words[4])])
                qs.append(float(words[-1]) if len(words) >= 9 else 0.0)
            elif state == STATE_BOND:
                words = l.strip().split()
                if len(words) < 4:
                    continue
                i1, i2 = int(words[1]) - 1, int(words[2]) - 1
                bondtype = {'0': 0, '1': 1, '2': 2, '3': 3, 'ar': 3, 'am': 2, 'du': 0, 'un': 0}.get(words[3], 0)
                bonds.append([i1, i2])
                borders.append(bondtype)

    if state not in (STATE_IDLE, STATE_SKIP):
        out = _finalize_mol2_molecule(
            elems, qs, bonds, borders, xyzs, atms, atypes, tag, drop_H)
        if out is not None:
            yield out


def read_mol2_batch(mol2, tags_read=None, drop_H=True, tag_only=False):
    """Read all compounds from a multi-molecule mol2 into parallel lists.

    Backward-compatible wrapper around iter_mol2_batch. For very large files
    prefer iter_mol2_batch directly to avoid holding every compound in memory.
    """
    if tag_only:
        tags = []
        with open(mol2) as f:
            saw_molecule = False
            for l in f:
                if l.startswith('@<TRIPOS>MOLECULE'):
                    saw_molecule = True
                    continue
                if saw_molecule:
                    tags.append(l.strip())
                    saw_molecule = False
        return [], [], [], [], [], [], [], [], tags

    elems_s, qs_s, bonds_s, borders_s = [], [], [], []
    xyzs_s, nneighs_s, atms_s, atypes_s = [], [], [], []
    tags_order = []
    for out in iter_mol2_batch(mol2, drop_H=drop_H, tags_read=tags_read):
        elem, q, bond, border, coord, nneigh, atm, atype, tag = out
        elems_s.append(elem)
        qs_s.append(q)
        bonds_s.append(bond)
        borders_s.append(border)
        xyzs_s.append(coord)
        nneighs_s.append(nneigh)
        atms_s.append(atm)
        atypes_s.append(atype)
        tags_order.append(tag)
    return elems_s, qs_s, bonds_s, borders_s, xyzs_s, nneighs_s, atms_s, atypes_s, tags_order


def read_mol2s_xyzonly(mol2):
    read_cont = 0
    xyzs = []
    atms = []

    for l in open(mol2):
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            xyzs.append([])
            atms.append([])
            continue
        if l.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            read_cont = 0
            continue

        if l.startswith('@<TRIPOS>BOND'):
            read_cont = 2
            continue

        words = l[:-1].split()
        if read_cont == 1:
            is_H = (words[1][0] == 'H')
            if not is_H:
                atms[-1].append(words[1])
                xyzs[-1].append([float(words[2]),float(words[3]),float(words[4])])

    return np.array(xyzs), atms
