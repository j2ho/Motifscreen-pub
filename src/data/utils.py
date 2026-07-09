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


def read_mol2_batch(mol2, tags_read=None, drop_H=True, tag_only=False):
    qs_s, elems_s, xyzs_s = {}, {}, {}
    bonds_s, borders_s = {}, {}
    atms_s, nneighs_s, atypes_s = {}, {}, {}
    tags = []

    cont = open(mol2).readlines()
    il = [i for i, l in enumerate(cont) if l.startswith('@<TRIPOS>MOLECULE')] + [len(cont)]
    ihead = np.zeros(len(cont)+1, dtype=bool)
    ihead[il] = True

    # 초기화
    qs = elems = xyzs = bonds = borders = atms = atypes = []
    tag = None
    read_cont = 0

    def save_current_molecule():
        if tag is None or (tag_only or (tags_read is not None and tag not in tags_read)):
            return
        nneighs = [[0,0,0,0] for _ in qs]
        for i, j in bonds:
            if elems[i] in ['H','C','N','O']:
                k = ['H','C','N','O'].index(elems[i])
                nneighs[j][k] += 1.0
            if elems[j] in ['H','C','N','O']:
                l = ['H','C','N','O'].index(elems[j])
                nneighs[i][l] += 1.0

        if drop_H:
            nonHid = [i for i, a in enumerate(elems) if a != 'H']
        else:
            nonHid = list(range(len(elems)))
        nonH_set = set(nonHid)
        nonH_map = {old: new for new, old in enumerate(nonHid)}
        bonds_filt = [[nonH_map[i], nonH_map[j]] for i,j in bonds if i in nonH_set and j in nonH_set]
        borders_filt = [b for b, ij in zip(borders, bonds) if ij[0] in nonH_set and ij[1] in nonH_set]
        elems_s[tag]   = np.array(elems)[nonHid]
        qs_s[tag]      = np.array(qs)[nonHid]
        bonds_s[tag]   = bonds_filt
        borders_s[tag] = borders_filt
        xyzs_s[tag]    = np.array(xyzs)[nonHid]
        nneighs_s[tag] = np.array(nneighs, dtype=float)[nonHid]
        atms_s[tag]    = list(np.array(atms)[nonHid])
        atypes_s[tag]  = np.array(atypes)[nonHid]

    for i, l in enumerate(cont):
        if l.startswith('#'):
            continue

        if ihead[i]:
            save_current_molecule()

            read_cont = 3
            tag = cont[i+1].strip()
            if tag not in tags:
                tags.append(tag)
            qs, elems, xyzs, bonds, borders, atms, atypes = [], [], [], [], [], [], []
            continue

        if (not ihead[i+1] and len(l.strip()) <= 1) or tag is None:
            continue

        if read_cont == 3:
            if tags_read is None or tag in tags_read:
                read_cont = 4
            else:
                read_cont = -1
            continue

        if read_cont < 0 or tag_only:
            continue

        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            continue
        elif l.startswith('@<TRIPOS>BOND'):
            read_cont = 2
            continue
        elif l.startswith('@<TRIPOS>SUBSTRUCTURE') or l.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            read_cont = 0
            continue

        if read_cont == 1:
            words = l.strip().split()
            if len(words) < 6:
                continue
            name = words[1]
            if name.startswith('BR'):
                name = 'Br'
            # same elem logic as in read_mol2
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

        elif read_cont == 2:
            words = l.strip().split()
            if len(words) < 4:
                continue
            i1, i2 = int(words[1])-1, int(words[2])-1
            bondtype = {'0':0, '1':1, '2':2, '3':3, 'ar':3, 'am':2, 'du':0, 'un':0}.get(words[3], 0)
            bonds.append([i1, i2])
            borders.append(bondtype)

    save_current_molecule()

    tags_order = [tag for tag in (tags_read or tags) if tag in tags]
    if not tag_only:
        elems_s   = [elems_s[tag]   for tag in tags_order]
        qs_s      = [qs_s[tag]      for tag in tags_order]
        bonds_s   = [bonds_s[tag]   for tag in tags_order]
        borders_s = [borders_s[tag] for tag in tags_order]
        xyzs_s    = [xyzs_s[tag]    for tag in tags_order]
        nneighs_s = [nneighs_s[tag] for tag in tags_order]
        atms_s    = [atms_s[tag]    for tag in tags_order]
        atypes_s  = [atypes_s[tag]  for tag in tags_order]

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
