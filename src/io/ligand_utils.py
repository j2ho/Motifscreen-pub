import numpy as np

def get_atom_lines(mol2_file):
    with open(mol2_file,'r') as f:
        lines = f.readlines()

    atminfo = {}
    for i, ln in enumerate(lines):
        if ln.startswith('@<TRIPOS>MOLECULE'):
            cmpd = lines[i+1][:-1]
        if ln.startswith('@<TRIPOS>ATOM'):
            first_atom_idx = i+1
        if (ln.startswith('@<TRIPOS>BOND') or ln.startswith('@<TRIPOS>UNITY')) and (cmpd not in atminfo):
            last_atom_idx = i-1
            atminfo[cmpd] = lines[first_atom_idx:last_atom_idx+1]

    return atminfo


def xyz_from_mol2(mol2, centered=True):
    lines = get_atom_lines(mol2)
    atms = {}
    xyz = {}

    for key in lines:
        xyz[key] = []
        atms[key] = []

        coordinates = []
        for ln in lines[key]:
            x = ln.strip().split()
            atm = x[1]
            Rx = float(x[2])
            Ry = float(x[3])
            Rz = float(x[4])
            R = np.array([Rx,Ry,Rz])
            coordinates.append(R)
            atms[key].append(atm)
        coordinates = np.array(coordinates)
        center = np.average(coordinates,axis=0)
        if centered:
            xyz[key] = coordinates - center
        else:
            xyz[key] = coordinates
    return xyz, atms


def xyz_from_pdb(pdb, centered=True):
    atms = []
    xyz = {}
    with open(pdb,'r') as f:
        for l in f:
            if l.startswith('ATOM') or l.startswith('HETATM'):
                atm = l[12:16].strip()
                x = float(l[30:38])
                y = float(l[38:46])
                z = float(l[46:54])
                xyz[atm] = np.array([x,y,z])
                atms.append(atm)

    if centered:
        com = np.mean(list(xyz.values()), axis=0)
        for atm in xyz:
            xyz[atm] -= com

    return xyz, atms


def xyz_from_batch_pdb(pdb, centered=True):
    """Read a COMPND/ENDMDL-delimited batch PDB.

    Returns the same structure as xyz_from_mol2:
        xyz:  {compound_id: np.array shape (N,3)}
        atms: {compound_id: [atom_name, ...]}
    """
    xyz = {}
    atms = {}
    tag = None
    with open(pdb, 'r') as f:
        for l in f:
            if l.startswith('COMPND'):
                tag = l[:-1].split()[-1].replace('.pdb', '')
                atms[tag] = []
                coords = []
            elif l.startswith('ENDMDL'):
                if tag is not None and coords:
                    coordinates = np.array(coords)
                    if centered:
                        coordinates = coordinates - np.average(coordinates, axis=0)
                    xyz[tag] = coordinates
                tag = None
                coords = []
            elif (l.startswith('ATOM') or l.startswith('HETATM')) and tag is not None:
                atm = l[12:16].strip()
                coords.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
                atms[tag].append(atm)

    return xyz, atms