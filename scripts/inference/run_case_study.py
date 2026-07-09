#!/usr/bin/env python
"""
Case study inference with visualization outputs.

Flat directory layout:
    {datapath}/{target}.grid.npz
    {datapath}/{target}.prop.npz
    {datapath}/{target}.keyatom.def.npz
    {datapath}/{target}_lig.qh.mol2      (or {target}_lig.qh.conf.mol2 for conformers)

Outputs per (target, compound):
    - scores.csv: compound_id, score, per_key_scores
    - {compound}_keyatoms_pred.pdb: predicted key atom coords (B-factor = per-key score)
    - {target}_motif_pred.pdb: grid points (B-factor = max motif score across types)
    - {target}_motif_pred_types.npz: full NxT motif prediction matrix

Usage:
    python scripts/inference/run_case_study.py \
        --datapath case-studies \
        --targets 7qie 7urd \
        --checkpoint models/sm2e150-trans-b16/epoch26.pkl \
        --base-config configs/training/transfer.yaml \
        --output-dir results/case-studies
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import dgl

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_jiho import MolecularLoader, GraphBuilder
from src.model.models.msk1 import EndtoEndModel as MSK_1
from src.model.models.msk_ab import EndtoEndModel as MSK_ablation
from configs.config_loader import load_config, Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Motif type names from featurize_merge.py motif2label()
# Column 0 (H):  unused (mtype=0 = no motif)
# Column 1 (CB): H-bond donor + acceptor (both)
# Column 2 (CA): H-bond acceptor
# Column 3 (CD): H-bond donor
# Column 4 (CH): Aliphatic/hydrophobic contact
# Column 5 (CR): Aromatic ring contact
MOTIF_TYPES = ['H', 'CB', 'CA', 'CD', 'CH', 'CR']


# ---------------------------------------------------------------------------
# PDB writers
# ---------------------------------------------------------------------------

def write_keyatom_pdb(path, atom_names, xyz, scores, compound_id):
    """Write predicted key atom positions as PDB with per-key scores as B-factor.

    Args:
        path: output PDB file path
        atom_names: list of atom name strings
        xyz: (K, 3) array in PDB frame
        scores: (K,) array of per-key-atom contribution scores
        compound_id: compound identifier for REMARK
    """
    with open(path, 'w') as f:
        f.write(f"REMARK  compound: {compound_id}\n")
        f.write(f"REMARK  B-factor = per-key-atom binding contribution score\n")
        for i, (name, coord, sc) in enumerate(zip(atom_names, xyz, scores)):
            aname = name[:4].ljust(4)
            f.write(
                f"HETATM{i+1:5d}  {aname}LIG A   1    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00{sc:6.2f}\n"
            )
        f.write("END\n")


def write_motif_pdb_per_type(outdir, prefix, grid_xyz, motif_scores, motif_types=None):
    """Write per-type PDB files for grid motif scores.

    Creates one PDB per motif type, with B-factor = that type's score.
    Only includes grid points with score > 0.01 for that type.

    Args:
        outdir: output directory
        prefix: filename prefix (e.g. 'pred' or 'gt')
        grid_xyz: (N, 3) grid coordinates in PDB frame
        motif_scores: (N, T) scores (sigmoid for pred, raw labels for gt)
        motif_types: list of motif type names
    """
    if motif_types is None:
        motif_types = MOTIF_TYPES

    for tidx, tname in enumerate(motif_types):
        scores_t = motif_scores[:, tidx]
        mask = scores_t > 0.01
        if mask.sum() == 0:
            continue

        path = os.path.join(outdir, f"motif_{prefix}_{tname}.pdb")
        with open(path, 'w') as f:
            f.write(f"REMARK  Motif type: {tname} (index {tidx})\n")
            f.write(f"REMARK  Source: {prefix}\n")
            f.write(f"REMARK  B-factor = motif score for type {tname}\n")
            f.write(f"REMARK  Grid points with score > 0.01: {mask.sum()} of {len(scores_t)}\n")
            atom_idx = 1
            for coord, sc in zip(grid_xyz[mask], scores_t[mask]):
                f.write(
                    f"HETATM{atom_idx:5d}  {tname:2s}  MOT A   1    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                    f"  1.00{sc:6.2f}\n"
                )
                atom_idx += 1
            f.write("END\n")

    # Also write combined (max across types)
    max_scores = motif_scores.max(axis=1)
    max_type_idx = motif_scores.argmax(axis=1)
    path = os.path.join(outdir, f"motif_{prefix}_all.pdb")
    with open(path, 'w') as f:
        f.write(f"REMARK  Combined motif {prefix} (max score across types)\n")
        f.write(f"REMARK  B-factor = max score, atom name = dominant type\n")
        f.write(f"REMARK  Types: {', '.join(f'{i}={t}' for i, t in enumerate(motif_types))}\n")
        for i, (coord, sc, tidx) in enumerate(zip(grid_xyz, max_scores, max_type_idx)):
            tname = motif_types[tidx] if tidx < len(motif_types) else 'X'
            f.write(
                f"HETATM{i+1:5d}  {tname:2s}  MOT A   1    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00{sc:6.2f}\n"
            )
        f.write("END\n")


# ---------------------------------------------------------------------------
# Data loading (flat directory)
# ---------------------------------------------------------------------------

def load_receptor(datapath, target, graph_builder):
    """Load receptor from flat directory layout."""
    grid_path = os.path.join(datapath, f"{target}.grid.npz")
    prop_path = os.path.join(datapath, f"{target}.prop.npz")

    if not os.path.exists(grid_path) or not os.path.exists(prop_path):
        logger.error(f"Missing grid/prop for {target}")
        return None

    grid_data = np.load(grid_path, allow_pickle=True)
    grids = grid_data['xyz']
    origin = np.mean(grids, axis=0)
    origin_t = torch.tensor(origin).float()

    receptor_graph, processed_grids, grid_indices = graph_builder.build_receptor_graph(
        prop_path, grids, origin_t, gridchain=None
    )
    if receptor_graph is None:
        logger.error(f"Failed to build receptor graph for {target}")
        return None

    return {
        'receptor_graph': receptor_graph,
        'grid_indices': grid_indices,
        'origin': origin,
        'grids_pdb': grids,  # original PDB-frame grid coordinates
    }


def load_compounds(datapath, target, loader, graph_builder, model_config):
    """Load ligand compounds from mol2 files."""
    import glob as glob_mod

    # Find mol2 files (try multiple patterns)
    patterns = [
        f"{target}_lig.qh.mol2",
        f"{target}_lig.qh.conf.mol2",
        f"{target}_lig.mol2",
        f"{target}.mol2",
    ]

    mol2_files = []
    for pat in patterns:
        p = os.path.join(datapath, pat)
        if os.path.exists(p):
            mol2_files.append(p)

    if not mol2_files:
        logger.error(f"No mol2 files found for {target}")
        return []

    # Find keyatom file
    keyatom_path = os.path.join(datapath, f"{target}.keyatom.def.npz")
    if not os.path.exists(keyatom_path):
        logger.error(f"Missing keyatom file for {target}")
        return []

    keyatoms_dict = loader.load_keyatoms(keyatom_path, targetname="")

    all_compounds = []
    for mol2_path in mol2_files:
        mol2_stem = Path(mol2_path).stem
        try:
            mol_data = loader.read_mol2_batch(mol2_path, tags=None)
            if mol_data is None:
                continue

            elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags = mol_data

            for elem, q, bond, border, coord, nneigh, atm, atype, tag in zip(
                elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags
            ):
                mol_tuple = (elem, q, bond, border, coord, nneigh, atype)
                graph = graph_builder.build_ligand_graph(mol_tuple, name=tag)
                if graph is None:
                    continue

                com = torch.mean(graph.ndata['x'], axis=0).float()
                graph.ndata['x'] = (graph.ndata['x'] - com).float()

                if model_config.processing.drop_H:
                    filtered_atoms = [a for a, e in zip(atm, elem) if e != 'H']
                else:
                    filtered_atoms = atm

                if tag not in keyatoms_dict:
                    continue
                key_indices = [filtered_atoms.index(a) for a in keyatoms_dict[tag] if a in filtered_atoms]
                if not key_indices:
                    continue
                if len(key_indices) > 10:
                    key_indices = list(np.random.choice(key_indices, 10, replace=False))

                key_atom_names = [filtered_atoms[i] for i in key_indices]

                all_compounds.append({
                    'compound_id': tag,
                    'graph': graph,
                    'key_indices': key_indices,
                    'key_atom_names': key_atom_names,
                    'source_mol2': mol2_stem,
                })

        except Exception as e:
            logger.error(f"Error loading {mol2_path}: {e}")

    return all_compounds


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def prepare_batch(compounds, receptor_graph, grid_indices, device):
    """Prepare batch for model forward pass."""
    graphs = [c['graph'] for c in compounds]
    key_indices = [c['key_indices'] for c in compounds]

    batched = dgl.batch(graphs)
    gdata = torch.stack([g.gdata for g in graphs])
    setattr(batched, "gdata", gdata)

    key_matrices = [
        torch.eye(n)[idx]
        for n, idx in zip(batched.batch_num_nodes(), key_indices)
    ]
    nK = torch.tensor([len(idx) for idx in key_indices])

    return (
        receptor_graph,
        batched.to(device),
        [k.to(device) for k in key_matrices],
        grid_indices,
        nK.to(device),
    )


def load_model(checkpoint, config, device):
    if config.version == "v1.0":
        model = MSK_1(config)
    elif config.version == "ablation":
        model = MSK_ablation(config)
    else:
        raise ValueError(f"Unknown model version: {config.version}")

    model.to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state['model_state_dict'], strict=False)
    logger.info(f"Loaded model from {checkpoint} (epoch {state.get('epoch', '?')})")
    model.eval()
    return model


@torch.no_grad()
def run_case_study(model, receptor, compounds, device, batch_size=16):
    """Run inference and return full predictions.

    Returns dict with:
        results: list of per-compound dicts (compound_id, score, per_key_scores, pred_xyz)
        motif_pred: (N_grid, T) motif predictions for grid points (sigmoid)
        grid_xyz_pdb: (N_grid, 3) grid coordinates in PDB frame
    """
    rec_graph = receptor['receptor_graph'].to(device)
    grid_idx_t = torch.tensor(receptor['grid_indices'], dtype=torch.long).to(device)
    origin = receptor['origin']

    all_results = []
    motif_pred_all = None

    for i in range(0, len(compounds), batch_size):
        batch = compounds[i:i + batch_size]

        try:
            rec, lig, key_idx, grid_idx, nK = prepare_batch(
                batch, rec_graph, grid_idx_t, device
            )

            Ykey_s, _, z_norm, motif_logits, bind_pred, _ = model(
                rec, lig, key_idx, grid_idx,
                gradient_checkpoint=False, drop_out=False
            )

            if bind_pred is None:
                continue

            # bind_pred = (Aff, Aff_contrast) for former_contrast mode
            bind_scores = torch.sigmoid(bind_pred[0]).cpu().numpy()  # (B,)
            per_key_scores = bind_pred[1].cpu().numpy()  # (B, K) per-key contribution

            # Motif predictions: take from grid nodes only (same for all compounds in batch)
            if motif_pred_all is None and motif_logits is not None:
                # motif_logits is N_rec x T for all receptor nodes
                # We want grid nodes only
                grid_motif = motif_logits[receptor['grid_indices']]
                motif_pred_all = torch.sigmoid(grid_motif).cpu().numpy()

            # Predicted key atom coordinates
            if Ykey_s is not None:
                Ykey_np = Ykey_s.cpu().numpy()  # (B, K_max, 3)
            else:
                Ykey_np = None

            for j, c in enumerate(batch):
                num_k = len(c['key_indices'])
                result = {
                    'compound_id': c['compound_id'],
                    'score': float(bind_scores[j]),
                    'per_key_scores': per_key_scores[j, :num_k],
                    'key_atom_names': c['key_atom_names'],
                    'source_mol2': c['source_mol2'],
                }
                if Ykey_np is not None:
                    # Transform from grid-centered to PDB frame
                    result['pred_xyz'] = Ykey_np[j, :num_k, :] + origin
                all_results.append(result)

        except Exception as e:
            logger.error(f"Batch error: {e}")
            import traceback
            traceback.print_exc()
            continue

    return {
        'results': all_results,
        'motif_pred': motif_pred_all,
        'grid_xyz_pdb': receptor['grids_pdb'],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Case study inference with visualization')
    parser.add_argument('--datapath', required=True, help='Directory with case study data')
    parser.add_argument('--targets', nargs='+', required=True, help='Target names (e.g. 7qie 7urd)')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--base-config', required=True, help='Model config YAML')
    parser.add_argument('--output-dir', default='results/case-studies', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Inference batch size')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for key-atom subsampling reproducibility')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    model_config = load_config(args.base_config)
    model = load_model(args.checkpoint, model_config, device)

    loader = MolecularLoader(
        config_paths=model_config.paths,
        config_processing=model_config.processing,
        config_augmentation=model_config.augmentation
    )
    graph_builder = GraphBuilder(
        config_graph=model_config.graph,
        config_augmentation=model_config.augmentation,
        config_processing=model_config.processing,
        static=True
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for target in args.targets:
        logger.info(f"=== {target} ===")
        target_outdir = os.path.join(args.output_dir, target)
        os.makedirs(target_outdir, exist_ok=True)

        # Load data
        receptor = load_receptor(args.datapath, target, graph_builder)
        if receptor is None:
            continue

        compounds = load_compounds(args.datapath, target, loader, graph_builder, model_config)
        if not compounds:
            logger.error(f"No compounds loaded for {target}")
            continue

        logger.info(f"  {len(compounds)} compounds loaded")

        # Run inference
        output = run_case_study(model, receptor, compounds, device, args.batch_size)

        # Save scores CSV
        rows = []
        for r in output['results']:
            rows.append({
                'compound_id': r['compound_id'],
                'score': r['score'],
                'source_mol2': r['source_mol2'],
                'per_key_scores': ','.join(f'{s:.4f}' for s in r['per_key_scores']),
                'key_atom_names': ','.join(r['key_atom_names']),
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(target_outdir, 'scores.csv'), index=False)
        logger.info(f"  Saved scores.csv ({len(df)} compounds)")

        # Save per-compound key atom PDBs
        for i, r in enumerate(output['results']):
            if 'pred_xyz' not in r:
                continue
            pdb_path = os.path.join(target_outdir, f"{i:03d}_{r['compound_id']}_{r['source_mol2']}_keyatoms.pdb")
            write_keyatom_pdb(
                pdb_path,
                r['key_atom_names'],
                r['pred_xyz'],
                r['per_key_scores'],
                r['compound_id'],
            )
        logger.info(f"  Saved key atom PDBs")

        # Save motif predictions (per-type PDBs)
        if output['motif_pred'] is not None:
            write_motif_pdb_per_type(
                target_outdir, 'pred',
                output['grid_xyz_pdb'],
                output['motif_pred'],
            )
            # Also save full motif matrix as NPZ
            np.savez(
                os.path.join(target_outdir, f"{target}_motif_pred.npz"),
                grid_xyz=output['grid_xyz_pdb'],
                motif_scores=output['motif_pred'],
                motif_types=MOTIF_TYPES,
            )
            n_active = (output['motif_pred'].max(axis=1) > 0.5).sum()
            logger.info(f"  Saved pred motif PDBs ({output['motif_pred'].shape[0]} grid points, "
                       f"{n_active} with score > 0.5)")

        # Save ground truth motif labels (per-type PDBs, loaded separately from grid.npz)
        grid_npz_path = os.path.join(args.datapath, f"{target}.grid.npz")
        grid_data = np.load(grid_npz_path, allow_pickle=True)
        if 'labels' in grid_data:
            gt_labels = grid_data['labels']  # (N, 6)
            write_motif_pdb_per_type(
                target_outdir, 'gt',
                output['grid_xyz_pdb'],
                gt_labels,
            )
            n_labeled = (gt_labels.sum(axis=1) > 0).sum()
            logger.info(f"  Saved GT motif PDBs ({n_labeled} labeled grid points)")

        logger.info(f"  Done: {target_outdir}")


if __name__ == '__main__':
    main()
