#!/usr/bin/env python
"""
General inference script for MotifScreen-Aff model.

Expected directory structure per target:
    {datapath}/{target_id}/
    ├── {target_id}.grid.npz
    ├── {target_id}.prop.npz
    ├── {target_id}.keyatom.def.npz   # or per-mol2: {mol2_stem}.keyatom.def.npz
    └── *.mol2                         # batch mol2 files to score

Output: compound_id, score, source_mol2, target_id
(No active/decoy labels - for general virtual screening)

Usage:
    python scripts/inference/run_inference_general.py \
        --datapath /path/to/data \
        --targets TARGET1 TARGET2 \
        --output results/scores.csv

    python scripts/inference/run_inference_general.py \
        --config configs/inference_general.yaml
"""

import os
import sys
import argparse
import glob
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import dgl
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_jiho import MolecularLoader, GraphBuilder
from src.data.utils import iter_mol2_batch
from src.model.models.msk1 import EndtoEndModel as MSK_1
from src.model.models.msk_ab import EndtoEndModel as MSK_ablation
from configs.config_loader import load_config, Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Track inference timing statistics."""
    model_load_time: float = 0.0
    total_time: float = 0.0
    n_targets: int = 0
    n_compounds: int = 0

    # Per-target breakdown
    receptor_load_times: List[float] = field(default_factory=list)
    mol2_load_times: List[float] = field(default_factory=list)
    forward_times: List[float] = field(default_factory=list)
    batch_counts: List[int] = field(default_factory=list)
    compound_counts: List[int] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)

    def add_target(self, target_id: str, receptor_time: float, mol2_time: float,
                   forward_time: float, n_batches: int, n_compounds: int):
        self.target_names.append(target_id)
        self.receptor_load_times.append(receptor_time)
        self.mol2_load_times.append(mol2_time)
        self.forward_times.append(forward_time)
        self.batch_counts.append(n_batches)
        self.compound_counts.append(n_compounds)
        self.n_targets += 1
        self.n_compounds += n_compounds

    def print_summary(self):
        """Print timing summary."""
        print("\n" + "=" * 70)
        print(" INFERENCE TIMING SUMMARY")
        print("=" * 70)

        print(f"\n  Model load time:     {self.model_load_time:.2f}s")
        print(f"  Total inference time: {self.total_time:.2f}s")
        print(f"  Targets processed:    {self.n_targets}")
        print(f"  Compounds scored:     {self.n_compounds}")

        if self.n_compounds > 0:
            throughput = self.n_compounds / self.total_time if self.total_time > 0 else 0
            avg_per_compound = self.total_time / self.n_compounds * 1000 if self.n_compounds > 0 else 0
            print(f"\n  Throughput:           {throughput:.1f} compounds/sec")
            print(f"  Avg time/compound:    {avg_per_compound:.2f} ms")

        if self.n_targets > 0:
            print(f"\n  Breakdown (averages):")
            print(f"    Receptor loading:   {np.mean(self.receptor_load_times):.3f}s")
            print(f"    Mol2 loading:       {np.mean(self.mol2_load_times):.3f}s")
            print(f"    Model forward:      {np.mean(self.forward_times):.3f}s")

            total_forward = sum(self.forward_times)
            total_load = sum(self.receptor_load_times) + sum(self.mol2_load_times)
            if total_forward + total_load > 0:
                print(f"\n  Time distribution:")
                print(f"    Data loading:       {total_load / (total_forward + total_load) * 100:.1f}%")
                print(f"    Model forward:      {total_forward / (total_forward + total_load) * 100:.1f}%")

        print("\n" + "-" * 70)
        print(" Per-Target Timing")
        print("-" * 70)
        print(f"  {'Target':<20} {'Compounds':>10} {'Load(s)':>10} {'Forward(s)':>12} {'ms/cmpd':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

        for i, target in enumerate(self.target_names):
            n_cmpd = self.compound_counts[i]
            load_time = self.receptor_load_times[i] + self.mol2_load_times[i]
            fwd_time = self.forward_times[i]
            ms_per_cmpd = (fwd_time / n_cmpd * 1000) if n_cmpd > 0 else 0
            print(f"  {target:<20} {n_cmpd:>10} {load_time:>10.2f} {fwd_time:>12.2f} {ms_per_cmpd:>10.1f}")

        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# ProcessPool worker for parallel ligand graph building
# ---------------------------------------------------------------------------
# Workers hold a GraphBuilder in module-level state. Initialized once per
# worker via ProcessPoolExecutor(initializer=_init_predict_worker, ...).

_worker_state = {}


def _init_predict_worker(graph_builder_args):
    """Initialize a graph builder in each worker process."""
    config_graph, config_augmentation, config_processing, static = graph_builder_args
    _worker_state['gb'] = GraphBuilder(
        config_graph=config_graph,
        config_augmentation=config_augmentation,
        config_processing=config_processing,
        static=static,
    )


def _noop_warmup(_):
    """Force worker process to fully spawn + init before any real work lands."""
    return _worker_state.get('gb') is not None


def _build_compound(args):
    """Build one DGL ligand graph. Kept for backwards compat / small runs."""
    return _build_compound_impl(args)


def _build_compound_impl(args):
    (elem, q, bond, border, coord, nneigh, atm, atype, tag,
     ka_names, drop_H, source) = args
    try:
        if not ka_names:
            return None
        gb = _worker_state['gb']
        mol_tuple = (elem, q, bond, border, coord, nneigh, atype)
        graph = gb.build_ligand_graph(mol_tuple, name=tag)
        if graph is None:
            return None
        com = torch.mean(graph.ndata['x'], axis=0).float()
        graph.ndata['x'] = (graph.ndata['x'] - com).float()
        filtered = ([a for a, e in zip(atm, elem) if e != 'H']
                    if drop_H else atm)
        key_atom_names = [a for a in ka_names if a in filtered]
        if not key_atom_names:
            return None
        indices = [filtered.index(a) for a in key_atom_names]
        if len(indices) > 10:
            selected = np.random.choice(len(indices), 10, replace=False)
            indices = [indices[i] for i in selected]
            key_atom_names = [key_atom_names[i] for i in selected]
        return {
            'compound_id': tag,
            'graph': graph,
            'key_indices': indices,
            'key_atom_names': key_atom_names,
            'key_atom_orig_xyz': graph.ndata['x'][indices].cpu().numpy(),
            'source_mol2': source,
        }
    except Exception:
        return None


def _build_compound_chunk(chunk_args):
    """Build a chunk of ligand graphs. Amortizes IPC over many compounds.

    Trades per-compound submit overhead for per-chunk pickle cost. On a 30k
    compound target with 8 workers, submitting 30k tasks costs seconds in
    pickling; submitting 8 chunks of 3750 costs milliseconds.
    """
    return [_build_compound_impl(a) for a in chunk_args]


@dataclass
class InferenceConfig:
    """Configuration for general inference"""
    checkpoint: str
    base_config: str
    datapath: str
    targets_file: Optional[str]
    mol2_pattern: str  # glob pattern for mol2 files, e.g., "*.mol2" or "*_b.mol2"
    batch_size: int
    output_dir: str
    output_pattern: str
    combined_output: Optional[str]
    device: str
    save_keyatom_xyz: bool  # Save key atom coordinates to NPZ file
    keyatom_xyz_pattern: str  # Output filename pattern for key atom xyz, e.g., "{target_id}_keyatom_xyz.npz"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'InferenceConfig':
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        return cls(
            checkpoint=cfg['model']['checkpoint'],
            base_config=cfg['model']['base_config'],
            datapath=os.path.expanduser(cfg['data']['datapath']),
            targets_file=cfg['data'].get('targets_file'),
            mol2_pattern=cfg['data'].get('mol2_pattern', '*.mol2'),
            batch_size=cfg['inference']['batch_size'],
            output_dir=cfg['inference']['output_dir'],
            output_pattern=cfg['inference'].get('output_pattern', '{target_id}_scores.csv'),
            combined_output=cfg['inference'].get('combined_output'),
            device=cfg['inference'].get('device', 'cuda'),
            save_keyatom_xyz=cfg['inference'].get('save_keyatom_xyz', False),
            keyatom_xyz_pattern=cfg['inference'].get('keyatom_xyz_pattern', '{target_id}_keyatom_xyz.npz'),
        )

    @classmethod
    def from_args(cls, args) -> 'InferenceConfig':
        """Create config from command line args"""
        return cls(
            checkpoint=args.checkpoint,
            base_config=args.base_config,
            datapath=os.path.expanduser(args.datapath),
            targets_file=args.targets_file,
            mol2_pattern=args.mol2_pattern,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            output_pattern=args.output_pattern,
            combined_output=args.combined_output,
            device=args.device,
            save_keyatom_xyz=args.save_keyatom_xyz,
            keyatom_xyz_pattern=args.keyatom_xyz_pattern,
        )


class GeneralDataset:
    """Dataset handler for general inference."""

    def __init__(self, model_config: Config, datapath: str, mol2_pattern: str):
        self.model_config = model_config
        self.datapath = datapath
        self.mol2_pattern = mol2_pattern

        self.loader = MolecularLoader(
            config_paths=model_config.paths,
            config_processing=model_config.processing,
            config_augmentation=model_config.augmentation
        )
        self.graph_builder = GraphBuilder(
            config_graph=model_config.graph,
            config_augmentation=model_config.augmentation,
            config_processing=model_config.processing,
            static=True
        )

    def load_receptor(self, target_id: str) -> Optional[Dict]:
        """Load receptor graph for target."""
        target_dir = os.path.join(self.datapath, target_id)
        grid_path = os.path.join(target_dir, f"{target_id}.grid.npz")
        prop_path = os.path.join(target_dir, f"{target_id}.prop.npz")

        if not os.path.exists(grid_path):
            logger.error(f"Grid file not found: {grid_path}")
            return None
        if not os.path.exists(prop_path):
            logger.error(f"Prop file not found: {prop_path}")
            return None

        try:
            grid_data = np.load(grid_path, allow_pickle=True)
            grids = grid_data['xyz']

            origin = torch.tensor(np.mean(grids, axis=0)).float()
            receptor_graph, processed_grids, grid_indices = self.graph_builder.build_receptor_graph(
                prop_path, grids, origin, gridchain=None
            )

            if receptor_graph is None:
                logger.error(f"Failed to build receptor graph for {target_id}")
                return None

            return {
                'receptor_graph': receptor_graph,
                'grid_indices': grid_indices,
                'target_dir': target_dir,
                'origin': origin.numpy(),  # Grid centroid for coordinate transform
            }
        except Exception as e:
            logger.error(f"Error loading receptor {target_id}: {e}")
            return None

    def find_mol2_files(self, target_id: str, target_dir: str) -> List[str]:
        """Find all mol2 files matching pattern in target directory."""
        pattern = os.path.join(target_dir, self.mol2_pattern)
        mol2_files = glob.glob(pattern)

        # Filter out keyatom files if pattern is too broad
        mol2_files = [f for f in mol2_files if not f.endswith('.keyatom.def.npz')]

        return sorted(mol2_files)

    def find_keyatom_file(self, target_id: str, target_dir: str, mol2_path: str) -> Optional[str]:
        """
        Find keyatom file for mol2. Searches in order:
        1. Same directory as mol2: {mol2_stem}.keyatom.def.npz
        2. Target root: {mol2_stem}.keyatom.def.npz
        3. Same directory as mol2: {target_id}.keyatom.def.npz (shared)
        4. Target root: {target_id}.keyatom.def.npz (shared)
        """
        mol2_stem = Path(mol2_path).stem
        mol2_dir = str(Path(mol2_path).parent)

        for directory in (mol2_dir, target_dir):
            for name in (f"{mol2_stem}.keyatom.def.npz", f"{target_id}.keyatom.def.npz"):
                path = os.path.join(directory, name)
                if os.path.exists(path):
                    return path

        return None

    def load_mol2_compounds(self, mol2_path: str, keyatom_path: str) -> List[Dict]:
        """Load all compounds from a mol2 file."""
        try:
            keyatoms_dict = self.loader.load_keyatoms(keyatom_path, targetname="")

            mol_data = self.loader.read_mol2_batch(mol2_path, tags=None)
            if mol_data is None:
                return []

            elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags = mol_data
            compounds = []

            for elem, q, bond, border, coord, nneigh, atm, atype, tag in zip(
                elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags
            ):
                try:
                    mol_tuple = (elem, q, bond, border, coord, nneigh, atype)
                    graph = self.graph_builder.build_ligand_graph(mol_tuple, name=tag)
                    if graph is None:
                        continue

                    com = torch.mean(graph.ndata['x'], axis=0).float()
                    graph.ndata['x'] = (graph.ndata['x'] - com).float()

                    if self.model_config.processing.drop_H:
                        filtered_atoms = [a for a, e in zip(atm, elem) if e != 'H']
                    else:
                        filtered_atoms = atm

                    key_indices, key_atom_names = self._get_key_indices(tag, filtered_atoms, keyatoms_dict)
                    if not key_indices:
                        continue

                    # Get original key atom coordinates (centered)
                    key_atom_orig_xyz = graph.ndata['x'][key_indices].cpu().numpy()

                    compounds.append({
                        'compound_id': tag,
                        'graph': graph,
                        'key_indices': key_indices,
                        'key_atom_names': key_atom_names,
                        'key_atom_orig_xyz': key_atom_orig_xyz,  # shape: [K, 1, 3]
                    })
                except Exception:
                    continue

            return compounds
        except Exception as e:
            logger.error(f"Error loading {mol2_path}: {e}")
            return []

    def _get_key_indices(self, compound_id: str, atoms: List[str],
                         keyatoms_dict: Dict) -> Tuple[List[int], List[str]]:
        """Get key atom indices and names for compound."""
        if compound_id not in keyatoms_dict:
            return [], []
        key_atom_names = [a for a in keyatoms_dict[compound_id] if a in atoms]
        indices = [atoms.index(a) for a in key_atom_names]
        if len(indices) > 10:
            selected = np.random.choice(len(indices), 10, replace=False)
            indices = [indices[i] for i in selected]
            key_atom_names = [key_atom_names[i] for i in selected]
        return indices, key_atom_names

    def _find_precomputed_graph_files(self, target_id: str) -> List[Tuple[str, str]]:
        """Find (graphs.bin, graphs.meta.npz) pairs in the target dir.

        Returns list of (bin_path, meta_path) for each mol2 that has both.
        """
        target_dir = os.path.join(self.datapath, target_id)
        pairs = []
        for f in sorted(os.listdir(target_dir)):
            if f.endswith('.graphs.bin'):
                stem = f[:-len('.graphs.bin')]
                bin_path = os.path.join(target_dir, f)
                meta_path = os.path.join(target_dir, f'{stem}.graphs.meta.npz')
                if os.path.exists(meta_path):
                    pairs.append((bin_path, meta_path))
        return pairs

    def load_precomputed_compounds(self, bin_path: str, meta_path: str) -> List[Dict]:
        """Load precomputed DGL graphs + metadata. Zero CPU featurization."""
        import dgl
        graphs, _ = dgl.load_graphs(bin_path)
        meta = np.load(meta_path, allow_pickle=True)
        tags = meta['tags']
        gdata = meta['gdata']  # (N, 19)
        key_indices = meta['key_indices']
        key_atom_names = meta['key_atom_names']
        key_atom_orig_xyz = meta['key_atom_orig_xyz']
        source = Path(bin_path).stem.replace('.graphs', '')
        out = []
        for i, g in enumerate(graphs):
            # Reattach gdata that dgl.save_graphs doesn't preserve
            setattr(g, 'gdata', torch.tensor(gdata[i]).float())
            out.append({
                'compound_id': str(tags[i]),
                'graph': g,
                'key_indices': list(key_indices[i]),
                'key_atom_names': list(key_atom_names[i]),
                'key_atom_orig_xyz': key_atom_orig_xyz[i],
                'source_mol2': source,
            })
        return out

    def iter_batches(self, target_id: str, batch_size: int):
        """Yield batches of compounds for a target (single-threaded).

        Fast path: precomputed .graphs.bin (skips CPU featurization).
        """
        precomputed = self._find_precomputed_graph_files(target_id)
        if precomputed:
            for bin_path, meta_path in precomputed:
                compounds = self.load_precomputed_compounds(bin_path, meta_path)
                batch = []
                for c in compounds:
                    batch.append(c)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch
            return

        target_dir = os.path.join(self.datapath, target_id)
        mol2_files = self.find_mol2_files(target_id, target_dir)
        # Also check batch_mol2s/ subdirectory
        batch_dir = os.path.join(target_dir, 'batch_mol2s')
        if os.path.isdir(batch_dir):
            mol2_files += self.find_mol2_files(target_id, batch_dir)

        batch = []
        for mol2_path in mol2_files:
            keyatom_path = self.find_keyatom_file(target_id, target_dir, mol2_path)
            if keyatom_path is None:
                continue
            compounds = self.load_mol2_compounds(mol2_path, keyatom_path)
            for c in compounds:
                c['source_mol2'] = Path(mol2_path).stem
                batch.append(c)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def iter_batches_threaded(self, target_id: str, batch_size: int, pool):
        """Yield batches with parallel graph building via a shared ProcessPool.

        Fast path: if <mol2_stem>.graphs.bin + .graphs.meta.npz exist for a
        mol2 file, load precomputed graphs (zero CPU featurization).
        Slow path: parse mol2 + build graphs in worker processes.
        """
        # Fast path: precomputed graphs
        precomputed = self._find_precomputed_graph_files(target_id)
        if precomputed:
            for bin_path, meta_path in precomputed:
                compounds = self.load_precomputed_compounds(bin_path, meta_path)
                batch = []
                for c in compounds:
                    batch.append(c)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch
            return

        target_dir = os.path.join(self.datapath, target_id)
        mol2_files = self.find_mol2_files(target_id, target_dir)
        batch_dir = os.path.join(target_dir, 'batch_mol2s')
        if os.path.isdir(batch_dir):
            mol2_files += self.find_mol2_files(target_id, batch_dir)

        drop_H = self.model_config.processing.drop_H

        for mol2_path in mol2_files:
            keyatom_path = self.find_keyatom_file(target_id, target_dir, mol2_path)
            if keyatom_path is None:
                continue

            keyatoms_dict = self.loader.load_keyatoms(keyatom_path, targetname="")
            source = Path(mol2_path).stem

            # Stream: consume compounds one at a time via iter_mol2_batch,
            # accumulate submit_chunk_size items, submit to build_pool, drain
            # completed futures to yield batches. Peak in-memory footprint is
            # ~(submit_chunk_size + a few in-flight futures + batch_size)
            # compounds, independent of the source mol2 size.
            n_workers = getattr(pool, '_max_workers', 8)
            submit_chunk_size = 128
            max_in_flight = n_workers * 4

            pending = []
            futures = []
            batch = []

            def _drain(block_until):
                """Pop and process completed futures until <= block_until remain."""
                while len(futures) > block_until:
                    fut = futures.pop(0)
                    for r in fut.result():
                        if r is None:
                            continue
                        batch.append(r)
                        if len(batch) >= batch_size:
                            yield batch
                            batch.clear()

            for elem, q, bond, border, coord, nneigh, atm, atype, tag in iter_mol2_batch(
                    mol2_path, drop_H=drop_H, tags_read=None):
                pending.append((elem, q, bond, border, coord, nneigh, atm, atype, tag,
                                keyatoms_dict.get(tag), drop_H, source))
                if len(pending) >= submit_chunk_size:
                    futures.append(pool.submit(_build_compound_chunk, pending))
                    pending = []
                    if len(futures) >= max_in_flight:
                        yield from _drain(max_in_flight - n_workers)

            if pending:
                futures.append(pool.submit(_build_compound_chunk, pending))
                pending = []

            yield from _drain(0)

            if batch:
                yield batch
                batch = []


def load_model(checkpoint: str, config: Config, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint."""
    if config.version == "v1.0":
        model = MSK_1(config)
    elif config.version == "ablation":
        model = MSK_ablation(config)
    else:
        raise ValueError(f"Unknown model version: {config.version}")

    model.to(device)

    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    logger.info(f"Loaded model from {checkpoint} (epoch {checkpoint_data.get('epoch', '?')})")

    model.eval()
    return model


def prepare_batch(compounds: List[Dict], receptor_graph: dgl.DGLGraph,
                  grid_indices: np.ndarray, device: torch.device) -> Tuple:
    """Prepare batch for inference."""
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
        receptor_graph.to(device),
        batched.to(device),
        [k.to(device) for k in key_matrices],
        torch.tensor(grid_indices, dtype=torch.long).to(device),
        nK.to(device)
    )


@torch.no_grad()
def run_target(model: torch.nn.Module, dataset: GeneralDataset,
               target_id: str, batch_size: int, device: torch.device,
               save_keyatom_xyz: bool = False,
               timing_stats: Optional[TimingStats] = None) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """Run inference on single target.

    Returns:
        Tuple of (DataFrame with scores, Dict with key atom xyz data if save_keyatom_xyz=True)
    """
    logger.info(f"Processing: {target_id}")

    # Time receptor loading
    t_receptor_start = time.perf_counter()
    receptor = dataset.load_receptor(target_id)
    t_receptor_end = time.perf_counter()
    receptor_time = t_receptor_end - t_receptor_start

    if receptor is None:
        return None, None

    target_dir = receptor['target_dir']
    mol2_files = dataset.find_mol2_files(target_id, target_dir)

    if not mol2_files:
        logger.error(f"No mol2 files found for {target_id}")
        return None, None

    logger.info(f"  Found {len(mol2_files)} mol2 files")

    all_results = []
    keyatom_xyz_data = {} if save_keyatom_xyz else None
    origin = receptor['origin']  # Grid centroid for coordinate transform

    # Timing accumulators
    mol2_load_time = 0.0
    forward_time = 0.0
    n_batches = 0

    for mol2_path in mol2_files:
        mol2_stem = Path(mol2_path).stem

        keyatom_path = dataset.find_keyatom_file(target_id, target_dir, mol2_path)
        if keyatom_path is None:
            logger.warning(f"  No keyatom file for {mol2_stem}, skipping")
            continue

        # Time mol2 loading
        t_mol2_start = time.perf_counter()
        compounds = dataset.load_mol2_compounds(mol2_path, keyatom_path)
        mol2_load_time += time.perf_counter() - t_mol2_start

        if not compounds:
            logger.warning(f"  No valid compounds in {mol2_stem}")
            continue

        logger.info(f"  {mol2_stem}: {len(compounds)} compounds")

        for i in range(0, len(compounds), batch_size):
            batch = compounds[i:i + batch_size]

            try:
                rec, lig, key_idx, grid_idx, nK = prepare_batch(
                    batch, receptor['receptor_graph'], receptor['grid_indices'], device
                )

                # Time model forward pass
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t_forward_start = time.perf_counter()

                Ykey_s, _, _, _, bind_pred, _ = model(
                    rec, lig, key_idx, grid_idx,
                    gradient_checkpoint=False, drop_out=False
                )

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                forward_time += time.perf_counter() - t_forward_start
                n_batches += 1

                if bind_pred is not None:
                    scores = torch.sigmoid(bind_pred[0]).cpu().numpy()

                    # Get predicted key atom coordinates if requested
                    if save_keyatom_xyz and Ykey_s is not None:
                        Ykey_s_np = Ykey_s.cpu().numpy()  # shape: [batch, max_K, 3]

                    for j, c in enumerate(batch):
                        all_results.append({
                            'compound_id': c['compound_id'],
                            'score': float(scores[j]),
                            'source_mol2': mol2_stem,
                        })

                        # Store key atom xyz data
                        if save_keyatom_xyz and Ykey_s is not None:
                            num_keyatoms = len(c['key_indices'])
                            pred_xyz_centered = Ykey_s_np[j, :num_keyatoms, :]  # In grid-centered frame
                            pred_xyz_pdb = pred_xyz_centered + origin  # Transform to original PDB frame
                            keyatom_xyz_data[c['compound_id']] = {
                                'atom_names': c['key_atom_names'],
                                'orig_xyz': c['key_atom_orig_xyz'],  # Ligand-COM centered frame
                                'pred_xyz_centered': pred_xyz_centered,  # Grid-centered frame
                                'pred_xyz': pred_xyz_pdb,  # Original PDB frame (comparable to grid.npz)
                                'origin': origin,  # Grid centroid used for centering
                                'score': float(scores[j]),
                                'source_mol2': mol2_stem,
                            }
            except Exception as e:
                logger.error(f"Batch error: {e}")
                continue

    # Record timing stats
    if timing_stats is not None:
        timing_stats.add_target(
            target_id=target_id,
            receptor_time=receptor_time,
            mol2_time=mol2_load_time,
            forward_time=forward_time,
            n_batches=n_batches,
            n_compounds=len(all_results)
        )

    if not all_results:
        return None, None

    df = pd.DataFrame(all_results)
    logger.info(f"  Total scored: {len(df)} compounds")
    return df, keyatom_xyz_data


def parse_targets(targets_arg: Optional[List[str]], targets_file: Optional[str]) -> List[str]:
    """Parse target list from args or file."""
    if targets_arg:
        return targets_arg

    if not targets_file or not os.path.exists(targets_file):
        raise ValueError("Must provide --targets or valid --targets-file")

    if targets_file.endswith('.csv'):
        df = pd.read_csv(targets_file)
        col = next((c for c in ['target_id', 'target_name', 'target', 'id'] if c in df.columns), df.columns[0])
        return df[col].astype(str).tolist()
    else:
        with open(targets_file) as f:
            return [l.strip() for l in f if l.strip() and not l.startswith('#')]


def main():
    parser = argparse.ArgumentParser(
        description='General inference for MotifScreen-Aff',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # With config file
    python scripts/inference/run_inference_general.py --config configs/inference.yaml

    # With command line args
    python scripts/inference/run_inference_general.py \\
        --datapath /path/to/data \\
        --targets TARGET1 TARGET2 \\
        --checkpoint models/best.pkl \\
        --base-config configs/model.yaml \\
        --output-dir results/
        """
    )

    # Config file (alternative to individual args)
    parser.add_argument('--config', help='Config YAML file')

    # Individual args (used if no config)
    parser.add_argument('--datapath', help='Base path to target directories')
    parser.add_argument('--targets', nargs='+', help='Target IDs')
    parser.add_argument('--targets-file', help='File with target IDs')
    parser.add_argument('--checkpoint', help='Model checkpoint path')
    parser.add_argument('--base-config', help='Model base config')
    parser.add_argument('--mol2-pattern', default='*.mol2', help='Glob pattern for mol2 files')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size')
    parser.add_argument('--output-dir', default='results/', help='Output directory')
    parser.add_argument('--output-pattern', default='{target_id}_scores.csv', help='Output filename pattern')
    parser.add_argument('--combined-output', help='Combined output CSV path')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save-keyatom-xyz', action='store_true',
                        help='Save key atom coordinates (original and predicted) to NPZ file')
    parser.add_argument('--keyatom-xyz-pattern', default='{target_id}_keyatom_xyz.npz',
                        help='Output filename pattern for key atom xyz')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for key-atom subsampling reproducibility')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load config
    if args.config:
        cfg = InferenceConfig.from_yaml(args.config)
        targets = parse_targets(args.targets, args.targets_file or cfg.targets_file)
        # Allow CLI overrides for specific options
        if args.save_keyatom_xyz:
            cfg.save_keyatom_xyz = True
        if args.keyatom_xyz_pattern != '{target_id}_keyatom_xyz.npz':
            cfg.keyatom_xyz_pattern = args.keyatom_xyz_pattern
    else:
        if not args.datapath or not args.checkpoint or not args.base_config:
            parser.error("Without --config, must provide --datapath, --checkpoint, and --base-config")
        cfg = InferenceConfig.from_args(args)
        targets = parse_targets(args.targets, args.targets_file)

    # Setup device
    device = torch.device('cuda' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Initialize timing stats
    timing_stats = TimingStats()

    # Load model (with timing)
    t_model_start = time.perf_counter()
    model_config = load_config(cfg.base_config)
    model = load_model(cfg.checkpoint, model_config, device)
    timing_stats.model_load_time = time.perf_counter() - t_model_start

    # Initialize dataset
    dataset = GeneralDataset(model_config, cfg.datapath, cfg.mol2_pattern)

    logger.info(f"Processing {len(targets)} targets")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Process targets (with timing)
    t_inference_start = time.perf_counter()
    all_dfs = []
    all_keyatom_xyz = {}
    for target_id in targets:
        df, keyatom_xyz_data = run_target(
            model, dataset, target_id, cfg.batch_size, device,
            save_keyatom_xyz=cfg.save_keyatom_xyz,
            timing_stats=timing_stats
        )
        if df is not None:
            df['target_id'] = target_id

            output_path = os.path.join(cfg.output_dir, cfg.output_pattern.format(target_id=target_id))
            df.to_csv(output_path, index=False)
            logger.info(f"Saved: {output_path}")

            all_dfs.append(df)

            # Save key atom xyz data for this target
            if cfg.save_keyatom_xyz and keyatom_xyz_data:
                xyz_output_path = os.path.join(
                    cfg.output_dir,
                    cfg.keyatom_xyz_pattern.format(target_id=target_id)
                )
                # Save as object array to preserve dict structure (load with allow_pickle=True)
                np.savez(xyz_output_path, keyatom_xyz=np.array(keyatom_xyz_data, dtype=object))
                logger.info(f"Saved key atom xyz: {xyz_output_path} ({len(keyatom_xyz_data)} compounds)")

                # Accumulate for combined output
                for cid, data in keyatom_xyz_data.items():
                    data['target_id'] = target_id
                    all_keyatom_xyz[f"{target_id}_{cid}"] = data

    # Record total inference time
    timing_stats.total_time = time.perf_counter() - t_inference_start

    # Combined output
    if cfg.combined_output and all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined[['target_id', 'compound_id', 'score', 'source_mol2']]
        combined.to_csv(cfg.combined_output, index=False)
        logger.info(f"Combined: {cfg.combined_output} ({len(combined)} compounds)")

        # Save combined key atom xyz
        if cfg.save_keyatom_xyz and all_keyatom_xyz:
            combined_xyz_path = cfg.combined_output.replace('.csv', '_keyatom_xyz.npz')
            np.savez(combined_xyz_path, keyatom_xyz=np.array(all_keyatom_xyz, dtype=object))
            logger.info(f"Combined key atom xyz: {combined_xyz_path} ({len(all_keyatom_xyz)} compounds)")

    # Print timing summary
    timing_stats.print_summary()

    logger.info("Done!")


if __name__ == "__main__":
    main()
