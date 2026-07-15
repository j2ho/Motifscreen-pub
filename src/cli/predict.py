#!/usr/bin/env python
"""Run MotifScreen-Aff inference on prepared data.

Supports single-GPU and multi-GPU (auto-detected or via --gpus).

Usage:
    # Single GPU (auto)
    uv run python motifscreen.py predict \
        --datapath prepared/ \
        --checkpoint models/best.pkl \
        --base-config configs/training/endtoend.yaml \
        --output scores.csv

    # Multi-GPU
    uv run python motifscreen.py predict \
        --datapath prepared/ \
        --checkpoint models/best.pkl \
        --base-config configs/training/endtoend.yaml \
        --gpus 0,1,2,3 \
        --output scores.csv
"""

import argparse
import itertools
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.inference.run_inference_general import (
    GeneralDataset,
    TimingStats,
    load_model,
    prepare_batch,
    _init_predict_worker,
)
from configs.config_loader import load_config, Config

import numpy as np
import pandas as pd
import torch
import dgl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def detect_targets(datapath: str) -> list:
    """Auto-detect targets: subdirectories containing .grid.npz."""
    targets = []
    for entry in sorted(Path(datapath).iterdir()):
        if entry.is_dir():
            grid_files = list(entry.glob('*.grid.npz'))
            if grid_files:
                targets.append(entry.name)
    return targets


# ---------------------------------------------------------------------------
# Single-GPU inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_target_single(model, dataset, target_id, batch_size, device):
    """Run inference on one target, single GPU."""
    receptor = dataset.load_receptor(target_id)
    if receptor is None:
        return None

    rec_graph = receptor['receptor_graph'].to(device)
    grid_idx = torch.tensor(receptor['grid_indices'], dtype=torch.long).to(device)

    results = []
    compound_count = 0

    for batch in dataset.iter_batches(target_id, batch_size):
        compound_count += len(batch)
        try:
            graphs = [c['graph'] for c in batch]
            key_indices = [c['key_indices'] for c in batch]

            batched = dgl.batch(graphs)
            gdata = torch.stack([g.gdata for g in graphs])
            setattr(batched, "gdata", gdata)

            key_matrices = [
                torch.eye(n)[idx]
                for n, idx in zip(batched.batch_num_nodes(), key_indices)
            ]
            nK = torch.tensor([len(idx) for idx in key_indices])

            _, _, _, _, bind_pred, _ = model(
                rec_graph,
                batched.to(device),
                [k.to(device) for k in key_matrices],
                grid_idx,
                gradient_checkpoint=False, drop_out=False,
            )

            if bind_pred is not None:
                scores = torch.sigmoid(bind_pred[0]).cpu().numpy()
                for j, c in enumerate(batch):
                    results.append({
                        'compound_id': c['compound_id'],
                        'score': float(scores[j]),
                        'source_mol2': c.get('source_mol2', ''),
                    })
        except Exception as e:
            logger.error(f"Batch error on {target_id}: {e}")
            continue

    if not results:
        return None

    logger.info(f"  {target_id}: {len(results)} compounds scored")
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Multi-GPU inference
# ---------------------------------------------------------------------------

class MultiGPURunner:
    """Thread-safe multi-GPU inference. One model replica + receptor per GPU."""

    def __init__(self, checkpoint: str, config: Config, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.models = {}
        self.locks = {}
        self.receptors = {}  # per-GPU receptor graph + grid indices

        for gpu_id in gpu_ids:
            device = torch.device(f'cuda:{gpu_id}')
            model = load_model(checkpoint, config, device)
            self.models[gpu_id] = model
            self.locks[gpu_id] = threading.Lock()

        logger.info(f"Loaded model on {len(gpu_ids)} GPUs: {gpu_ids}")

    def set_receptor(self, receptor_graph: dgl.DGLGraph, grid_indices: np.ndarray):
        """Move receptor to each GPU once (called per target)."""
        self.receptors = {}
        for gpu_id in self.gpu_ids:
            device = torch.device(f'cuda:{gpu_id}')
            self.receptors[gpu_id] = {
                'graph': receptor_graph.clone().to(device),
                'grid_idx': torch.tensor(grid_indices, dtype=torch.long).to(device),
            }

    @torch.no_grad()
    def run_batch(self, gpu_id: int, compounds: List[Dict]) -> List[Dict]:
        device = torch.device(f'cuda:{gpu_id}')
        model = self.models[gpu_id]
        rec = self.receptors[gpu_id]

        with self.locks[gpu_id]:
            try:
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

                _, _, _, _, bind_pred, _ = model(
                    rec['graph'],
                    batched.to(device),
                    [k.to(device) for k in key_matrices],
                    rec['grid_idx'],
                    gradient_checkpoint=False, drop_out=False,
                )

                results = []
                if bind_pred is not None:
                    scores = torch.sigmoid(bind_pred[0]).cpu().numpy()
                    for j, c in enumerate(compounds):
                        results.append({
                            'compound_id': c['compound_id'],
                            'score': float(scores[j]),
                            'source_mol2': c.get('source_mol2', ''),
                        })
                return results

            except Exception as e:
                logger.error(f"[GPU {gpu_id}] Batch error: {e}")
                return []


def run_target_multigpu(runner: MultiGPURunner, dataset: GeneralDataset,
                        target_id: str, batch_size: int,
                        build_pool, gpu_pool) -> Optional[pd.DataFrame]:
    """Run inference on one target across multiple GPUs.

    build_pool and gpu_pool are long-lived pools created in main() so that
    process startup happens once, not once per target.
    """
    receptor = dataset.load_receptor(target_id)
    if receptor is None:
        return None

    # Move receptor to each GPU once for this target
    runner.set_receptor(receptor['receptor_graph'], receptor['grid_indices'])

    gpu_cycle = itertools.cycle(runner.gpu_ids)
    results = []
    batch_count = 0
    compound_count = 0

    gpu_futures: List[Future] = []

    for batch in dataset.iter_batches_threaded(target_id, batch_size, build_pool):
        if not batch:
            continue

        batch_count += 1
        compound_count += len(batch)

        gpu_id = next(gpu_cycle)
        future = gpu_pool.submit(runner.run_batch, gpu_id, batch)
        gpu_futures.append(future)

        if batch_count % 50 == 0:
            logger.info(f"  [{target_id}] Batch {batch_count}: {compound_count} compounds")

    for future in gpu_futures:
        results.extend(future.result())

    if not results:
        return None

    logger.info(f"  {target_id}: {len(results)} compounds scored")
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run MotifScreen-Aff inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--datapath', required=True,
                        help='Path to prepared data directory')
    parser.add_argument('--targets', nargs='+',
                        help='Target IDs (default: auto-detect from datapath)')
    parser.add_argument('--checkpoint', required=True,
                        help='Model checkpoint (.pkl)')
    parser.add_argument('--base-config', required=True,
                        help='Model config YAML (e.g., configs/training/endtoend.yaml)')
    parser.add_argument('--run-name', default=None,
                        help='Run name. Outputs go to results/{run-name}/scores.csv')
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: scores.csv, or results/{run-name}/scores.csv)')
    parser.add_argument('--mol2-pattern', default='*.mol2',
                        help='Glob pattern for mol2 files (default: *.mol2)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Compounds per batch (default: 64)')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs (default: auto-detect all)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Graph building threads for multi-GPU (default: 8)')
    parser.add_argument('--device', default='cuda',
                        help='Device: cuda or cpu (default: cuda)')

    args = parser.parse_args()

    # Resolve output path
    if args.run_name:
        run_dir = os.path.join('results', args.run_name)
        output_path = os.path.join(run_dir, 'scores.csv')
    elif args.output:
        output_path = args.output
    else:
        output_path = 'scores.csv'

    datapath = os.path.expanduser(args.datapath)
    if not os.path.isdir(datapath):
        logger.error(f"Datapath not found: {datapath}")
        sys.exit(1)

    # Detect or validate targets
    if args.targets:
        targets = args.targets
    else:
        targets = detect_targets(datapath)
        if not targets:
            logger.error(f"No targets found in {datapath} (need subdirs with .grid.npz)")
            sys.exit(1)
    logger.info(f"Targets: {targets}")

    # Determine GPUs
    if args.device == 'cpu':
        gpu_ids = []
    elif args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(',')]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    model_config = load_config(args.base_config)
    dataset = GeneralDataset(model_config, datapath, args.mol2_pattern)

    use_multigpu = len(gpu_ids) > 1
    t_start = time.time()

    if use_multigpu:
        logger.info(f"Multi-GPU: {gpu_ids}")
        runner = MultiGPURunner(args.checkpoint, model_config, gpu_ids)

        # ProcessPool for CPU-bound ligand graph building (bypasses GIL).
        # Passed graph-builder config gets rebuilt inside each worker.
        gb_args = (
            model_config.graph,
            model_config.augmentation,
            model_config.processing,
            True,  # static=True
        )
        build_pool = ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_predict_worker,
            initargs=(gb_args,),
        )
        gpu_pool = ThreadPoolExecutor(max_workers=len(gpu_ids))
        try:
            all_dfs = []
            for target_id in targets:
                df = run_target_multigpu(
                    runner, dataset, target_id, args.batch_size,
                    build_pool, gpu_pool,
                )
                if df is not None:
                    df['target_id'] = target_id
                    all_dfs.append(df)
        finally:
            build_pool.shutdown(wait=True)
            gpu_pool.shutdown(wait=True)
    else:
        device = torch.device(f'cuda:{gpu_ids[0]}' if gpu_ids else 'cpu')
        logger.info(f"Single device: {device}")
        model = load_model(args.checkpoint, model_config, device)
        all_dfs = []
        for target_id in targets:
            df = run_target_single(model, dataset, target_id, args.batch_size, device)
            if df is not None:
                df['target_id'] = target_id
                all_dfs.append(df)

    total_time = time.time() - t_start

    if not all_dfs:
        logger.error("No results produced")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    cols = ['target_id', 'compound_id', 'score']
    if 'source_mol2' in combined.columns:
        cols.append('source_mol2')
    combined = combined[cols].sort_values('score', ascending=False)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    combined.to_csv(output_path, index=False)

    total_compounds = len(combined)
    throughput = total_compounds / total_time if total_time > 0 else 0
    logger.info(f"Saved {total_compounds} scores -> {output_path} "
                f"({total_time:.1f}s, {throughput:.0f} compounds/s)")


if __name__ == '__main__':
    main()
