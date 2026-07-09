import torch
import numpy as np
import copy
import csv
import os
from datetime import datetime

def calc_AUC(Pt, Pf):
    Pt = np.array(Pt)
    Pf = np.array(Pf)
    if Pt.size == 0 or Pf.size == 0:
        return -1.0

    comparison = Pt[:, None] - Pf[None, :]
    count = np.sum(comparison > 0) + 0.5 * np.sum(comparison == 0)

    auc = count / (Pt.size * Pf.size)
    return auc


def calc_enrichment_factor(Pt, Pf, fraction=0.01):
    """
    Calculate enrichment factor at a given fraction of the dataset.

    Args:
        Pt: Scores for positive (active) ligands
        Pf: Scores for negative (decoy) ligands
        fraction: Fraction of dataset to consider (default: 0.01 for 1%)

    Returns:
        Enrichment factor at the specified fraction
    """
    Pt = np.array(Pt)
    Pf = np.array(Pf)

    if Pt.size == 0 or Pf.size == 0:
        return -1.0

    # Combine all scores with labels
    all_scores = np.concatenate([Pt, Pf])
    all_labels = np.concatenate([np.ones(len(Pt)), np.zeros(len(Pf))])

    # Sort by scores in descending order
    sorted_indices = np.argsort(all_scores)[::-1]
    sorted_labels = all_labels[sorted_indices]

    # Calculate how many compounds to consider
    n_total = len(all_scores)
    n_selected = max(1, int(fraction * n_total))

    # Count actives in top fraction
    n_actives_found = np.sum(sorted_labels[:n_selected])

    # Calculate enrichment factor
    # EF = (actives_found / n_selected) / (total_actives / n_total)
    n_actives_total = len(Pt)

    # Expected number of actives in a random selection of n_selected compounds
    expected_actives = (n_actives_total / n_total) * n_selected

    if expected_actives == 0:
        return -1.0

    ef = n_actives_found / expected_actives
    return ef

def count_parameters(model):
    #print([p.numel() for p in model.parameters()])
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_cuda(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, tuple):
        return (to_cuda(v, device) for v in x)
    elif isinstance(x, list):
        return [to_cuda(v, device) for v in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v, device) for k, v in x.items()}
    else:
        # DGLGraph or other objects
        return x.to(device=device)


class EMA:
    """
    Exponential Moving Average for model parameters

    Args:
        model: The model to track
        decay: EMA decay rate (default: 0.9999)
        device: Device to store EMA weights
    """
    def __init__(self, model, decay=0.9999, device=None):
        self.model = model
        self.decay = decay
        self.device = device or next(model.parameters()).device

        # Create shadow parameters
        self.shadow = {}
        self.backup = {}

        # Initialize EMA weights with current model parameters
        self._init_shadow()

    def _init_shadow(self):
        """Initialize shadow parameters with current model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach().to(self.device)

    def update(self):
        """Update EMA weights: ema_weight = decay * ema_weight + (1 - decay) * current_weight"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    # EMA update: shadow = decay * shadow + (1 - decay) * param
                    self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data.to(self.device)

    def apply_shadow(self):
        """Apply EMA weights to model (for inference/validation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # Backup current weights
                self.backup[name] = param.data.clone()
                # Apply EMA weights
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model weights (after inference/validation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self):
        """Return EMA state dict for checkpointing"""
        return {
            'shadow': {name: param.clone() for name, param in self.shadow.items()},
            'decay': self.decay
        }

    def load_state_dict(self, state_dict):
        """Load EMA state dict from checkpoint"""
        self.shadow = {name: param.to(self.device) for name, param in state_dict['shadow'].items()}
        self.decay = state_dict.get('decay', self.decay)

    def to(self, device):
        """Move EMA weights to device"""
        self.device = device
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)
        return self


class MetricsLogger:
    """
    Logs training/validation metrics to CSV files for easy post-processing.

    Creates two CSV files per rank (in multi-GPU setup):
    - batch_metrics_rankX.csv: Per-batch metrics (very detailed, for debugging)
    - epoch_metrics_rankX.csv: Per-epoch summary metrics (for overall analysis)
    """

    def __init__(self, log_dir="logs", rank=0):
        """
        Initialize the metrics logger.

        Args:
            log_dir: Directory to save CSV files
            rank: GPU rank (for multi-GPU logging). If 0, logs to rank0 suffix; otherwise rankX
        """
        self.log_dir = log_dir
        self.rank = rank
        os.makedirs(log_dir, exist_ok=True)

        # Timestamped filenames with rank suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rank_suffix = f"_rank{rank}" if rank > 0 else ""
        self.batch_csv_path = os.path.join(log_dir, f"batch_metrics{rank_suffix}_{timestamp}.csv")
        self.epoch_csv_path = os.path.join(log_dir, f"epoch_metrics{rank_suffix}_{timestamp}.csv")
        self.binding_csv_path = os.path.join(log_dir, f"binding_predictions{rank_suffix}_{timestamp}.csv")

        # Track if CSV headers have been written
        self.batch_header_written = False
        self.epoch_header_written = False
        self.binding_header_written = False

        # Store fieldnames and rows to handle dynamic columns
        self.batch_fieldnames = []
        self.batch_rows = []
        self.epoch_fieldnames = [
            'epoch', 'mode',
            'auc_biolip', 'auc_chembl', 'auc_pdbbind',
            'auc_per_target_biolip', 'auc_per_target_chembl', 'auc_per_target_pdbbind',
            'ef1_per_target_biolip', 'ef1_per_target_chembl', 'ef1_per_target_pdbbind',
            'loss_keyatm_attmap', 'loss_motif_contrast', 'loss_motif_neg', 'loss_motif_pos',
            'loss_norm_penalty', 'loss_screening_bce', 'loss_screening_contrast',
            'loss_screening_rank', 'loss_structure', 'loss_structure_mae',
            'loss_structure_pair', 'loss_structure_rmsd', 'loss_total',
        ]
        self.epoch_rows = []
        self.binding_fieldnames = ['epoch', 'batch', 'target', 'ligand', 'binding_score', 'is_active']
        self.binding_rows = []

    def log_batch(self, epoch, batch, total_batches, mode, metrics):
        """
        Log batch-level metrics to CSV.

        Args:
            epoch: Epoch number
            batch: Batch number
            total_batches: Total batches in epoch
            mode: 'train' or 'valid'
            metrics: Dict of metric_name -> value
        """
        # Prepare row with common fields
        row = {
            'epoch': epoch,
            'batch': batch,
            'total_batches': total_batches,
            'mode': mode,
        }
        row.update(metrics)

        # Update fieldnames if new fields appear
        new_fields = set(row.keys()) - set(self.batch_fieldnames)
        if new_fields:
            self.batch_fieldnames.extend(sorted(new_fields))

        # Store row for later writing
        self.batch_rows.append(row)

        # Periodically flush to disk (every 10 rows)
        if len(self.batch_rows) >= 10:
            self._flush_batch_csv()

    def log_binding(self, epoch, batch, target, ligand_names, binding_scores, labels=None):
        """
        Log binding predictions to CSV.

        Args:
            epoch: Epoch number
            batch: Batch number
            target: Target protein name
            ligand_names: List of ligand names (decoys/actives)
            binding_scores: List of predicted binding scores
            labels: List of true labels (1=active, 0=decoy). If None, label column will be empty.
        """
        # Ensure lists are same length
        if len(ligand_names) != len(binding_scores):
            return

        if labels is not None and len(ligand_names) != len(labels):
            return

        # Add a row for each ligand
        for i, (lig_name, score) in enumerate(zip(ligand_names, binding_scores)):
            row = {
                'epoch': epoch,
                'batch': batch,
                'target': target,
                'ligand': lig_name,
                'binding_score': float(score),
                'is_active': int(labels[i]) if labels is not None else '',
            }
            self.binding_rows.append(row)

        # Periodically flush to disk (every 50 rows)
        if len(self.binding_rows) >= 50:
            self._flush_binding_csv()

    def log_epoch(self, epoch, mode, metrics):
        """
        Log epoch-level summary metrics to CSV.

        Args:
            epoch: Epoch number
            mode: 'train' or 'valid'
            metrics: Dict of metric_name -> value
        """
        # Round float values to 3 decimal places
        rounded_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, float):
                rounded_metrics[key] = round(value, 3)
            else:
                rounded_metrics[key] = value

        row = {
            'epoch': epoch,
            'mode': mode,
        }
        row.update(rounded_metrics)

        # Update fieldnames if new fields appear
        new_fields = set(row.keys()) - set(self.epoch_fieldnames)
        if new_fields:
            self.epoch_fieldnames.extend(sorted(new_fields))

        # Store row for later writing
        self.epoch_rows.append(row)

        # Flush epoch metrics immediately (one per epoch)
        self._flush_epoch_csv()

    def _flush_batch_csv(self):
        """Append all accumulated batch rows to CSV with proper headers."""
        if not self.batch_rows:
            return

        # Ensure fieldnames includes base fields first
        base_fields = ['epoch', 'batch', 'total_batches', 'mode']
        fieldnames = [f for f in base_fields if f in self.batch_fieldnames]
        fieldnames.extend([f for f in self.batch_fieldnames if f not in base_fields])

        # Append to file (create if doesn't exist)
        file_exists = os.path.exists(self.batch_csv_path)
        with open(self.batch_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
            # Only write header if file is new
            if not file_exists:
                writer.writeheader()
            for row in self.batch_rows:
                writer.writerow(row)

        self.batch_rows = []
        self.batch_header_written = True

    def _flush_epoch_csv(self):
        """Append all accumulated epoch rows to CSV with proper headers."""
        if not self.epoch_rows:
            return

        # Ensure fieldnames includes base fields first
        base_fields = ['epoch', 'mode']
        fieldnames = [f for f in base_fields if f in self.epoch_fieldnames]
        fieldnames.extend([f for f in self.epoch_fieldnames if f not in base_fields])

        # Append to file (create if doesn't exist)
        file_exists = os.path.exists(self.epoch_csv_path)
        with open(self.epoch_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
            # Only write header if file is new
            if not file_exists:
                writer.writeheader()
            for row in self.epoch_rows:
                writer.writerow(row)

        self.epoch_rows = []
        self.epoch_header_written = True

    def _flush_binding_csv(self):
        """Append all accumulated binding prediction rows to CSV."""
        if not self.binding_rows:
            return

        # Append to file (create if doesn't exist)
        file_exists = os.path.exists(self.binding_csv_path)
        with open(self.binding_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.binding_fieldnames, restval='')
            # Only write header if file is new
            if not file_exists:
                writer.writeheader()
            for row in self.binding_rows:
                writer.writerow(row)

        self.binding_rows = []
        self.binding_header_written = True

    def flush_all(self):
        """Flush all remaining data to disk. Call at end of training."""
        self._flush_batch_csv()
        self._flush_epoch_csv()
        self._flush_binding_csv()
