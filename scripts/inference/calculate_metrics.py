#!/usr/bin/env python
"""
Calculate virtual screening metrics from inference results.

Metrics computed:
- AUROC: Area Under ROC Curve
- EF1: Enrichment Factor at 1%
- EF5: Enrichment Factor at 5%
- BEDROC: Boltzmann-Enhanced Discrimination of ROC (alpha=20 and alpha=80.5)

Usage:
    # Single target
    python scripts/inference/calculate_metrics.py --input results/chembl_test/B2RXC2_scores.csv

    # All targets in directory
    python scripts/inference/calculate_metrics.py --input-dir results/chembl_test/

    # Combined file with multiple targets
    python scripts/inference/calculate_metrics.py --input results/chembl_test_all_scores.csv --by-target
"""

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, fraction: float) -> float:
    """
    Calculate Enrichment Factor at a given fraction.

    EF = (actives in top X%) / (expected actives in top X%)
       = (actives in top X% / N_top) / (total actives / N_total)

    Args:
        y_true: Binary labels (1 = active, 0 = decoy)
        y_score: Predicted scores (higher = more likely active)
        fraction: Fraction of dataset to consider (e.g., 0.01 for EF1%)

    Returns:
        Enrichment factor value
    """
    n_total = len(y_true)
    n_actives_total = np.sum(y_true)

    if n_actives_total == 0:
        return 0.0

    # Sort by score descending
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Top X% of compounds
    n_top = max(1, int(n_total * fraction))
    n_actives_top = np.sum(y_true_sorted[:n_top])

    # Expected actives in random selection
    expected_actives = n_top * (n_actives_total / n_total)

    if expected_actives == 0:
        return 0.0

    ef = n_actives_top / expected_actives
    return ef


def bedroc(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    """
    Calculate BEDROC (Boltzmann-Enhanced Discrimination of ROC).

    BEDROC emphasizes early recognition of actives more than AUROC.
    Alpha parameter controls the degree of early recognition emphasis.
    Higher alpha = more emphasis on top-ranked compounds.

    Uses the RIE-based formula from Truchon & Bayly (2007), Eq. 3,
    consistent with RDKit's CalcBEDROC implementation.

    Reference:
        Truchon & Bayly, J. Chem. Inf. Model. 2007, 47, 488-508

    Args:
        y_true: Binary labels (1 = active, 0 = decoy)
        y_score: Predicted scores (higher = more likely active)
        alpha: Exponential weight parameter (default 20.0, standard in VS)

    Returns:
        BEDROC score in [0, 1]
    """
    n = len(y_true)
    n_actives = int(np.sum(y_true))

    if n_actives == 0 or n_actives == n:
        return 0.0

    # Sort by score descending
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Get ranks of actives (1-indexed)
    active_ranks = np.where(y_true_sorted == 1)[0] + 1

    # Calculate the exponential sum for observed active ranks
    s = np.sum(np.exp(-alpha * active_ranks / n))

    # Ratio of actives
    r_a = n_actives / n

    # Expected sum under random ranking (= n_actives * denom)
    random_sum = r_a * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)

    # RIE (Robust Initial Enhancement) = observed / expected
    RIE = s / random_sum

    # Analytic RIE bounds (Truchon & Bayly 2007, Eq. 4-5)
    RIE_max = (1 - np.exp(-alpha * r_a)) / (r_a * (1 - np.exp(-alpha)))
    RIE_min = (1 - np.exp(alpha * r_a)) / (r_a * (1 - np.exp(alpha)))

    # BEDROC = normalized RIE (Eq. 3)
    if RIE_max == RIE_min:
        return 0.0

    bedroc_score = (RIE - RIE_min) / (RIE_max - RIE_min)

    return bedroc_score


def calculate_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    Calculate all virtual screening metrics.

    Args:
        y_true: Binary labels (1 = active, 0 = decoy)
        y_score: Predicted scores

    Returns:
        Dictionary with metric names and values
    """
    metrics = {}

    n_actives = np.sum(y_true)
    n_decoys = len(y_true) - n_actives

    metrics['n_compounds'] = len(y_true)
    metrics['n_actives'] = int(n_actives)
    metrics['n_decoys'] = int(n_decoys)

    if n_actives == 0 or n_decoys == 0:
        logger.warning(f"Cannot calculate metrics: {n_actives} actives, {n_decoys} decoys")
        metrics['AUROC'] = np.nan
        metrics['EF1'] = np.nan
        metrics['EF5'] = np.nan
        metrics['BEDROC_a20'] = np.nan
        metrics['BEDROC_a80'] = np.nan
        return metrics

    # AUROC
    try:
        metrics['AUROC'] = roc_auc_score(y_true, y_score)
    except Exception as e:
        logger.warning(f"Could not calculate AUROC: {e}")
        metrics['AUROC'] = np.nan

    # Enrichment Factors
    metrics['EF1'] = enrichment_factor(y_true, y_score, 0.01)
    metrics['EF5'] = enrichment_factor(y_true, y_score, 0.05)

    # BEDROC at both standard alpha values
    metrics['BEDROC_a20'] = bedroc(y_true, y_score, alpha=20.0)
    metrics['BEDROC_a80'] = bedroc(y_true, y_score, alpha=80.5)

    return metrics


def process_single_file(filepath: str, score_col: str = 'score',
                        label_col: str = 'is_active') -> Optional[Dict[str, float]]:
    """
    Process a single results CSV file.

    Args:
        filepath: Path to CSV file
        score_col: Column name for predicted scores
        label_col: Column name for active/decoy labels

    Returns:
        Dictionary of metrics or None if error
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Could not read {filepath}: {e}")
        return None

    if score_col not in df.columns:
        logger.error(f"Score column '{score_col}' not found in {filepath}")
        return None

    if label_col not in df.columns:
        logger.error(f"Label column '{label_col}' not found in {filepath}. "
                    "Make sure inference was run with actives_csv configured.")
        return None

    y_true = df[label_col].values.astype(int)
    y_score = df[score_col].values.astype(float)

    return calculate_metrics(y_true, y_score)


def process_by_target(filepath: str, target_col: str = 'target_id',
                      score_col: str = 'score',
                      label_col: str = 'is_active') -> pd.DataFrame:
    """
    Process a combined CSV file with multiple targets.

    Args:
        filepath: Path to combined CSV file
        target_col: Column name for target ID
        score_col: Column name for predicted scores
        label_col: Column name for active/decoy labels

    Returns:
        DataFrame with metrics per target
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Could not read {filepath}: {e}")
        return pd.DataFrame()

    required_cols = [target_col, score_col, label_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing columns in {filepath}: {missing}")
        return pd.DataFrame()

    results = []
    targets = df[target_col].unique()

    for target in targets:
        target_df = df[df[target_col] == target]
        y_true = target_df[label_col].values.astype(int)
        y_score = target_df[score_col].values.astype(float)

        metrics = calculate_metrics(y_true, y_score)
        metrics['target_id'] = target
        results.append(metrics)

    results_df = pd.DataFrame(results)
    # Reorder columns
    cols = ['target_id', 'n_compounds', 'n_actives', 'n_decoys', 'AUROC', 'EF1', 'EF5', 'BEDROC_a20', 'BEDROC_a80']
    results_df = results_df[[c for c in cols if c in results_df.columns]]

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Calculate virtual screening metrics from inference results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single target file
    python scripts/inference/calculate_metrics.py --input results/chembl_test/B2RXC2_scores.csv

    # All CSV files in directory
    python scripts/inference/calculate_metrics.py --input-dir results/chembl_test/ --output metrics.csv

    # Combined file with multiple targets
    python scripts/inference/calculate_metrics.py --input results/chembl_test_all_scores.csv --by-target
        """
    )
    parser.add_argument('--input', type=str, default=None,
                        help='Input CSV file with scores and labels')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Directory containing per-target CSV files')
    parser.add_argument('--by-target', action='store_true',
                        help='Process combined file by target_id column')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for metrics (default: print to stdout)')
    parser.add_argument('--score-col', type=str, default='score',
                        help='Column name for predicted scores (default: score)')
    parser.add_argument('--label-col', type=str, default='is_active',
                        help='Column name for active/decoy labels (default: is_active)')
    parser.add_argument('--target-col', type=str, default='target_id',
                        help='Column name for target ID when using --by-target (default: target_id)')
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Must specify either --input or --input-dir")

    all_results = []

    if args.input:
        if args.by_target:
            # Process combined file by target
            results_df = process_by_target(
                args.input,
                target_col=args.target_col,
                score_col=args.score_col,
                label_col=args.label_col
            )
            if not results_df.empty:
                all_results.append(results_df)
        else:
            # Process single file
            metrics = process_single_file(
                args.input,
                score_col=args.score_col,
                label_col=args.label_col
            )
            if metrics:
                # Extract target name from filename
                target_name = Path(args.input).stem.replace('_scores', '')
                metrics['target_id'] = target_name
                all_results.append(pd.DataFrame([metrics]))

    if args.input_dir:
        # Process all CSV files in directory
        csv_files = glob.glob(os.path.join(args.input_dir, '*_scores.csv'))
        if not csv_files:
            csv_files = glob.glob(os.path.join(args.input_dir, '*.csv'))

        logger.info(f"Found {len(csv_files)} CSV files in {args.input_dir}")

        for filepath in sorted(csv_files):
            target_name = Path(filepath).stem.replace('_scores', '')
            logger.info(f"Processing {target_name}...")

            metrics = process_single_file(
                filepath,
                score_col=args.score_col,
                label_col=args.label_col
            )
            if metrics:
                metrics['target_id'] = target_name
                all_results.append(pd.DataFrame([metrics]))

    if not all_results:
        logger.error("No results to report")
        return

    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)

    # Reorder columns
    cols = ['target_id', 'n_compounds', 'n_actives', 'n_decoys', 'AUROC', 'EF1', 'EF5', 'BEDROC_a20', 'BEDROC_a80']
    final_df = final_df[[c for c in cols if c in final_df.columns]]

    # Calculate summary statistics
    metric_cols = ['AUROC', 'EF1', 'EF5', 'BEDROC_a20', 'BEDROC_a80']
    summary_mean = {
        'target_id': 'MEAN',
        'n_compounds': final_df['n_compounds'].mean(),
        'n_actives': final_df['n_actives'].mean(),
        'n_decoys': final_df['n_decoys'].mean(),
    }
    summary_std = {
        'target_id': 'STD',
        'n_compounds': final_df['n_compounds'].std(),
        'n_actives': final_df['n_actives'].std(),
        'n_decoys': final_df['n_decoys'].std(),
    }
    for col in metric_cols:
        summary_mean[col] = final_df[col].mean()
        summary_std[col] = final_df[col].std()
    summary_df = pd.DataFrame([summary_mean, summary_std])

    # Print results
    print("\n" + "="*80)
    print("VIRTUAL SCREENING METRICS")
    print("="*80)

    if len(final_df) > 1:
        print("\nPer-target results:")
        print(final_df.to_string(index=False, float_format='%.4f'))
        print("\n" + "-"*80)
        print("Summary (mean ± std across targets):")
        print(summary_df.to_string(index=False, float_format='%.4f'))
    else:
        print("\nResults:")
        print(final_df.to_string(index=False, float_format='%.4f'))

    print("="*80 + "\n")

    # Save to file if requested
    if args.output:
        # Append summary row
        output_df = pd.concat([final_df, summary_df], ignore_index=True)
        output_df.to_csv(args.output, index=False)
        logger.info(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    import os
    main()
