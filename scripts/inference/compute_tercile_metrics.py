"""Compute per-target AUROC/EF10 and per-tercile aggregates for the ChEMBL benchmark.

Reads:
  --scores    scores.csv with columns [target_id, compound_id, score, source_mol2]
  --labels    directory with per-target active_smiles_clu.csv files (from the Zenodo tarball)
  --manifest  manifest.tsv with columns [target_id, tercile, ...]

Writes:
  --metrics-out  per-target metrics (target_id, tercile, n, n_actives, auroc, ef10)

Prints per-tercile mean AUROC / EF10 to stdout.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_tercile(manifest_path):
    m = {}
    with open(manifest_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            m[row['target_id']] = row.get('tercile', 'unknown')
    return m


def load_actives(labels_dir, target):
    path = Path(labels_dir) / target / 'active_smiles_clu.csv'
    if not path.exists():
        return set()
    with open(path) as f:
        reader = csv.DictReader(f)
        col = 'chemblid(with_best_aff)'
        return {row[col] for row in reader if col in row}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True, help='Path to scores.csv from predict')
    ap.add_argument('--labels', required=True, help='Directory with per-target active_smiles_clu.csv')
    ap.add_argument('--manifest', required=True, help='Path to manifest.tsv (with tercile column)')
    ap.add_argument('--metrics-out', default=None, help='Output CSV path (default: derived from --scores)')
    args = ap.parse_args()

    metrics_out = args.metrics_out or args.scores.replace('.csv', '_metrics.csv')

    tercile = load_tercile(args.manifest)
    actives_cache = {}
    rows = []
    with open(args.scores) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row['target_id']
            if t not in actives_cache:
                actives_cache[t] = load_actives(args.labels, t)
            row['is_active'] = 1 if row['compound_id'] in actives_cache[t] else 0
            rows.append(row)

    per_target = {}
    for row in rows:
        per_target.setdefault(row['target_id'], []).append(
            (float(row['score']), row['is_active'])
        )

    target_metrics = {}
    for t, entries in per_target.items():
        scores = np.array([e[0] for e in entries])
        labels = np.array([e[1] for e in entries])
        n = len(entries)
        n_act = int(labels.sum())
        terc = tercile.get(t, 'unknown')
        if n_act == 0 or n_act == n:
            auroc = ef10 = float('nan')
        else:
            auroc = roc_auc_score(labels, scores)
            k = max(1, int(0.1 * n))
            topk = np.argsort(-scores)[:k]
            hit_rate = labels[topk].sum() / k
            base_rate = n_act / n
            ef10 = hit_rate / base_rate
        target_metrics[t] = (terc, n, n_act, auroc, ef10)

    with open(metrics_out, 'w') as f:
        f.write("target_id,tercile,n,n_actives,auroc,ef10\n")
        for t, (terc, n, n_act, auroc, ef10) in sorted(target_metrics.items()):
            f.write(f"{t},{terc},{n},{n_act},{auroc:.4f},{ef10:.4f}\n")

    print(f"Per-target metrics: {metrics_out}")
    print()

    per_terc = {}
    for t, (terc, n, n_act, auroc, ef10) in target_metrics.items():
        if not np.isnan(auroc):
            per_terc.setdefault(terc, []).append((auroc, ef10))

    print(f"{'tercile':7} {'n':>4} {'AUROC mean':>10} {'std':>7} {'EF@10% mean':>12} {'AUROC>0.7':>10}")
    print("-" * 60)
    for terc in ['top', 'mid', 'bottom']:
        vals = per_terc.get(terc)
        if not vals:
            continue
        aurocs = np.array([v[0] for v in vals])
        ef10s = np.array([v[1] for v in vals])
        print(f"{terc:7} {len(vals):>4} {aurocs.mean():>10.3f} {aurocs.std():>7.3f} {ef10s.mean():>12.2f} {(aurocs > 0.7).sum():>10}")

    all_labels = np.array([r['is_active'] for r in rows])
    all_scores = np.array([float(r['score']) for r in rows])
    print()
    print(f"Pooled AUROC: {roc_auc_score(all_labels, all_scores):.4f}")
    print(f"Actives: {all_labels.sum()}, decoys: {len(all_labels) - all_labels.sum()}")


if __name__ == '__main__':
    main()
