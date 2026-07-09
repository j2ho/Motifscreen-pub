# Loss Functions

Source: `src/model/loss/losses.py`, `scripts/train/train.py`

## Overview

Total loss per batch:

```
L = w_motif * (L_motif_pos + L_motif_neg)
  + w_motif_contrast * L_motif_contrast
  + w_motif_penalty * L_motif_penalty
  + str_warmup * (w_str * L_str + w_str_pair * L_str_pair + w_str_attmap * L_str_attmap)
  + screen_sw * (w_screen_bce * L_screen_bce + w_screen_rank * L_screen_rank + w_screen_contrast * L_screen_contrast)
  + w_penalty * L2_penalty
```

Then scaled by per-source weight: `L = L * source_loss_weight[source]`.

---

## 1. Motif Losses

### 1.1 Masked BCE (`MaskedBCE`)

**Weight**: `w_motif` (default 5.0)

Compares sigmoid(motif_logits) at grid nodes against soft ground-truth labels.

For each sample:
```
Q = sigmoid(logits)[-n_grid:]    -- predictions at grid nodes only
cat = labels                      -- (N_grid, 6) soft labels

# Positive loss (where label > 0):
a = -cat * log(Q + 1e-6)

# Negative loss (where label ~= 0):
b = -(1 - cat) * log(clamp(1 - Q, min=1e-5))
```

The mask restricts loss to grid points that have any motif annotation (mask > 0). Positive and negative losses are **independently normalized** by their element counts:

```
L_pos = sum(mask * a) / n_pos_elements
L_neg = sum(mask * b) / n_neg_elements
```

This fixes two issues: protein-size dependence (raw sum scaled with grid count) and pos/neg class imbalance (~4-5 neg types per 1-2 pos per grid).

### 1.2 Motif Contrast (`MotifContrastLoss`)

**Weight**: `w_motif_contrast` (default 2.0)

Penalizes predictions on **unlabeled** grid points (those with mask = 0):

```
L = mean(sigmoid(logits) on grids where mask == 0)
```

Encourages the model to stay quiet on grids outside any binding interaction zone.

### 1.3 Motif Penalty

**Weight**: `w_motif_penalty` (default 0.0, typically disabled)

Logit norm penalty to prevent saturation:
```
L = ReLU(sum(logits^2) - 25.0)
```

Only activates when total squared logit magnitude exceeds 25.

---

## 2. Structure Losses

All structure losses are conditioned on `eval_struct = True` (requires structure ground truth). For cross-receptor (near-native) samples, structure loss is scaled by `nonnative_struct_weight` (default 0.2).

Structure warmup: all structure losses multiplied by `w_str_warmup_multiplier` during the first `w_str_warmup_epochs` epochs.

### 2.1 Keyatom Coordinate Loss (`StructureLoss`)

**Weight**: `w_str` (default 1.0)

MSE between predicted and ground-truth keyatom coordinates:
```
dY = Y_pred[:k] - Y_gt[:k]       -- (K, 3) displacement vectors
L = mean(dY^2)                     -- MSE over all coordinates
```

Also computes MAE and RMSD for logging (not used in loss).

### 2.2 Pairwise Distance Loss (`PairDistanceLoss`)

**Weight**: `w_str_pair` (default 1.0)

Distogram loss on inter-keyatom distances. Two components:

1. **Cross-entropy** on distance bins:
   ```
   bins: 0.25 A resolution, -0.1 to 15.75 A (64 bins)
   L_ce = CrossEntropy(pred_logits[:k,:k], gt_distance_bins[:k,:k])
   ```

2. **Huber loss** on expected distances:
   ```
   d_pred = einsum('k, ijk -> ij', bin_centers, softmax(pred))
   L_huber = HuberLoss(d_pred, d_gt)
   ```

Total: `L = L_ce + L_huber`

### 2.3 Attention Spread Loss

**Weight**: `w_str_attmap` (default 2.0)

Encourages the StructModule's attention map to concentrate near true keyatom positions. Two components combined:

**SpreadLoss** (reward overlap):
```
overlap = exp(-||grid_xyz - Y_gt||^2 / sigma^2)   -- Gaussian around true positions
L_pos = -sum(overlap * attention_weights)           -- negative = reward
```

**SpreadLoss_v2** (penalize distant attention):
```
dev = ||grid_xyz - Y_gt||^2 / sigma^2
L_neg = sum(dev * attention_weights)
```

Combined: `L_attmap = L_pos + 0.2 * L_neg`. The sigma = 2.0 A.

---

## 3. Screening Losses

Screening losses are scaled by per-source `screen_source_weight` (default 1.0 for all).

### 3.1 BCE Loss

**Weight**: `w_screen_bce` (default 3.0 for e2e, 0.0 for pretrain)

Standard binary cross-entropy with logits on the scalar binding score:
```
L = BCEWithLogitsLoss(Aff_scalar, binary_label)
```

Where `binary_label` = 1 for active, 0 for decoy.

### 3.2 Pairwise AUC Loss (`PairwiseAUCLoss`)

**Weight**: `w_screen_rank` (default 5.0 for e2e, 0.0 for pretrain)

Pairwise ranking loss. For each (active, decoy) pair:
```
diff = score_active - score_decoy
pair_loss = softplus(-diff)          -- = -log(sigmoid(diff)), numerically stable
```

When `rank_alpha = 0` (default): all pairs weighted equally. Optimizes AUROC.

When `rank_alpha > 0`: exponential weighting by decoy rank:
```
ranks = argsort(argsort(decoy_scores, descending=True))
weights = exp(-alpha * ranks / n_decoys)
weights = weights / mean(weights)    -- normalize to mean=1
L = mean(pair_loss * weights)
```

Higher alpha focuses more on top-ranked decoys (alpha=20 approximates BEDROC alpha=20).

Alpha warmup: linearly ramps from 0 to target over `screen_rank_alpha_warmup` epochs.

### 3.3 Screening Contrast Loss (`ScreeningMarginLoss`)

**Weight**: `w_screen_contrast` (default 2.0 for e2e, 0.0 for pretrain)

Margin-based contrastive loss on per-keyatom contrast scores (`Aff_contrast`):

```
avg_score = mean(Aff_contrast[:nK])   -- per-ligand average over valid keyatoms

# Pairwise margin:
pair_loss = ReLU(margin - (active_score - decoy_score))

# Hard negative weighting:
top 1% of decoys get 10x weight (configurable)

L = mean(weighted pair_losses)
```

Default: `margin=1.0`, `top_k_percent=0.2`, `top_weight=5.0`.

---

## 4. Regularization

### 4.1 L2 Penalty

**Weight**: `w_penalty` (default 1e-5)

```
L = sum(||param||_2 for param in model.parameters())
```

Note: this is the L2 norm sum, not the squared norm.

---

## 5. Weight Configurations

### Pretrain (str+motif)

| Loss | Weight |
|------|--------|
| `w_motif` | 5.0 |
| `w_motif_contrast` | 2.0 |
| `w_str` | 1.0 |
| `w_str_pair` | 1.0 |
| `w_str_attmap` | 2.0 |
| `w_screen_*` | 0.0 |
| `w_penalty` | 1e-5 |

### E2E screen only

| Loss | Weight |
|------|--------|
| `w_motif` | 0.0 |
| `w_str` | 0.0 |
| `w_screen_bce` | 3.0 |
| `w_screen_rank` | 5.0 |
| `w_screen_contrast` | 2.0 |
| `w_penalty` | 1e-5 |

### Transfer (full)

| Loss | Weight |
|------|--------|
| `w_motif` | 5.0 |
| `w_motif_contrast` | 2.0 |
| `w_str` | 1.0 |
| `w_str_pair` | 1.0 |
| `w_str_attmap` | 2.0 |
| `w_screen_bce` | 3.0 |
| `w_screen_rank` | 5.0 |
| `w_screen_contrast` | 2.0 |
| `w_penalty` | 1e-5 |

---

## 6. Loss Ablation Flags

Losses can be disabled at config level via two mechanisms:

1. **Weight = 0**: loss is still computed but multiplied by zero (gradient still flows through dependencies).

2. **Ablation flags**: loss computation is skipped entirely:
   - `ablate_motif_loss`, `ablate_motif_contrast_loss`
   - `ablate_structure_loss`, `ablate_str_pair_loss`, `ablate_str_attmap_loss`
   - `ablate_screen_bce_loss`, `ablate_screen_rank_loss`, `ablate_screen_contrast_loss`

The pretrain configs use ablation flags (e.g., `pretrain_stronly.yaml` sets `ablate_motif_loss: true`). The e2e screen-only config uses weight=0 instead.

---

## 7. Per-Source Weighting

Training data comes from three sources with configurable loss weights:
```yaml
source_loss_weight:
  pdbbind: 1.0
  biolip: 1.0
  chembl: 1.0
```

The total loss is multiplied by the source weight. ChEMBL has no structure ground truth, so structure losses are always zero for ChEMBL regardless of weight.

There's also a separate `screen_source_weight` for screening losses specifically (defaults to 1.0 for all sources).
