# Model Outputs

## Overview

The model's `forward()` returns a 6-tuple:
```python
Y_key, D_key, z_norm, cs, Aff, None
```

---

## 1. Motif Predictions (`cs`)

**Shape**: (N_nodes, 6) where N_nodes = all receptor + grid nodes in the batch.

**Content**: Raw logits (pre-sigmoid) for 6 motif types per node. Sigmoid is applied at loss/inference time.

| Column | Motif type |
|--------|-----------|
| 0 | (unused, column exists but label is always 0) |
| 1 | Both (H-bond donor AND acceptor) |
| 2 | Acceptor |
| 3 | Donor |
| 4 | Aliphatic |
| 5 | Aromatic |

Only predictions at **grid nodes** are used. Grid nodes are identified by `grididx` within the batched receptor graph.

**How produced**: 6 independent `Linear(64, 1, bias=False)` heads in `Grid_SE3.Cblock`, applied to the SE(3) Transformer's output embeddings.

---

## 2. Keyatom Structure Predictions (`Y_key`)

**Shape**: (B, K_max, 3) -- predicted 3D coordinates for each keyatom.

**Content**: Coordinates in the same frame as input (grid-centered). Each keyatom coordinate is a **soft attention-weighted sum** of grid point positions:

```
attention = masked_softmax(linear(z), dim=grid)   -- (B, N_grid, K)
Y_key = einsum('bnk, bn3 -> bk3', attention, grid_xyz)
```

The attention map concentrates around where the model believes each keyatom's fragment should bind. Masked to zero for padding keyatoms (beyond `nK` valid keyatoms).

**Ground truth**: keyatom coordinates from the co-crystal ligand pose, centered by the same origin.

---

## 3. Attention Map (`z_norm`)

**Shape**: (B, N_grid_max, K_max)

**Content**: The softmax-normalized attention weights from the StructModule. This is the same attention used to compute `Y_key`. Used in the attention spread loss to encourage the attention to concentrate near true keyatom positions.

---

## 4. Key Pairwise Distance Predictions (`D_key`)

**Shape**: (B, K_max, K_max, c) where c = embedding dimension (64).

**Content**: Learned pairwise representation between keyatoms, produced by `DistanceModule`:
```
h = LayerNorm(ReLU(Linear(h_key)))   -- per-key projection
D_key = einsum('bic, bjc -> bijc', h, h)  -- outer product
```

Used in the pairwise distance loss as a distogram: the model predicts distance bins between keyatom pairs. The loss compares against ground-truth inter-keyatom distances (binned at 0.25 A resolution, range -0.1 to 15.75 A, 64 bins).

---

## 5. Binding Score (`Aff`)

**Shape**: tuple of `(Aff_scalar, Aff_contrast)`

### 5.1 Scalar Score (`Aff_scalar`)

**Shape**: (B,)

**Content**: Single binding score per ligand. Higher = more likely to be active. Combines:
- **Key-grid interaction score**: max-pooled attention, scaled by learnable `Pcoeff` and `Poff`, averaged over valid keyatoms
- **Global ligand score**: `Linear(4, 1)` on the ligand global embedding, weighted by learnable `Gamma`

Formula:
```
Aff = (aff_key + Gamma * aff_lig) / (1 + Gamma)
```

Used for BCE loss (sigmoid applied at loss time) and pairwise AUC ranking loss.

### 5.2 Contrast Score (`Aff_contrast`)

**Shape**: (B, K_max)

**Content**: Per-keyatom binding signal via dot product with a learned `Affmap` vector:
```
Aff_contrast = einsum('bkd, d -> bk', h_key, Affmap)
```

Used in the screening contrast (margin) loss. Averaged over valid keyatoms to produce per-ligand scores.

---

## 6. Inference Outputs

At inference time, the model produces:

| Output | Used for |
|--------|----------|
| `Aff_scalar` | Virtual screening ranking (active vs decoy) |
| `cs` (grid nodes, after sigmoid) | Motif type prediction visualization |
| `Y_key` | Predicted binding pose (keyatom placement) |
| `z_norm` | Attention visualization (where model "looks") |

The primary screening output is `sigmoid(Aff_scalar)`. The motif and structure outputs provide interpretability: which pharmacophoric features the model detects, and where it predicts fragment placement.
