# Model Architecture

## Overview

MotifScreen-Aff (`EndtoEndModel` in `src/model/models/msk1.py`) is an SE(3)-equivariant structure-based virtual screening model. It processes a receptor graph and ligand graph through independent encoders, fuses them via triangular attention, and produces three outputs: motif predictions, keyatom structure predictions, and a binding score.

---

## 1. Receptor Encoder (Grid_SE3)

**File**: `src/model/modules/featurizers.py:Grid_SE3`

SE(3) Transformer (`src/SE3/se3_transformer/`) operating on the receptor+grid graph.

| Parameter | Default |
|-----------|---------|
| `num_layers_grid` | 5 |
| `n_heads` | 4 |
| `num_channels` | 32 |
| `l0_in_features` | 102 |
| `l0_out_features` | 64 |
| `num_edge_features` | 3 |

**Fiber structure**:
- Input: `{0: 102}` (scalar features only)
- Hidden: `{0: 32, 1: 32, 2: 32}` (scalar + type-1 + type-2 equivariant features)
- Output: `{0: 64}` (scalar only)
- Edge: `{0: 3}`

**Motif head** (Cblock): 6 independent linear layers `Linear(64, 1, bias=False)`, one per motif type (`ntypes=6`). Produces raw logits `cs` of shape (N_nodes, 6).

**Output**:
- `hs0`: (N_nodes, 64) -- per-node embeddings
- `cs`: (N_nodes, 6) -- motif classification logits (raw, sigmoid applied later)

After encoding, the grid portion is extracted: `h_grid = gridmap @ h_rec`, selecting only grid-node embeddings.

---

## 2. Ligand Encoder (Ligand_GAT)

**File**: `src/model/modules/featurizers.py:Ligand_GAT`

Uses `dgl.nn.EGATConv` (Edge-enhanced Graph Attention) layers.

| Parameter | Default |
|-----------|---------|
| `num_layers` | 4 |
| `n_heads` | 4 |
| `num_channels` | 32 |
| `l0_in_features` | 18 |
| `l0_out_features` | 64 |
| `num_edge_features` | 5 |

**Architecture**:
1. `initial_linear`: Linear(18, 32) projects node features
2. `initial_linear_edge`: Linear(5, 32) projects edge features
3. 4x EGATConv layers, each followed by:
   - Multi-head mean pooling (`emb.mean(1)`)
   - InstanceNorm1d
   - ELU activation
4. `final_linear`: Linear(32, 64) output projection

**Output**: (N_lig_atoms, 64) per-atom embeddings.

### 2.1 Ligand Global Embedding (LigandModule)

**File**: `src/model/modules/modules.py:LigandModule`

A small MLP that processes the 19-dim global features:
```
Linear(19, 16) -> LayerNorm(16) -> Linear(16, 4)
```

**Output**: (B, 4) per-molecule global embedding.

---

## 3. Triangular Attention (TrigonModule)

**File**: `src/model/modules/trigon.py:TrigonModule`

Computes pairwise grid-ligand interaction features using triangular multiplicative updates (inspired by AlphaFold2).

### 3.1 Initial Pairwise Features

Distance matrices are binned into one-hot histograms:
- `D_grid`: grid-grid pairwise distances, bins of 0.25 A from -0.1 to 15.75 A (`d` = 64 bins)
- `D_lig`: ligand-ligand pairwise distances, same binning

Initial pairwise features: `z = einsum('bnd, bmd -> bnmd', h_grid, h_lig)` -- outer product of grid and ligand embeddings. Shape: (B, N_grid, M_lig, c).

### 3.2 First Trigon Pass (all ligand atoms)

`trigon_lig` with `n_trigon_lig_layers` = 2 stacks. Each stack contains:
1. **TriangleProteinToCompound**: Gated multiplicative update using distance matrices
   ```
   ab1 = gate1(z).sigmoid() * linear1(z)
   ab2 = gate2(z).sigmoid() * linear2(z)
   block1 = einsum('bikc, bkjc -> bijc', D_grid_proj, ab1)  -- grid-side triangle
   block2 = einsum('bikc, bjkc -> bijc', ab2, D_lig_proj)   -- lig-side triangle
   z += gate(z).sigmoid() * linear(LN(block1 + block2))
   ```
2. **TriangleSelfAttentionRowWise**: Multi-head self-attention along the grid dimension, row by row

3. **Transition**: `LayerNorm -> Linear(c, 4c) -> ReLU -> Linear(4c, c)`

### 3.3 Ligand-to-Key Mapping

After the first trigon pass, ligand atoms are grouped into keyatom representatives:
```
h_key = einsum('bkj, bjd -> bkd', key_idx, h_lig)   -- weighted sum per fragment
```

Optional `lig_to_key_attn` (default True): key embeddings attend back to all ligand atoms:
```
A = einsum('bkd, bjd -> bkj', h_key, h_lig)
A = masked_softmax(A, lig_to_key_mask, dim=2)
h_key = h_key + einsum('bkj, bjd -> bkd', A, h_lig)
h_key = LayerNorm(h_key)
```

The pairwise tensor `z` is reshaped from (B, N, M, c) to (B, N, K, c) via the same key_idx mapping.

### 3.4 Key Trigon Layers

Grid and key embeddings are first projected to matching dimension `c`:
```
h_grid = grid_proj(h_grid)   -- Linear(64, 64)
h_key = key_proj(h_key)      -- Linear(64, 64)
```

Then `n_trigon_key_layers` = 3 rounds of:
1. **DistanceModule**: `D_key = norm(relu(linear(h_key)))` then `D_key = einsum('bic, bjc -> bijc', D_key, D_key)` -- learned pairwise features from key embeddings
2. **XformModule** (key update): Cross-attention where keys attend to grids via z
3. **XformModule** (grid update): Cross-attention where grids attend to keys via z
4. **TrigonModule** (z update): Triangle attention updating z with D_grid and D_key

XformModule:
```
exp_z = exp(z) * mask
z_norm = exp_z / sum(exp_z, dim=target)   -- softmax-like normalization
Qa = linear1(Q)
Va = einsum('bikc, bkc -> bic', z_norm, Qa)  -- attention-weighted aggregation
V = V + linear2(Va)
```

---

## 4. Output Heads

### 4.1 Motif Prediction

Produced by the Grid_SE3 encoder's Cblock (6 parallel `Linear(64, 1)` heads). Applied to ALL receptor+grid nodes, but only grid-node predictions are used for loss.

**Output**: `cs` of shape (N_nodes, 6). Sigmoid is applied at loss time, not in the model.

### 4.2 Structure Prediction (StructModule)

**File**: `src/model/modules/modules.py:StructModule`

Predicts keyatom coordinates as attention-weighted sums of grid coordinates:
```
z = linear(z).squeeze(-1)           -- (B, N, K) attention logits
z = masked_softmax(scale * z, mask) -- normalize over grid dimension
Y_key = einsum('bij, bic -> bjc', z, grid_coords)  -- weighted sum of grid positions
```

**Output**:
- `Y_key`: (B, K, 3) predicted keyatom coordinates
- `z_norm`: (B, N, K) attention map (used for attention spread loss)

### 4.3 Screening / Classification (ClassModule, mode=`former_contrast`)

**File**: `src/model/modules/classification.py:ClassModule`

Two-component score:

1. **Contrast score** (per-keyatom):
   ```
   Aff_contrast = einsum('bkd, d -> bk', h_key, Affmap)  -- learned projection
   ```
   `Affmap` is a learnable (c,)-dim vector. Output: (B, K).

2. **Aggregated binding score**:
   ```
   key_P = einsum('bnkd, bkd -> bnk', z, h_key)  -- z-key attention scores
   key_P = max_pool(key_P, dim=N)                  -- max over grid dim -> (B, 1, K)
   aff_key = Pcoeff * (key_P + Poff)               -- learnable scale + offset
   aff_key = mean(aff_key * mask)                   -- average over valid keyatoms
   aff_lig = linear_lig(h_lig_global)               -- scalar from global features
   Aff = (aff_key + Gamma * aff_lig) / (1 + Gamma) -- weighted combination
   ```

   Learnable parameters: `Pcoeff` (init 5.0), `Poff` (init -1.0), `Gamma` (init 0.1).

**Output**: tuple `(Aff, Aff_contrast)` where:
- `Aff`: (B,) scalar binding score per ligand
- `Aff_contrast`: (B, K) per-keyatom contrast scores

Dropout mode `harsh`: drops out `h_key`, `h_grid`, and `h_lig_global` during training.

---

## 5. Full Forward Pass

```
Input: Grec (receptor+grid graph), Glig (ligand graph), keyidx, grididx

1. h_rec, cs = GridFeaturizer(Grec)           -- SE(3) encoding
   h_grid = extract grid nodes from h_rec

2. h_lig = LigandFeaturizer(Glig)             -- GAT encoding
   h_lig_global = LigandModule(Glig.gdata)    -- global MLP

3. D_grid = pairwise_distance_onehot(grid_xyz)  -- 64-bin histograms
   D_lig = pairwise_distance_onehot(lig_xyz)

4. z = trigon_lig(h_grid, h_lig, D_grid, D_lig)  -- 2-layer triangle attention

5. h_key = key_idx @ h_lig                      -- ligand -> keyatom pooling
   h_key += lig_to_key_attention(h_key, h_lig)  -- optional attention refinement
   z = key_idx @ z                               -- reshape z to key dimension

6. for i in range(3):                            -- 3 rounds of key trigon
       D_key = DistanceModule(h_key)
       h_key = XformKey(h_key, h_grid, z)
       h_grid = XformGrid(h_grid, h_key, z)
       z = trigon_key(h_grid, h_key, D_grid, D_key)

7. Y_key, z_norm = StructModule(z, grid)          -- keyatom coordinate prediction
8. Aff = ClassModule(z, h_grid, h_key, h_lig_global)  -- binding score

Output: (Y_key, D_key, z_norm, cs, Aff, None)
```

---

## 6. Parameter Counts (typical config)

| Component | Approx. params |
|-----------|---------------|
| Grid_SE3 (5-layer) | ~2.5M |
| Ligand_GAT (4-layer) | ~0.3M |
| LigandModule | ~0.4K |
| TrigonModule (lig, 2-stack) | ~0.2M |
| TrigonModule (key, 3x 1-stack) | ~0.3M |
| XformModules (3x2) | ~50K |
| StructModule | ~64 |
| ClassModule | ~0.3K |
| **Total** | **~3.4M** |
