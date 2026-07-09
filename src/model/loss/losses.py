import torch
import torch.nn.functional as F
from collections import defaultdict


# === Screening losses (scalar) ===

CELoss = torch.nn.CrossEntropyLoss()
BCELoss = torch.nn.BCEWithLogitsLoss()


# === Hard negative memory bank ===

class HardNegativeBank:
    """Per-target memory bank of highest-scoring decoy logits.

    Stores detached decoy scores from recent forward passes. During loss
    computation, these cached scores are mixed with the current batch's
    decoys, giving the model a harder ranking challenge without extra
    GPU memory for graphs.

    Usage:
        bank = HardNegativeBank(capacity=64)

        # After forward pass:
        bank.update(target_name, decoy_logits.detach())

        # In loss:
        hard_scores = bank.get(target_name, n=32, device=device)
        all_decoys = torch.cat([batch_decoys, hard_scores])
    """

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._store = defaultdict(list)

    def update(self, target: str, decoy_logits: torch.Tensor):
        """Add decoy logits (detached) for a target. Keeps top-K by score."""
        scores = decoy_logits.detach().cpu()
        existing = self._store[target]
        combined = torch.cat(existing + [scores]) if existing else scores
        if len(combined) > self.capacity:
            topk = torch.topk(combined, self.capacity).values
            self._store[target] = [topk]
        else:
            self._store[target] = [combined]

    def get(self, target: str, n: int, device: torch.device) -> torch.Tensor:
        """Get up to n hard negative scores for a target."""
        if target not in self._store or not self._store[target]:
            return torch.tensor([], device=device)
        stored = self._store[target][0]
        k = min(n, len(stored))
        # Return the top-k hardest (highest scoring) decoys
        topk = torch.topk(stored, k).values
        return topk.to(device)

    def __len__(self):
        return sum(len(v[0]) for v in self._store.values() if v)

    def clear(self):
        self._store.clear()


def KLLoss(ps, qs):
    """KL divergence between softmax-normalized predictions and labels."""
    eps = 1.0e-6
    ps = torch.nn.functional.softmax(ps, dim=-1)
    qs = torch.nn.functional.softmax(qs, dim=-1)
    ps = torch.clamp(ps, min=eps, max=1.0-eps)
    qs = torch.clamp(qs, min=eps, max=1.0-eps)
    loss = torch.sum(ps * torch.log(ps / qs + eps))
    return loss


def PairwiseAUCLoss(scores, blabel, rank_alpha=0.0):
    """
    Pairwise ranking loss on scalar binding logits.

    When rank_alpha=0 (default): standard BPR/AUC surrogate, all pairs weighted
    equally. Optimizes AUROC.

    When rank_alpha>0: top-weighted variant that penalizes high-scoring decoys
    exponentially more. Approximates a differentiable BEDROC surrogate.
    Higher alpha = more focus on early retrieval (alpha=20 ~ BEDROC_a20).

    Args:
        scores: Raw logits (before sigmoid). Shape: [B] where B = 1 active + N decoys
        blabel: Binary labels. Shape: [B], e.g. [1, 0, 0, ..., 0]
        rank_alpha: Exponential weighting for top decoys. 0 = uniform (AUC), >0 = top-heavy (BEDROC).

    Returns:
        Scalar loss
    """
    active_mask = (blabel == 1)
    decoy_mask = (blabel == 0)

    active_scores = scores[active_mask]   # [N_actives]
    decoy_scores = scores[decoy_mask]     # [N_decoys]

    if active_scores.numel() == 0 or decoy_scores.numel() == 0:
        return scores.sum() * 0.0

    # Pairwise differences: [N_actives, N_decoys]
    diffs = active_scores.unsqueeze(1) - decoy_scores.unsqueeze(0)

    # -log(sigmoid(diff)) = softplus(-diff), numerically stable
    pair_losses = F.softplus(-diffs)

    if rank_alpha > 0:
        # Weight by decoy rank: highest-scoring decoys get exponentially more weight
        n_decoys = decoy_scores.numel()
        _, rank_order = torch.sort(decoy_scores, descending=True)
        ranks = torch.zeros_like(decoy_scores)
        ranks[rank_order] = torch.arange(n_decoys, device=scores.device, dtype=scores.dtype)
        weights = torch.exp(-rank_alpha * ranks / n_decoys)  # [N_decoys]
        weights = weights / weights.sum() * n_decoys  # normalize so mean weight = 1
        weighted_losses = pair_losses * weights.unsqueeze(0)  # [N_actives, N_decoys]
        return weighted_losses.mean()

    return pair_losses.mean()


# === Screening losses (per-key-atom contrast) ===

def ScreeningMarginLoss(embs, blabel, nK, margin=1.0, top_k_percent=0.01, top_weight=10.0):
    """
    Weighted margin-based contrastive loss on per-key-atom scores.
    Optimized for early enrichment (EF@1%, BEDROC) by penalizing
    high-scoring decoys more heavily.

    Args:
        embs: Per-atom scores. Shape: [B, K_max]
        blabel: Ligand labels (0 or 1). Shape: [B]
        nK: Number of key atoms for each ligand. Shape: [B]
        margin: Desired score gap between actives and decoys.
        top_k_percent: Fraction of top decoys to weight heavily (default: 0.01 for 1%)
        top_weight: Weight multiplier for top-k decoys (default: 10.0)
    """
    # Mask for valid key atoms
    k_indices = torch.arange(embs.shape[1], device=embs.device)
    mask = k_indices < nK.unsqueeze(1)

    # Masked average score per ligand
    sum_scores = torch.sum(embs * mask, dim=1)
    avg_ligand_scores = sum_scores / (nK.float() + 1e-6)

    active_mask = (blabel == 1)
    decoy_mask = (blabel == 0)
    active_scores = avg_ligand_scores[active_mask]
    decoy_scores = avg_ligand_scores[decoy_mask]

    if active_scores.numel() == 0 or decoy_scores.numel() == 0:
        return torch.sum(avg_ligand_scores) * 0.0

    # Identify top-k hardest decoys
    _, sorted_indices = torch.sort(decoy_scores, descending=True)
    n_top_decoys = max(1, int(top_k_percent * decoy_scores.numel()))

    decoy_weights = torch.ones_like(decoy_scores)
    decoy_weights[sorted_indices[:n_top_decoys]] = top_weight

    # Pairwise margin loss: [N_actives, N_decoys]
    pair_losses = F.relu(margin - (active_scores.unsqueeze(1) - decoy_scores.unsqueeze(0)))

    # Apply hard-negative weights
    weighted_losses = pair_losses * decoy_weights.unsqueeze(0)

    return torch.mean(weighted_losses)


# === Motif prediction losses ===

def MaskedBCE(cats, preds, masks, debug=False):
    device = masks.device

    lossP = torch.tensor(0.0).to(device)
    lossN = torch.tensor(0.0).to(device)

    for cat, mask, pred in zip(cats, masks, preds):
        # cat: NxT, mask: N (count of positive motifs per grid), pred: NxT
        ngrid = cat.shape[0]
        Q = pred[-ngrid:]

        a = -cat * torch.log(Q + 1.0e-6)               # -PlogQ
        icat = (cat < 0.001).float()
        iQt = torch.clamp(1.0 - Q, min=1.0e-5)
        b = -icat * torch.log(iQt)                      # penalize if high

        mask_binary = (mask > 0).float()
        mask_2d = mask_binary.unsqueeze(1)

        # Normalize independently by element count to fix:
        # 1) protein-size dependence (raw sum scaled with grid count)
        # 2) pos/neg class imbalance (~4-5 neg types per 1-2 pos per grid)
        n_pos = (mask_2d * (cat > 0.001).float()).sum().clamp(min=1.0)
        n_neg = (mask_2d * icat).sum().clamp(min=1.0)

        lossP += torch.sum(mask_2d * a) / n_pos
        lossN += torch.sum(mask_2d * b) / n_neg

        if debug:
            print("\n" + "="*60)
            print("DEBUG: MaskedBCE")
            print("="*60)
            print(f"  cats shape: {cat.shape}, pred shape: {pred.shape}")
            print(f"  Original mask (raw counts) - unique values: {torch.unique(mask).tolist()}")
            print(f"  Original mask stats: min={mask.min():.0f}, max={mask.max():.0f}, mean={mask.float().mean():.2f}")
            print(f"  Binary mask (after >0) - unique values: {torch.unique(mask_binary).tolist()}")
            print(f"  Grids with labels: {mask_binary.sum().int().item()} / {ngrid}")
            print(f"  n_pos elements: {n_pos.int().item()}, n_neg elements: {n_neg.int().item()}")
            print(f"  Sample mask values (first 20): {mask[:20].tolist()}")
            print(f"  Sample binary mask (first 20): {mask_binary[:20].tolist()}")
            print(f"\n  Q (pred) stats: min={Q.min():.4f}, max={Q.max():.4f}, mean={Q.mean():.4f}")
            print(f"  cat (GT) stats: min={cat.min():.4f}, max={cat.max():.4f}, mean={cat.mean():.4f}")
            print(f"\n  lossP: {lossP.item():.6f}, lossN: {lossN.item():.6f}")
            print("="*60 + "\n")

    return lossP, lossN


def MotifContrastLoss(preds, masks, debug=False):
    """Penalize predictions on unlabeled grids."""
    loss = torch.tensor(0.0).to(masks.device)

    for mask, pred in zip(masks, preds):
        imask = (mask == 0).float()
        ngrid = mask.shape[0]
        imask_2d = imask.unsqueeze(1)
        masked_pred = imask_2d * pred[-ngrid:]
        num_unmasked = torch.sum(imask)
        if num_unmasked >= 1.0:
            psum = torch.sum(masked_pred) / num_unmasked
        else:
            psum = torch.tensor(0.0, device=masks.device)

        loss += psum

        if debug:
            print("\n" + "="*60)
            print("DEBUG: ContrastLoss (penalize preds on unlabeled grids)")
            print("="*60)
            print(f"  Original mask (raw counts) - unique values: {torch.unique(mask).tolist()}")
            print(f"  Original mask stats: min={mask.min():.0f}, max={mask.max():.0f}")
            print(f"  Inverse binary mask (==0) - unique values: {torch.unique(imask).tolist()}")
            print(f"  Grids WITHOUT labels (for contrast): {imask.sum().int().item()} / {ngrid}")
            print(f"  Sample mask values (first 20): {mask[:20].tolist()}")
            print(f"  Sample imask values (first 20): {imask[:20].tolist()}")
            print(f"\n  Pred on unlabeled grids - mean: {(torch.sum(masked_pred)/num_unmasked if num_unmasked >= 1 else 0):.4f}")
            print(f"  ContrastLoss (avg pred on unlabeled): {psum.item():.6f}")
            print("="*60 + "\n")

    return loss


# === Structure losses ===

def StructureLoss(Yrec, Ylig, nK, opt='mse', debug=False):
    # Yrec: [B,Kmax,3] Ylig: [1,K,3]
    k = nK[0].item()
    dY = Yrec[0, :k, :] - Ylig[0, :k, :]

    if debug:
        print("\n" + "="*60)
        print("DEBUG: StructureLoss")
        print("="*60)
        print(f"  nK (num key atoms): {k}")
        print(f"  Yrec (predicted) shape: {Yrec.shape}")
        print(f"  Ylig (ground truth) shape: {Ylig.shape}")
        print(f"\n  Predicted key atom coords (first {min(k,5)} atoms):")
        for i in range(min(k, 5)):
            print(f"    atom {i}: [{Yrec[0,i,0]:.3f}, {Yrec[0,i,1]:.3f}, {Yrec[0,i,2]:.3f}]")
        print(f"\n  Ground truth key atom coords (first {min(k,5)} atoms):")
        for i in range(min(k, 5)):
            print(f"    atom {i}: [{Ylig[0,i,0]:.3f}, {Ylig[0,i,1]:.3f}, {Ylig[0,i,2]:.3f}]")
        print(f"\n  Per-atom displacement dY (first {min(k,5)} atoms):")
        for i in range(min(k, 5)):
            print(f"    atom {i}: [{dY[i,0]:.3f}, {dY[i,1]:.3f}, {dY[i,2]:.3f}]")

    if opt == 'mse':
        loss1_sum = torch.mean(dY ** 2)
    elif opt == 'Huber':
        d = torch.sqrt(torch.sum(dY*dY, dim=-1)) / nK[0]
        loss1_sum = 10.0 * torch.nn.functional.huber_loss(d, torch.zeros_like(d))

    mae = torch.sum(torch.abs(dY)) / k
    sqrd_dist = torch.sum(dY**2, dim=-1)
    rmsd = torch.sqrt(torch.mean(sqrd_dist))

    if debug:
        per_atom_dist = torch.sqrt(sqrd_dist)
        print(f"\n  Per-atom distances (first {min(k,5)} atoms):")
        for i in range(min(k, 5)):
            print(f"    atom {i}: {per_atom_dist[i]:.3f} Å")
        print(f"\n  Loss: {loss1_sum.item():.6f}, MAE: {mae.item():.3f} Å, RMSD: {rmsd.item():.3f} Å")
        print("="*60 + "\n")

    return loss1_sum, mae, rmsd


def PairDistanceLoss(Dpred, X, nK, bin_min=-0.1, bin_size=0.25, bin_max=15.75, debug=False):
    """Distogram loss: CE on distance bins + Huber on expected distance."""
    pair_dis = torch.cdist(X, X, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis[pair_dis > bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    LossFunc1 = torch.nn.CrossEntropyLoss()
    LossFunc2 = torch.nn.HuberLoss()

    if debug:
        print("\n" + "="*60)
        print("DEBUG: PairDistanceLoss")
        print("="*60)
        print(f"  Dpred (distogram logits) shape: {Dpred.shape}")
        print(f"  X (ground truth coords) shape: {X.shape}")
        print(f"  nK: {nK}")
        print(f"  Bin params: min={bin_min}, size={bin_size}, max={bin_max}")

    loss = torch.tensor(0.0).to(X.device)
    for batch_idx, (pred, label, _, d_l, k) in enumerate(zip(Dpred, pair_dis_bin_index, Dpred, pair_dis, nK)):
        dbins = torch.arange(0.0, 15.9, 0.25).to(X.device)
        pred_prob = torch.softmax(pred[:k, :k], dim=-1)
        d_p = torch.einsum('k,ijk->ij', dbins, pred_prob)
        pred_logits = pred[:k, :k, :].transpose(1, 2)
        label = label[:k, :k]
        loss1 = LossFunc1(pred_logits, label)
        loss2 = LossFunc2(d_p, d_l[:k, :k])
        loss = loss + loss1 + loss2

        if debug:
            print(f"\n  Batch {batch_idx}, k={k}:")
            print(f"    pred_logits shape: {pred_logits.shape} (K x num_bins x K)")
            print(f"    pred_prob shape: {pred_prob.shape} (K x K x num_bins)")
            print(f"    label (bin indices) shape: {label.shape}")
            print(f"\n    Sample pairwise distances (first 3x3 pairs):")
            print(f"    {'Pair':<10} {'GT dist':<10} {'GT bin':<10} {'Pred dist':<12} {'Pred top bin':<12}")
            print(f"    {'-'*54}")
            for i in range(min(3, k)):
                for j in range(min(3, k)):
                    gt_dist = d_l[i, j].item()
                    gt_bin = label[i, j].item()
                    pred_dist = d_p[i, j].item()
                    pred_top_bin = torch.argmax(pred_prob[i, j]).item()
                    print(f"    ({i},{j})     {gt_dist:<10.2f} {gt_bin:<10} {pred_dist:<12.2f} {pred_top_bin:<12}")
            print(f"\n    Logit stats: min={pred_logits.min():.3f}, max={pred_logits.max():.3f}, mean={pred_logits.mean():.3f}")
            print(f"    CE Loss: {loss1.item():.6f}, Huber Loss: {loss2.item():.6f}")

    if debug:
        print(f"\n  Total PairDistanceLoss: {loss.item():.6f}")
        print("="*60 + "\n")

    return loss


# === Attention spread losses ===

def SpreadLoss(Ylig, A, grid, nK, sig=2.0):
    """Encourage attention to overlap with key atom positions (reward)."""
    # Ylig: B x K x 3, A: B x Nmax x K, grid: B x Nmax x 3
    loss2 = torch.tensor(0.0)

    for b, (x, k) in enumerate(zip(grid, nK)):
        n = x.shape[0]
        z = A[0, :n, :k]
        x = x[:, None, :]

        dX = x - Ylig
        overlap = torch.exp(-torch.sum(dX*dX, axis=-1) / (sig*sig))
        if z.shape[0] != overlap.shape[0]:
            continue

        loss2 = loss2 - torch.sum(overlap * z)

    return loss2


def SpreadLoss_v2(Ylig, A, grid, nK, sig=2.0):
    """Penalize attention far from key atom positions (penalty)."""
    # Ylig: B x K x 3, A: B x Nmax x K, grid: B x Nmax x 3
    loss2 = torch.tensor(0.0)

    for b, (x, k) in enumerate(zip(grid, nK)):
        n = x.shape[0]
        z = A[0, :n, :k]
        x = x[:, None, :]

        dX = (x - Ylig) / sig
        dev = torch.sum(dX*dX, axis=-1)

        if z.shape[0] != dev.shape[0]:
            continue

        loss2 = loss2 + torch.sum(dev * z)

    return loss2
