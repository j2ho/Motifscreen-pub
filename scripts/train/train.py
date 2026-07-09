# train.py
#!/usr/bin/env python

import os
import sys
import numpy as np
from os.path import join, isdir
from collections import defaultdict
import torch
import time
import dgl
import argparse
import contextlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# DDP related modules
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.dataset_jiho import TrainingDataSet, collate
from src.model.models.msk1 import EndtoEndModel as MSK_1
from src.model.models.msk_ab import EndtoEndModel as MSK_ablation

from scripts.train.utils import count_parameters, to_cuda, calc_AUC, calc_enrichment_factor, EMA, MetricsLogger
import src.model.loss.losses as Loss
from configs.config_loader import load_config, load_config_with_base, Config #

import warnings
warnings.filterwarnings("ignore", message="sourceTensor.clone")

from datetime import datetime
import wandb

def load_params(rank, config: Config):
    """Load model, optimizer, and training state"""
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if config.version == "v1.0":
        if not config.training.silent:
            print("Loading MSK_1 model")
        model = MSK_1(config)
    elif config.version == "ablation":
        if not config.training.silent:
            print("Loading MSK_ablation model")
        model = MSK_ablation(config)
    model.to(device)

    train_loss_empty = {
        "total": [], "MotifP": [], "MotifN": [], "MotifCont": [], "NormPenalty": [],
        "Str": [], "StrMAE": [], "StrPair": [], "StrRMSD": [], "KeyatmAttmap": [],
        "Screen": [], "ScreenC": [], "ScreenR": []
        }
    valid_loss_empty = {
        "total": [], "MotifP": [], "MotifN": [], "MotifCont": [], "NormPenalty": [],
        "Str": [], "StrMAE": [], "StrPair": [], "StrRMSD": [], "KeyatmAttmap": [],
        "Screen": [], "ScreenC": [], "ScreenR": []
        }
    epoch = 0

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )

    # Initialize EMA with default decay of 0.9999
    ema_decay = getattr(config.training, 'ema_decay', 0.9999) if hasattr(config.training, 'ema_decay') else 0.9999
    use_ema = getattr(config.training, 'use_ema', False) if hasattr(config.training, 'use_ema') else False
    ema = EMA(model, decay=ema_decay, device=device) if use_ema else None

    scheduler = None
    if config.training.scheduler.use_scheduler:
        if config.training.scheduler.scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.training.scheduler.factor,
                patience=config.training.scheduler.patience,
                verbose=True,
                min_lr=config.training.scheduler.min_lr,
                threshold=config.training.scheduler.threshold
            )
        elif config.training.scheduler.scheduler_type == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.training.scheduler.step_size,
                gamma=config.training.scheduler.gamma
            )
        elif config.training.scheduler.scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training.scheduler.T_max
            )
        else:
            print(f"Unknown scheduler type: {config.training.scheduler.scheduler_type}. No scheduler will be used.")
    else:
        print("Scheduler disabled in config.")

    # Checkpoint path uses model_note (matching save path)
    checkpoint_path = join("models", f"{config.model_note}", "model.pkl")
    # Allow custom checkpoint path via --chkpt_name
    if hasattr(config, 'chkpt_name') and config.chkpt_name:
        checkpoint_path = config.chkpt_name
    # Access load_checkpoint via config.training.load_checkpoint
    if os.path.exists(checkpoint_path) and config.training.load_checkpoint:
        if not config.training.silent:
            print("Loading a checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        trained_dict = {}
        model_dict = model.state_dict()
        model_keys = list(model_dict.keys())

        for key in checkpoint["model_state_dict"]:
            if key in model_keys:
                wts = checkpoint["model_state_dict"][key]
                if wts.shape == model_dict[key].shape:
                    trained_dict[key] = wts
                else:
                    print("skip", key)

        nnew, nexist = 0, 0
        for key in model_keys:
            if key not in trained_dict:
                nnew += 1
                print("new", key)
            else:
                nexist += 1

        model.load_state_dict(trained_dict, strict=False)

        transfer = getattr(config, 'transfer', False)
        if transfer:
            # Transfer learning: keep model weights but reset everything else
            if not config.training.silent:
                print(f"Transfer learning from checkpoint (epoch {checkpoint['epoch']}): resetting epoch, optimizer, and loss history")
            train_loss = train_loss_empty
            valid_loss = valid_loss_empty
        else:
            epoch = checkpoint["epoch"] + 1
            train_loss = checkpoint["train_loss"]
            valid_loss = checkpoint["valid_loss"]

            # Load optimizer state if available
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if not config.training.silent:
                    print("Loaded optimizer state from checkpoint")

            # Load EMA state if available and EMA is enabled
            if ema is not None and "ema_state_dict" in checkpoint:
                ema.load_state_dict(checkpoint["ema_state_dict"])
                if not config.training.silent:
                    print("Loaded EMA state from checkpoint")

            for key in train_loss_empty:
                if key not in train_loss:
                    train_loss[key] = []
            for key in valid_loss_empty:
                if key not in valid_loss:
                    valid_loss[key] = []

            if not config.training.silent:
                print("Restarting at epoch", epoch)

    else:
        if not config.training.silent:
            print("Training a new model")
        train_loss = train_loss_empty
        valid_loss = valid_loss_empty

    # Always ensure model directory exists (needed when loading from custom checkpoint path)
    model_dir = join("models", f"{config.model_note}")
    if not isdir(model_dir):
        if not config.training.silent:
            print("Creating a new dir at", model_dir)
        os.makedirs(model_dir, exist_ok=True)

    if epoch == 0:
        for i, (name, layer) in enumerate(model.named_modules()):
            if isinstance(layer, torch.nn.Linear) and \
               ("class" in name or 'Xform' in name):
                layer.weight.data *= 0.1

    if rank == 0:
        print("Nparams:", count_parameters(model))
        print("Loaded")

    return model, optimizer, scheduler, epoch, train_loss, valid_loss, ema


def load_data(txt_file, world_size, rank, main_config: Config, static: bool):
    """Load dataset using grouped configuration"""
    from torch.utils import data

    # Parse training data file
    targets = []
    ligands = []
    weights = {}

    print(f"Loading training data from: {txt_file}")
    with open(txt_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split()

            target = parts[0]
            if not static: # Apply ablation only for training data
                if main_config.ablation.ablate_pdbbind:
                    if 'pdbbind' in target.lower():
                        continue
                if main_config.ablation.ablate_biolip:
                    if 'biolip' in target.lower():
                        continue
                if main_config.ablation.ablate_chembl:
                    if 'chembl' in target.lower():
                        continue
            active_ligand = parts[1]
            mol2_file_type = parts[2]
            # Optional weight
            if len(parts) > 3:
                weights[target.split('/')[-1]] = float(parts[3])
            else:
                weights[target.split('/')[-1]] = 1.0

            targets.append(target)
            ligands.append((active_ligand, mol2_file_type))

    print(f"Loaded {len(targets)} samples from {txt_file}")

    # Pass the main config object directly
    dataset = TrainingDataSet(
        targets=targets, # 'targets' and 'ligands' need to be defined outside this snippet's scope
        ligands=ligands, # Same for 'ligands'
        config=main_config, # Pass the entire main config object
        static=static
    )

    # Dataloader parameters now come from main_config.dataloader
    dataloader_params = {
        'shuffle': main_config.dataloader.shuffle,
        'num_workers': main_config.dataloader.num_workers,
        'pin_memory': main_config.dataloader.pin_memory,
        'collate_fn': collate,
        'batch_size': main_config.dataloader.batch_size
    }

    # DDP logic uses main_config.training.ddp
    if main_config.training.ddp:
        sampler = data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        # shuffle is controlled by DistributedSampler in DDP mode
        dataloader_params_ddp = dict(dataloader_params)
        dataloader_params_ddp['shuffle'] = False
        dataloader = data.DataLoader(dataset, sampler=sampler, **dataloader_params_ddp)
    else:
        dataloader = data.DataLoader(dataset, **dataloader_params)

    return dataloader, weights # 'weights' needs to be defined outside this snippet's scope


def train_one_epoch(model, optimizer, loader, rank, epoch, is_train, config: Config, weights, global_step=0, ema=None, metrics_logger=None, scaler=None, hard_neg_bank=None):
    """Train for one epoch"""
    temp_loss = {
        "total": [], "MotifP": [], "MotifN": [], "MotifCont": [], "NormPenalty": [],
        "Str": [], "StrMAE": [], "StrRMSD": [], "StrPair": [], "KeyatmAttmap": [],
        "Screen": [], "ScreenC": [], "ScreenR": []
        }

    b_count, e_count = 0, 0
    valid_micro_count = 0  # Track valid backward passes in current accumulation window
    # Access accumulation_steps from config.training
    accum = config.training.accumulation_steps
    if config.training.debug: # Access debug from config.training
        accum = 1
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    Pt = {'chembl': [], 'biolip': [], 'pdbbind': [], 'dude': []}
    Pf = {'chembl': [], 'biolip': [], 'pdbbind': [], 'dude': []}
    Pt_per_target = {'chembl': {}, 'biolip': {}, 'pdbbind': {}, 'dude': {}}
    Pf_per_target = {'chembl': {}, 'biolip': {}, 'pdbbind': {}, 'dude': {}}

    # Track motif label availability for diagnostic logging
    motif_label_stats = {'with_labels': 0, 'without_labels': 0, 'by_source': {}}

    # Curriculum hard negatives: collect (compound_id, score) for decoys
    decoy_scores_by_target = defaultdict(list)

    for i, inputs in enumerate(loader):
        skip_batch = False

        if inputs is None:
            e_count += 1
            skip_batch = True

        if not skip_batch:
            (Grec, Glig, cats, masks, keyxyz, keyidx, blabel, info) = inputs
            grididx = info['grididx']
            if any(x is None for x in (Grec, Glig, keyidx, grididx)):
                e_count += 1
                skip_batch = True

        # DDP consensus: if ANY rank has bad data, ALL ranks skip this batch entirely.
        # Must happen BEFORE forward so no rank registers DDP gradient hooks.
        if is_train and config.training.ddp and dist.is_initialized():
            skip_vote = torch.tensor(1.0 if skip_batch else 0.0, device=device)
            dist.all_reduce(skip_vote, op=dist.ReduceOp.MAX)
            if skip_vote.item() > 0:
                skip_batch = True

        if skip_batch:
            b_count += 1
            continue

        is_last_loader_iter = (i + 1) == len(loader)

        with torch.cuda.amp.autocast(config.training.amp):
            should_sync = (not is_train) or ((b_count + 1) % accum == 0) or is_last_loader_iter
            sync_context = contextlib.nullcontext() if should_sync else (model.no_sync() if hasattr(model, 'no_sync') else contextlib.nullcontext())
            with sync_context:
                # Forward pass (data-loading skips handled above via consensus + continue)
                t0 = time.time()

                Glig = to_cuda(Glig, device)
                keyxyz = to_cuda(keyxyz, device)
                keyidx = to_cuda(keyidx, device)
                nK = info['nK'].to(device)
                blabel = to_cuda(blabel, device)

                Grec = to_cuda(Grec, device)
                pnames = info["pname"]
                source = info['source'][0]
                grid = info['grid'].to(device)
                eval_struct = info['eval_struct'][0]
                is_near_native = info.get('is_near_native', [False])[0]
                grididx = grididx.to(device)

                t1 = time.time()
                keyxyz_pred, key_pairdist_pred, rec_key_z, motif_pred, bind_pred, absaff_pred = model(
                    Grec, Glig, keyidx, grididx,
                    gradient_checkpoint=(is_train and config.training.gradient_checkpoint),
                    drop_out=is_train
                )

                # Post-forward failures: use zero_loss instead of skip_batch
                # so DDP backward hooks (registered during forward) still fire.
                zero_loss = False
                if motif_pred is None:
                    zero_loss = True

                # Early NaN detection after model forward pass
                nan_detected = False
                if keyxyz_pred is not None and torch.isnan(keyxyz_pred).any():
                    print(f"NaN detected in keyxyz_pred at epoch {epoch}, batch {b_count}")
                    nan_detected = True
                if key_pairdist_pred is not None and torch.isnan(key_pairdist_pred).any():
                    print(f"NaN detected in key_pairdist_pred at epoch {epoch}, batch {b_count}")
                    nan_detected = True
                if rec_key_z is not None and torch.isnan(rec_key_z).any():
                    print(f"NaN detected in rec_key_z at epoch {epoch}, batch {b_count}")
                    nan_detected = True
                if motif_pred is not None and torch.isnan(motif_pred).any():
                    print(f"NaN detected in motif_pred at epoch {epoch}, batch {b_count}")
                    nan_detected = True
                if bind_pred is not None:
                    for i, bp in enumerate(bind_pred):
                        if bp is not None and torch.isnan(bp).any():
                            print(f"NaN detected in bind_pred[{i}] at epoch {epoch}, batch {b_count}")
                            nan_detected = True

                if nan_detected:
                    print(f"Zeroing loss for batch {b_count} due to NaN in model outputs")
                    zero_loss = True

                l_motif_pos = torch.tensor(0.0, device=device)
                l_motif_neg = torch.tensor(0.0, device=device)
                l_motif_contrast = torch.tensor(0.0, device=device)
                motif_penalty = torch.tensor(0.0, device=device)
                has_motif_label = (not zero_loss) and cats is not None and eval_struct

                if has_motif_label:
                    cats = to_cuda(cats, device)
                    masks = to_cuda(masks, device)

                    # Debug output only on first few batches in debug mode
                    debug_motif = config.training.debug and b_count < 3

                    # Motif penalty on raw logits (before sigmoid) to prevent saturation
                    motif_penalty = torch.nn.functional.relu(torch.sum(motif_pred * motif_pred - 25.0))
                    motif_pred = torch.sigmoid(motif_pred)
                    motif_preds = [motif_pred]
                    l_motif_pos, l_motif_neg = Loss.MaskedBCE(cats, motif_preds, masks, debug=debug_motif)
                    l_motif_contrast = Loss.MotifContrastLoss(motif_preds, masks, debug=debug_motif)

                    # Debug logging for motif losses - uncomment to debug
                    if False and b_count % 50 == 0 and rank == 0 and not config.training.debug:
                        print(f"[Epoch {epoch}, Batch {b_count}] Motif loss DEBUG:")
                        print(f"  cats shape: {cats.shape}, min: {cats.min():.4f}, max: {cats.max():.4f}")
                        print(f"  masks shape: {masks.shape}, sum: {masks.sum():.1f}")
                        print(f"  motif_pred shape: {motif_pred.shape}, min: {motif_pred.min():.4f}, max: {motif_pred.max():.4f}")
                        print(f"  l_motif_pos: {l_motif_pos:.6f}, l_motif_neg: {l_motif_neg:.6f}")
                        print(f"  l_motif_contrast: {l_motif_contrast:.6f}, motif_penalty: {motif_penalty:.6f}")
                        print(f"  pname: {pnames[0]}, source: {source}")

                Pbind = []
                l_str_dist = torch.tensor(0.0, device=device)
                key_mae = torch.tensor(0.0, device=device)
                key_rmsd = torch.tensor(0.0, device=device)
                l_str_pair = torch.tensor(0.0, device=device)
                l_str_attmap = torch.tensor(0.0, device=device)
                l_screen = torch.tensor(0.0, device=device)
                l_screen_cont = torch.tensor(0.0, device=device)
                l_screen_rank = torch.tensor(0.0, device=device)

                if not zero_loss and keyxyz_pred is not None and grid.shape[1] == rec_key_z.shape[1]:
                    try:
                        if len(nK.shape) > 1:
                            nK = nK.squeeze(dim=0)

                        if eval_struct:
                            # Debug output only on first few batches in debug mode
                            debug_loss = config.training.debug and b_count < 3

                            # Access struct_loss from config.losses
                            l_str_dist, key_mae, key_rmsd = Loss.StructureLoss(keyxyz_pred, keyxyz, nK, opt=config.losses.struct_loss, debug=debug_loss)
                            l_str_pair = Loss.PairDistanceLoss(key_pairdist_pred, keyxyz, nK, debug=debug_loss)

                            l_str_attmap_pos = Loss.SpreadLoss(keyxyz, rec_key_z, grid, nK)
                            l_str_attmap_neg = Loss.SpreadLoss_v2(keyxyz, rec_key_z, grid, nK)
                            l_str_attmap = l_str_attmap_pos + 0.2 * l_str_attmap_neg

                            # Apply reduced weight for near-native structure loss
                            if is_near_native:
                                struct_weight = config.cross_validation.nonnative_struct_weight
                                l_str_dist = l_str_dist * struct_weight
                                l_str_pair = l_str_pair * struct_weight
                                l_str_attmap = l_str_attmap * struct_weight

                    except Exception as e:
                        print(f"Error in str loss calculation: {e}")
                        import traceback
                        traceback.print_exc()
                        pass
                if not zero_loss and bind_pred is not None:
                    try:
                        # Access screening loss weights from config.losses
                        l_screen = Loss.BCELoss(bind_pred[0], blabel)

                        # Ranking loss: pairwise_auc (default), ce, or kl
                        rank_type = getattr(config.losses, 'screen_rank_type', 'pairwise_auc')
                        rank_alpha_target = getattr(config.losses, 'screen_rank_alpha', 0.0)
                        rank_alpha_warmup = getattr(config.losses, 'screen_rank_alpha_warmup', 0)
                        if rank_alpha_warmup > 0 and epoch < rank_alpha_warmup:
                            rank_alpha = rank_alpha_target * epoch / rank_alpha_warmup
                        else:
                            rank_alpha = rank_alpha_target

                        # Augment decoy scores with hard negatives from memory bank
                        rank_scores = bind_pred[0]
                        rank_labels = blabel
                        hn_capacity = getattr(config.losses, 'hard_neg_capacity', 0)
                        if hard_neg_bank is not None and is_train and hn_capacity > 0:
                            target_key = f"{source}.{pnames[0]}"
                            # Always update bank with current decoy scores
                            decoy_mask_cur = (blabel == 0)
                            if decoy_mask_cur.any():
                                hard_neg_bank.update(target_key, bind_pred[0][decoy_mask_cur])
                            # Retrieve hard negatives (skip epoch 0: bank has no meaningful scores yet)
                            if epoch > 0:
                                n_hard = getattr(config.losses, 'hard_neg_per_batch', 32)
                                hard_scores = hard_neg_bank.get(target_key, n_hard, device)
                                if hard_scores.numel() > 0:
                                    rank_scores = torch.cat([rank_scores, hard_scores])
                                    rank_labels = torch.cat([rank_labels, torch.zeros(hard_scores.numel(), device=device)])

                        if rank_type == 'pairwise_auc':
                            l_screen_rank = Loss.PairwiseAUCLoss(rank_scores, rank_labels, rank_alpha=rank_alpha)
                        elif rank_type == 'ce':
                            logits = bind_pred[0].unsqueeze(0)
                            target_index = torch.tensor([0], device=logits.device)
                            l_screen_rank = Loss.CELoss(logits, target_index)
                        elif rank_type == 'kl':
                            l_screen_rank = Loss.KLLoss(bind_pred[0], blabel)

                        cont_top_k = getattr(config.losses, 'screen_cont_top_k', 0.2)
                        cont_top_w = getattr(config.losses, 'screen_cont_top_weight', 5.0)
                        cont_margin = getattr(config.losses, 'screen_cont_margin', 1.0)
                        l_screen_cont = Loss.ScreeningMarginLoss(bind_pred[1], blabel, nK, margin=cont_margin,
                                                                                  top_k_percent=cont_top_k, top_weight=cont_top_w)
                        # print (f"Screening losses - bce: {l_screen.item():.6f}, rank: {l_screen_rank.item():.6f}, cont: {l_screen_cont.item():.6f}")
                        bind_probs = torch.sigmoid(bind_pred[0])
                        # print ('bind_probs', bind_probs.cpu().detach().numpy())
                        Pbind = ['%4.2f' % float(p) for p in bind_probs]
                        Pt[source].append(float(bind_probs[0].cpu()))
                        Pf[source] += list(bind_probs[1:].cpu().detach().numpy())

                        blabel = blabel.cpu().detach().numpy()
                        bind_scores = bind_probs.cpu().detach().numpy()
                        # Initialize lists for this target if they don't exist
                        pname = pnames[0]
                        if pname not in Pt_per_target[source]:
                            Pt_per_target[source][pname] = []
                            Pf_per_target[source][pname] = []
                        # Iterate through the scores and labels to separate positives and negatives
                        ligand_ids = info['ligands'][0] if 'ligands' in info else []
                        for idx, (score, label) in enumerate(zip(bind_scores, blabel)):
                            if label == 1:
                                Pt_per_target[source][pname].append(score)
                            else:
                                Pf_per_target[source][pname].append(score)
                                if is_train and idx < len(ligand_ids):
                                    target_key = f"{source}.{pname}"
                                    decoy_scores_by_target[target_key].append((ligand_ids[idx], float(score)))

                    except Exception as e:
                        print(f"Error in binding loss calculation: {e}")
                        import traceback
                        traceback.print_exc()
                        pass

                l2_penalty = torch.tensor(0.0, device=device)
                if not zero_loss and is_train:
                    for param in model.parameters():
                        l2_penalty += torch.norm(param)

                if not zero_loss:
                    # Structure warmup: scale all structure weights during early epochs
                    str_warmup_mult = config.losses.w_str_warmup_multiplier if epoch < config.losses.w_str_warmup_epochs else 1.0

                    # Per-source screening weight (default 1.0 for all sources)
                    screen_source_weights = getattr(config.losses, 'screen_source_weight', None) or {}
                    screen_sw = screen_source_weights.get(source, 1.0)

                    # Flat loss assembly — each config weight is the effective weight
                    loss = (# Motif
                            config.losses.w_motif * (l_motif_pos + l_motif_neg) +
                            config.losses.w_motif_contrast * l_motif_contrast +
                            config.losses.w_motif_penalty * motif_penalty +
                            # Structure (with optional warmup multiplier)
                            str_warmup_mult * (
                                config.losses.w_str * l_str_dist +
                                config.losses.w_str_pair * l_str_pair +
                                config.losses.w_str_attmap * l_str_attmap) +
                            # Screening (scaled by source weight)
                            screen_sw * (
                                config.losses.w_screen_bce * l_screen +
                                config.losses.w_screen_rank * l_screen_rank +
                                config.losses.w_screen_contrast * l_screen_cont) +
                            # Regularization
                            config.losses.w_penalty * l2_penalty)

                    source_weights = getattr(config.losses, 'source_loss_weight', None) or {}
                    trg_weight = source_weights.get(source, 1.0)
                    loss = loss * trg_weight

                    # Check for NaN/Inf in total loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"NaN/Inf detected in total loss at epoch {epoch}, batch {b_count}")
                        print(f"  Individual losses: motif_pos={l_motif_pos:.6f}, motif_neg={l_motif_neg:.6f}")
                        print(f"  motif_contrast={l_motif_contrast:.6f}, str_dist={l_str_dist:.6f}")
                        print(f"  screen={l_screen:.6f}, screen_rank={l_screen_rank:.6f}")
                        print(f"  motif_penalty={motif_penalty:.6f}, l2_penalty={l2_penalty:.6f}")
                        print(f"  target_weight={trg_weight:.6f}")
                        zero_loss = True

                # Post-forward DDP consensus: if ANY rank must zero this microbatch
                # (NaN/missing outputs), ALL ranks must use zero loss to keep backward
                # communication paths identical across ranks.
                if is_train and config.training.ddp and dist.is_initialized():
                    zero_vote = torch.tensor(1.0 if zero_loss else 0.0, device=device)
                    dist.all_reduce(zero_vote, op=dist.ReduceOp.MAX)
                    if zero_vote.item() > 0 and not zero_loss:
                        zero_loss = True

                # Record losses (only for valid batches)
                if not zero_loss:
                    temp_loss["total"].append(loss.cpu().detach().numpy())
                    temp_loss["MotifP"].append(l_motif_pos.cpu().detach().numpy())
                    temp_loss["MotifN"].append(l_motif_neg.cpu().detach().numpy())
                    temp_loss["MotifCont"].append(l_motif_contrast.cpu().detach().numpy())
                    temp_loss["NormPenalty"].append(l2_penalty.cpu().detach().numpy())

                    # Track motif label availability
                    if has_motif_label:
                        motif_label_stats['with_labels'] += 1
                    else:
                        motif_label_stats['without_labels'] += 1
                    if source not in motif_label_stats['by_source']:
                        motif_label_stats['by_source'][source] = {'with': 0, 'without': 0}
                    if has_motif_label:
                        motif_label_stats['by_source'][source]['with'] += 1
                    else:
                        motif_label_stats['by_source'][source]['without'] += 1

                    if l_str_dist > 0.0:
                        temp_loss["Str"].append(l_str_dist.cpu().detach().numpy())
                        temp_loss["StrMAE"].append(key_mae.cpu().detach().numpy())
                        temp_loss['StrRMSD'].append(key_rmsd.cpu().detach().numpy())
                        temp_loss["StrPair"].append(l_str_pair.cpu().detach().numpy())
                        temp_loss["KeyatmAttmap"].append(l_str_attmap.cpu().detach().numpy())

                    if l_screen_rank > 0.0:
                        temp_loss["Screen"].append(l_screen.cpu().detach().numpy())
                        temp_loss["ScreenR"].append(l_screen_rank.cpu().detach().numpy())
                        temp_loss["ScreenC"].append(l_screen_cont.cpu().detach().numpy())

                # If zero_loss, build a dummy loss through ALL model parameters
                # so DDP AllReduce hooks fire uniformly across all ranks.
                # (Using model outputs would fail when model returned NullArgs — all None.)
                if zero_loss:
                    loss = sum(
                        (p.sum() * 0.0 for p in model.parameters()),
                        torch.tensor(0.0, device=device, requires_grad=True)
                    )

                # DDP diagnostics: detect should_sync or zero_loss mismatch across ranks
                if is_train and config.training.ddp and dist.is_initialized():
                    world_size = dist.get_world_size()
                    # Pack [should_sync, zero_loss] into a tensor and all_gather
                    local_flags = torch.tensor(
                        [1.0 if should_sync else 0.0, 1.0 if zero_loss else 0.0],
                        device=device
                    )
                    gathered = [torch.zeros(2, device=device) for _ in range(world_size)]
                    dist.all_gather(gathered, local_flags)
                    sync_flags = [int(g[0].item()) for g in gathered]
                    zero_flags = [int(g[1].item()) for g in gathered]
                    if len(set(sync_flags)) > 1:
                        print(f"[Rank {rank}] DDP DESYNC: should_sync mismatch at i={i}, "
                              f"b_count={b_count}, sync_flags={sync_flags}, "
                              f"len(loader)={len(loader)}, accum={accum}")
                    if len(set(zero_flags)) > 1:
                        print(f"[Rank {rank}] DDP DESYNC: zero_loss mismatch at i={i}, "
                              f"b_count={b_count}, zero_flags={zero_flags}")

                # Backward pass: always runs to keep DDP AllReduce aligned.
                if is_train:
                    scaled_loss = loss / accum
                    if scaler is not None:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    if not zero_loss:
                        valid_micro_count += 1

                    if should_sync:
                        if valid_micro_count > 0:
                            if scaler is not None:
                                scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.max_param_norm)

                            # Skip optimizer step if grad norm is inf/nan (prevents parameter corruption)
                            if not (torch.isfinite(torch.tensor(grad_norm)) if isinstance(grad_norm, float) else torch.isfinite(grad_norm)):
                                if rank == 0:
                                    print(f"WARNING: Non-finite grad_norm ({grad_norm}) at epoch {epoch}, batch {b_count}. Skipping optimizer step.")
                                if scaler is not None:
                                    scaler.update()  # Still update scaler state
                                optimizer.zero_grad()
                            else:
                                if scaler is not None:
                                    scaler.step(optimizer)
                                    scaler.update()
                                else:
                                    optimizer.step()

                                # Update EMA weights after optimizer step
                                if ema is not None:
                                    ema.update()

                        optimizer.zero_grad()  # Always reset gradients
                        valid_micro_count = 0  # Reset for next accumulation window

                        if not zero_loss and rank == 0 and config.training.log_steps and not config.training.debug:
                            step_metrics = {
                                "step": global_step + b_count,
                                "train_step/loss_total": float(loss.cpu().detach().numpy()),
                                "train_step/loss_motif_pos": float(l_motif_pos.cpu().detach().numpy()),
                                "train_step/loss_motif_neg": float(l_motif_neg.cpu().detach().numpy()),
                                "train_step/loss_motif_contrast": float(l_motif_contrast.cpu().detach().numpy()),
                                "train_step/loss_norm_penalty": float((motif_penalty + l2_penalty).cpu().detach().numpy()),
                                "train_step/loss_l2_penalty": float(l2_penalty.cpu().detach().numpy()),
                                "train_step/loss_motif_penalty": float(motif_penalty.cpu().detach().numpy()),
                                "train_step/grad_norm": float(grad_norm),
                                "train_step/learning_rate": optimizer.param_groups[0]['lr'],
                                # Weighted losses
                                "train_step/loss_motif_pos_w": float(config.losses.w_motif * l_motif_pos.cpu().detach().numpy()),
                                "train_step/loss_motif_neg_w": float(config.losses.w_motif * l_motif_neg.cpu().detach().numpy()),
                                "train_step/loss_motif_contrast_w": float(config.losses.w_motif_contrast * l_motif_contrast.cpu().detach().numpy()),
                            }

                            if l_str_dist > 0.0:
                                step_metrics.update({
                                    "train_step/loss_structure": float(l_str_dist.cpu().detach().numpy()),
                                    "train_step/loss_structure_mae": float(key_mae.cpu().detach().numpy()),
                                    "train_step/loss_structure_rmsd": float(key_rmsd.cpu().detach().numpy()),
                                    "train_step/loss_structure_pair": float(l_str_pair.cpu().detach().numpy()),
                                    "train_step/loss_keyatm_attmap": float(l_str_attmap.cpu().detach().numpy()),
                                    # Weighted losses
                                    "train_step/loss_structure_w": float(str_warmup_mult * config.losses.w_str * l_str_dist.cpu().detach().numpy()),
                                    "train_step/loss_structure_pair_w": float(str_warmup_mult * config.losses.w_str_pair * l_str_pair.cpu().detach().numpy()),
                                    "train_step/loss_keyatm_attmap_w": float(str_warmup_mult * config.losses.w_str_attmap * l_str_attmap.cpu().detach().numpy()),
                                })

                            if l_screen_rank > 0.0:
                                step_metrics.update({
                                    "train_step/loss_screening_bce": float(l_screen.cpu().detach().numpy()),
                                    "train_step/loss_screening_rank": float(l_screen_rank.cpu().detach().numpy()),
                                    "train_step/loss_screening_contrast": float(l_screen_cont.cpu().detach().numpy()),
                                    # Weighted losses
                                    "train_step/loss_screening_bce_w": float(config.losses.w_screen_bce * l_screen.cpu().detach().numpy()),
                                    "train_step/loss_screening_rank_w": float(config.losses.w_screen_rank * l_screen_rank.cpu().detach().numpy()),
                                    "train_step/loss_screening_contrast_w": float(config.losses.w_screen_contrast * l_screen_cont.cpu().detach().numpy()),
                                })

                            wandb.log(step_metrics)

                        if not zero_loss and metrics_logger is not None:
                            batch_metrics = {
                                'loss_total': float(np.sum(temp_loss['total'][-accum:])),
                                'loss_motif_pos': float(np.sum(temp_loss['MotifP'][-accum:])),
                                'loss_motif_neg': float(np.sum(temp_loss['MotifN'][-accum:])),
                                'loss_motif_contrast': float(np.sum(temp_loss['MotifCont'][-accum:])),
                                'target': pnames[0],
                            }
                            if eval_struct and len(temp_loss['StrMAE'][-accum:]) > 0:
                                batch_metrics.update({
                                    'loss_str_dist': float(np.sum(temp_loss['Str'][-accum:])),
                                    'loss_str_mae': float(np.sum(temp_loss['StrMAE'][-accum:])),
                                    'loss_str_rmsd': float(np.sum(temp_loss['StrRMSD'][-accum:])),
                                    'loss_str_pair': float(np.sum(temp_loss['StrPair'][-accum:])),
                                    'loss_str_attmap': float(np.sum(temp_loss['KeyatmAttmap'][-accum:])),
                                })
                            if len(temp_loss['Screen'][-accum:]) > 0:
                                batch_metrics.update({
                                    'loss_screen_bce': float(np.sum(temp_loss['Screen'][-accum:])),
                                    'loss_screen_rank': float(np.sum(temp_loss['ScreenR'][-accum:])),
                                    'loss_screen_contrast': float(np.sum(temp_loss['ScreenC'][-accum:])),
                                })
                            metrics_logger.log_batch(epoch, b_count, len(loader), 'train', batch_metrics)

                            # Log binding predictions if available
                            if len(Pbind) > 0 and len(info['ligands'][0]) > 0:
                                metrics_logger.log_binding(epoch, b_count, pnames[0], info['ligands'][0], Pbind, blabel)

                elif not zero_loss and (b_count + 1) % accum == 0:
                    if rank == 0 and config.training.log_steps and not config.training.debug:
                        step_metrics = {
                            "step": global_step + b_count,
                            "valid_step/loss_total": float(loss.cpu().detach().numpy()),
                            "valid_step/loss_motif_pos": float(l_motif_pos.cpu().detach().numpy()),
                            "valid_step/loss_motif_neg": float(l_motif_neg.cpu().detach().numpy()),
                            "valid_step/loss_motif_contrast": float(l_motif_contrast.cpu().detach().numpy()),
                            "valid_step/loss_norm_penalty": float((motif_penalty + l2_penalty).cpu().detach().numpy()),
                            # Weighted losses
                            "valid_step/loss_motif_pos_w": float(config.losses.w_motif * l_motif_pos.cpu().detach().numpy()),
                            "valid_step/loss_motif_neg_w": float(config.losses.w_motif * l_motif_neg.cpu().detach().numpy()),
                            "valid_step/loss_motif_contrast_w": float(config.losses.w_motif_contrast * l_motif_contrast.cpu().detach().numpy()),
                        }

                        if l_str_dist > 0.0:
                            step_metrics.update({
                                "valid_step/loss_structure": float(l_str_dist.cpu().detach().numpy()),
                                "valid_step/loss_structure_mae": float(key_mae.cpu().detach().numpy()),
                                "valid_step/loss_structure_rmsd": float(key_rmsd.cpu().detach().numpy()),
                                "valid_step/loss_structure_pair": float(l_str_pair.cpu().detach().numpy()),
                                "valid_step/loss_keyatm_attmap": float(l_str_attmap.cpu().detach().numpy()),
                                # Weighted losses
                                "valid_step/loss_structure_w": float(str_warmup_mult * config.losses.w_str * l_str_dist.cpu().detach().numpy()),
                                "valid_step/loss_structure_pair_w": float(str_warmup_mult * config.losses.w_str_pair * l_str_pair.cpu().detach().numpy()),
                                "valid_step/loss_keyatm_attmap_w": float(str_warmup_mult * config.losses.w_str_attmap * l_str_attmap.cpu().detach().numpy()),
                            })

                        if l_screen_rank > 0.0:
                            step_metrics.update({
                                "valid_step/loss_screening_bce": float(l_screen.cpu().detach().numpy()),
                                "valid_step/loss_screening_rank": float(l_screen_rank.cpu().detach().numpy()),
                                "valid_step/loss_screening_contrast": float(l_screen_cont.cpu().detach().numpy()),
                                # Weighted losses
                                "valid_step/loss_screening_bce_w": float(config.losses.w_screen_bce * l_screen.cpu().detach().numpy()),
                                "valid_step/loss_screening_rank_w": float(config.losses.w_screen_rank * l_screen_rank.cpu().detach().numpy()),
                                "valid_step/loss_screening_contrast_w": float(config.losses.w_screen_contrast * l_screen_cont.cpu().detach().numpy()),
                            })

                        wandb.log(step_metrics)

                    if metrics_logger is not None:
                        batch_metrics = {
                            'loss_total': float(np.sum(temp_loss['total'][-accum:])),
                            'loss_motif_pos': float(np.sum(temp_loss['MotifP'][-accum:])),
                            'loss_motif_neg': float(np.sum(temp_loss['MotifN'][-accum:])),
                            'loss_motif_contrast': float(np.sum(temp_loss['MotifCont'][-accum:])),
                            'target': pnames[0],
                        }
                        if len(temp_loss['Str'][-accum:]) > 0:
                            batch_metrics.update({
                                'loss_str_dist': float(np.sum(temp_loss['Str'][-accum:])),
                                'loss_str_mae': float(np.sum(temp_loss['StrMAE'][-accum:])),
                                'loss_str_pair': float(np.sum(temp_loss['StrPair'][-accum:])),
                                'loss_str_attmap': float(np.sum(temp_loss['KeyatmAttmap'][-accum:])),
                            })
                        if len(temp_loss['Screen'][-accum:]) > 0:
                            batch_metrics.update({
                                'loss_screen_bce': float(np.sum(temp_loss['Screen'][-accum:])),
                                'loss_screen_rank': float(np.sum(temp_loss['ScreenR'][-accum:])),
                                'loss_screen_contrast': float(np.sum(temp_loss['ScreenC'][-accum:])),
                            })
                        metrics_logger.log_batch(epoch, b_count, len(loader), 'valid', batch_metrics)

                        # Log binding predictions if available
                        if len(Pbind) > 0 and len(info['ligands'][0]) > 0:
                            metrics_logger.log_binding(epoch, b_count, pnames[0], info['ligands'][0], Pbind, blabel)

                b_count += 1

    # Log motif label availability for this epoch
    if rank == 0:
        if is_train: 
            status = "Training"
        else:
            status = "Validation"
        print(f"\n=== {status} Epoch {epoch} Motif Label Availability ===")
        print(f"Total batches: {motif_label_stats['with_labels'] + motif_label_stats['without_labels']}")
        print(f"  With motif labels: {motif_label_stats['with_labels']}")
        print(f"  Without motif labels: {motif_label_stats['without_labels']}")
        print("By source:")
        for source, stats in motif_label_stats['by_source'].items():
            print(f"  {source}: {stats['with']} with labels, {stats['without']} without labels")
        print("=" * 50)

    # Build curriculum hard negatives: top-K decoys per target by score
    hard_neg_ids = {}
    if is_train:
        n_keep = getattr(config.losses, 'curriculum_top_k', 10)
        for target_key, pairs in decoy_scores_by_target.items():
            pairs.sort(key=lambda x: x[1], reverse=True)
            seen = set()
            unique = []
            for cid, sc in pairs:
                if cid not in seen:
                    seen.add(cid)
                    unique.append(cid)
                    if len(unique) >= n_keep:
                        break
            hard_neg_ids[target_key] = unique

    return temp_loss, Pt, Pf, Pt_per_target, Pf_per_target, hard_neg_ids


def train_model(rank, world_size, config: Config, config_path: str = None):
    gpu = rank % world_size
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)

    # Access debug from config.training
    if rank == 0 and not config.training.debug:
        # Flatten config object to dict for wandb
        def flatten_config(cfg, prefix=""):
            flat_dict = {}
            for key, value in cfg.__dict__.items():
                full_key = f"{prefix}{key}" if prefix else key
                if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list)):
                    # Recursively flatten nested objects
                    flat_dict.update(flatten_config(value, f"{full_key}_"))
                else:
                    flat_dict[full_key] = value
            return flat_dict

        wandb.init(
            project="motifscreen-aff",
            name=f"{config.model_note}",
            mode=config.training.wandb_mode,
            config=flatten_config(config)
        )

        # Save raw YAML config file to wandb
        if config_path:
            wandb.save(config_path, policy="now")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    model, optimizer, scheduler, start_epoch, train_loss, valid_loss, ema = load_params(rank, config)

    # Access ddp from config.training
    print(f"[Rank {rank}] Wrapping model in DDP...", flush=True)
    if config.training.ddp:
        if torch.cuda.is_available():
            ddp_model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
        else:
            ddp_model = DDP(model, find_unused_parameters=False)
    print(f"[Rank {rank}] DDP ready", flush=True)

    if config.training.debug: # Access debug from config.training
        train_datasetf = 'data/small.txt'
        valid_datasetf = 'data/small.txt'
    else:
        train_datasetf = config.train_file # Direct access to train_file and valid_file
        valid_datasetf = config.valid_file

    # Load data now takes the main config object
    print(f"[Rank {rank}] Loading train data from {train_datasetf}...", flush=True)
    train_loader, weights_train = load_data(train_datasetf, world_size, rank, config, static=False)
    print(f"[Rank {rank}] Loading valid data from {valid_datasetf}...", flush=True)
    valid_loader, weights_valid = load_data(valid_datasetf, world_size, rank, config, static=True)
    print(f"[Rank {rank}] Data loading complete. Train: {len(train_loader)} batches, Valid: {len(valid_loader)} batches", flush=True)

    auc_train = {'chembl': [], 'biolip': [], 'pdbbind': [], 'dude':[]}
    auc_valid = {'chembl': [], 'biolip': [], 'pdbbind': [], 'dude':[]}
    auc_train_per_target = {'chembl': [], 'biolip': [], 'pdbbind': [], 'dude':[]}
    auc_valid_per_target = {'chembl': [], 'biolip': [], 'pdbbind': [], 'dude':[]}
    ef_train_per_target = {'chembl': [], 'biolip': [], 'pdbbind': [], 'dude':[]}
    ef_valid_per_target = {'chembl': [], 'biolip': [], 'pdbbind': [], 'dude':[]}
    global_step = start_epoch * len(train_loader)

    # Track best average AUC per target for efficient comparison
    best_avg_auc_per_target = -1.0
    # Track best structure loss for pretraining mode (lower is better)
    best_struct_loss = float('inf')

    # Initialize metrics logger with timestamp (on all ranks for per-GPU logging)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_log_dir = join("logs", f"{config.model_note}_{timestamp}", "metrics")
    metrics_logger = MetricsLogger(log_dir=metrics_log_dir, rank=rank)

    # AMP GradScaler (only when amp is enabled)
    scaler = torch.cuda.amp.GradScaler() if config.training.amp else None

    # Hard negative memory bank for screening loss
    hn_capacity = getattr(config.losses, 'hard_neg_capacity', 0)
    hard_neg_bank = Loss.HardNegativeBank(capacity=hn_capacity) if hn_capacity > 0 else None

    # Log screening loss config (once, rank 0 only)
    if rank == 0:
        print("\n=== Screening Loss Config ===")
        print(f"  rank_type:          {getattr(config.losses, 'screen_rank_type', 'pairwise_auc')}")
        print(f"  rank_alpha:         {getattr(config.losses, 'screen_rank_alpha', 0.0)}")
        rank_warmup = getattr(config.losses, 'screen_rank_alpha_warmup', 0)
        if rank_warmup > 0:
            print(f"  rank_alpha_warmup:  {rank_warmup} epochs (linear ramp 0 -> {getattr(config.losses, 'screen_rank_alpha', 0.0)})")
        print(f"  w_screen_bce:       {config.losses.w_screen_bce}")
        print(f"  w_screen_rank:      {config.losses.w_screen_rank}")
        print(f"  w_screen_contrast:  {config.losses.w_screen_contrast}")
        print(f"  cont_top_k:         {getattr(config.losses, 'screen_cont_top_k', 0.2)}")
        print(f"  cont_top_weight:    {getattr(config.losses, 'screen_cont_top_weight', 5.0)}")
        print(f"  cont_margin:        {getattr(config.losses, 'screen_cont_margin', 1.0)}")
        print(f"  hard_neg_capacity:  {hn_capacity}")
        if hn_capacity > 0:
            print(f"  hard_neg_per_batch: {getattr(config.losses, 'hard_neg_per_batch', 32)}")
            print(f"  hard_neg_start:     epoch 1 (epoch 0 = populate only)")
        ssw = getattr(config.losses, 'screen_source_weight', None)
        if ssw:
            print(f"  source_weights:     {ssw}")
        print("=" * 30 + "\n")

    # Access max_epoch from config.training
    for epoch in range(start_epoch, config.training.max_epoch):
        print(f"\n=== Epoch {epoch}/{config.training.max_epoch} ===")

        if config.training.ddp and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Access ddp from config.training
        if config.training.ddp:
            ddp_model.train()
            temp_loss, Pt, Pf, Pt_per_target, Pf_per_target, hard_neg_ids = train_one_epoch(ddp_model, optimizer, train_loader, rank, epoch, True, config, weights_train, global_step, ema, metrics_logger, scaler, hard_neg_bank)
        else:
            model.train()
            temp_loss, Pt, Pf, Pt_per_target, Pf_per_target, hard_neg_ids = train_one_epoch(model, optimizer, train_loader, rank, epoch, True, config, weights_train, global_step, ema, metrics_logger, scaler, hard_neg_bank)

        # Update curriculum hard negatives for next epoch
        curriculum_slots = getattr(config.losses, 'curriculum_hard_slots', 0)
        if curriculum_slots > 0 and hard_neg_ids:
            train_loader.dataset.set_hard_negatives(hard_neg_ids, n_slots=curriculum_slots)
            if rank == 0:
                print(f"  Curriculum: updated hard negatives for {len(hard_neg_ids)} targets ({curriculum_slots} slots/batch)")

        for k in train_loss:
            train_loss[k].append(np.array(temp_loss[k]))

        # Check if in pretraining mode (skip screening metrics)
        pretraining_mode = getattr(config.training, 'pretraining_mode', False)

        # AUC/EF calculations only in non-pretraining mode
        if rank == 0 and not pretraining_mode:
            for key in Pt:
                if len(Pt[key]) > 10 and len(Pf[key]) > 10:
                    auc_train[key].append(calc_AUC(Pt[key], Pf[key]))

            for source in Pt_per_target:
                source_aucs = []
                source_efs = []
                for pname in Pt_per_target[source]:
                # Check if there are enough positive and negative samples for this target
                    if len(Pt_per_target[source][pname]) > 0 and len(Pf_per_target[source][pname]) > 0:
                        target_auc = calc_AUC(Pt_per_target[source][pname], Pf_per_target[source][pname])
                        target_ef = calc_enrichment_factor(Pt_per_target[source][pname], Pf_per_target[source][pname], fraction=0.01)
                        if target_auc != -1.0: # Exclude cases with insufficient data
                            source_aucs.append(target_auc)
                        if target_ef != -1.0:
                            source_efs.append(target_ef)
            # Calculate the average AUROC and EF for the source
                if source_aucs:
                    auc_train_per_target[source].append(np.mean(source_aucs))
                if source_efs:
                    ef_train_per_target[source].append(np.mean(source_efs))

        # Epoch-level wandb logging (always for rank 0)
        if rank == 0 and not config.training.debug:
            ep_str_warmup = config.losses.w_str_warmup_multiplier if epoch < config.losses.w_str_warmup_epochs else 1.0
            train_metrics = {
                "epoch": epoch,
                "train/loss_total": np.mean(train_loss['total'][-1]),
                "train/loss_motif_pos": np.mean(train_loss['MotifP'][-1]),
                "train/loss_motif_neg": np.mean(train_loss['MotifN'][-1]),
                "train/loss_motif_contrast": np.mean(train_loss['MotifCont'][-1]),
                "train/loss_norm_penalty": np.mean(train_loss['NormPenalty'][-1]),
                # Weighted
                "train/loss_motif_pos_w": config.losses.w_motif * np.mean(train_loss['MotifP'][-1]),
                "train/loss_motif_neg_w": config.losses.w_motif * np.mean(train_loss['MotifN'][-1]),
                "train/loss_motif_contrast_w": config.losses.w_motif_contrast * np.mean(train_loss['MotifCont'][-1]),
            }

            if len(train_loss['Str'][-1]) > 0:
                train_metrics.update({
                    "train/loss_structure": np.mean(train_loss['Str'][-1]),
                    "train/loss_structure_mae": np.mean(train_loss['StrMAE'][-1]),
                    "train/loss_structure_rmsd": np.mean(train_loss['StrRMSD'][-1]),
                    "train/loss_structure_pair": np.mean(train_loss['StrPair'][-1]),
                    "train/loss_keyatm_attmap": np.mean(train_loss['KeyatmAttmap'][-1]),
                    # Weighted
                    "train/loss_structure_w": ep_str_warmup * config.losses.w_str * np.mean(train_loss['Str'][-1]),
                    "train/loss_structure_pair_w": ep_str_warmup * config.losses.w_str_pair * np.mean(train_loss['StrPair'][-1]),
                    "train/loss_keyatm_attmap_w": ep_str_warmup * config.losses.w_str_attmap * np.mean(train_loss['KeyatmAttmap'][-1]),
                })

            if len(train_loss['Screen'][-1]) > 0 and not pretraining_mode:
                train_metrics.update({
                    "train/loss_screening_bce": np.mean(train_loss['Screen'][-1]),
                    "train/loss_screening_rank": np.mean(train_loss['ScreenR'][-1]),
                    "train/loss_screening_contrast": np.mean(train_loss['ScreenC'][-1]),
                    # Weighted
                    "train/loss_screening_bce_w": config.losses.w_screen_bce * np.mean(train_loss['Screen'][-1]),
                    "train/loss_screening_rank_w": config.losses.w_screen_rank * np.mean(train_loss['ScreenR'][-1]),
                    "train/loss_screening_contrast_w": config.losses.w_screen_contrast * np.mean(train_loss['ScreenC'][-1]),
                })

            if not pretraining_mode:
                for key in ['pdbbind', 'chembl', 'biolip']:
                    if key in auc_train and len(auc_train[key]) > 0:
                        train_metrics[f"train/auc_{key}"] = auc_train[key][-1]
                    if key in auc_train_per_target and len(auc_train_per_target[key]) > 0:
                        train_metrics[f"train/auc_per_target_{key}"] = auc_train_per_target[key][-1]
                        print(f"train/auc_per_target_{key}: {auc_train_per_target[key][-1]:.4f}")
                    if key in ef_train_per_target and len(ef_train_per_target[key]) > 0:
                        train_metrics[f"train/ef1_per_target_{key}"] = ef_train_per_target[key][-1]
                        print(f"train/ef1_per_target_{key}: {ef_train_per_target[key][-1]:.4f}")

            wandb.log(train_metrics)

            # Log epoch-level metrics to CSV
            if metrics_logger is not None:
                epoch_metrics = {
                    'loss_total': float(np.mean(train_loss['total'][-1])),
                    'loss_motif_pos': float(np.mean(train_loss['MotifP'][-1])),
                    'loss_motif_neg': float(np.mean(train_loss['MotifN'][-1])),
                    'loss_motif_contrast': float(np.mean(train_loss['MotifCont'][-1])),
                    'loss_norm_penalty': float(np.mean(train_loss['NormPenalty'][-1])),
                }
                if len(train_loss['Str'][-1]) > 0:
                    epoch_metrics.update({
                        'loss_structure': float(np.mean(train_loss['Str'][-1])),
                        'loss_structure_mae': float(np.mean(train_loss['StrMAE'][-1])),
                        'loss_structure_rmsd': float(np.mean(train_loss['StrRMSD'][-1])),
                        'loss_structure_pair': float(np.mean(train_loss['StrPair'][-1])),
                        'loss_keyatm_attmap': float(np.mean(train_loss['KeyatmAttmap'][-1])),
                    })
                if len(train_loss['Screen'][-1]) > 0 and not pretraining_mode:
                    epoch_metrics.update({
                        'loss_screening_bce': float(np.mean(train_loss['Screen'][-1])),
                        'loss_screening_rank': float(np.mean(train_loss['ScreenR'][-1])),
                        'loss_screening_contrast': float(np.mean(train_loss['ScreenC'][-1])),
                    })
                if not pretraining_mode:
                    for key in ['pdbbind', 'chembl', 'biolip']:
                        if key in auc_train and len(auc_train[key]) > 0:
                            epoch_metrics[f'auc_{key}'] = float(auc_train[key][-1])
                        if key in auc_train_per_target and len(auc_train_per_target[key]) > 0:
                            epoch_metrics[f'auc_per_target_{key}'] = float(auc_train_per_target[key][-1])
                        if key in ef_train_per_target and len(ef_train_per_target[key]) > 0:
                            epoch_metrics[f'ef1_per_target_{key}'] = float(ef_train_per_target[key][-1])
                metrics_logger.log_epoch(epoch, 'train', epoch_metrics)

        optimizer.zero_grad()
        global_step += len(train_loader)

        with torch.no_grad():
            # Apply EMA weights for validation if available
            if ema is not None:
                ema.apply_shadow()

            # Access ddp from config.training
            if config.training.ddp:
                ddp_model.eval()
                temp_loss, Pt, Pf, Pt_per_target, Pf_per_target, _ = train_one_epoch(ddp_model, optimizer, valid_loader, rank, epoch, False, config, weights_valid, global_step, ema, metrics_logger, None)
            else:
                model.eval()
                temp_loss, Pt, Pf, Pt_per_target, Pf_per_target, _ = train_one_epoch(model, optimizer, valid_loader, rank, epoch, False, config, weights_valid, global_step, ema, metrics_logger, None)

            # Restore original weights after validation
            if ema is not None:
                ema.restore()

        for k in valid_loss:
            valid_loss[k].append(np.array(temp_loss[k]))

        # AUC/EF calculations only in non-pretraining mode
        if rank == 0 and not pretraining_mode:
            for key in Pt:
                if len(Pt[key]) > 10 and len(Pf[key]) > 10:
                    auc_valid[key].append(calc_AUC(Pt[key], Pf[key]))

            for source in Pt_per_target:
                source_aucs = []
                source_efs = []
                for pname in Pt_per_target[source]:
                # Check if there are enough positive and negative samples for this target
                    if len(Pt_per_target[source][pname]) > 0 and len(Pf_per_target[source][pname]) > 0:
                        target_auc = calc_AUC(Pt_per_target[source][pname], Pf_per_target[source][pname])
                        target_ef = calc_enrichment_factor(Pt_per_target[source][pname], Pf_per_target[source][pname], fraction=0.01)
                        if target_auc != -1.0: # Exclude cases with insufficient data
                            source_aucs.append(target_auc)
                        if target_ef != -1.0:
                            source_efs.append(target_ef)
            # Calculate the average AUROC and EF for the source
                if source_aucs:
                    auc_valid_per_target[source].append(np.mean(source_aucs))
                if source_efs:
                    ef_valid_per_target[source].append(np.mean(source_efs))
            print(auc_valid_per_target)

        # Epoch-level wandb logging (always for rank 0)
        if rank == 0 and not config.training.debug:
            ep_str_warmup_v = config.losses.w_str_warmup_multiplier if epoch < config.losses.w_str_warmup_epochs else 1.0
            valid_metrics = {
                "epoch": epoch,
                "valid/loss_total": np.mean(valid_loss['total'][-1]),
                "valid/loss_motif_pos": np.mean(valid_loss['MotifP'][-1]),
                "valid/loss_motif_neg": np.mean(valid_loss['MotifN'][-1]),
                "valid/loss_motif_contrast": np.mean(valid_loss['MotifCont'][-1]),
                "valid/loss_norm_penalty": np.mean(valid_loss['NormPenalty'][-1]),
                # Weighted
                "valid/loss_motif_pos_w": config.losses.w_motif * np.mean(valid_loss['MotifP'][-1]),
                "valid/loss_motif_neg_w": config.losses.w_motif * np.mean(valid_loss['MotifN'][-1]),
                "valid/loss_motif_contrast_w": config.losses.w_motif_contrast * np.mean(valid_loss['MotifCont'][-1]),
            }

            if len(valid_loss['Str'][-1]) > 0:
                valid_metrics.update({
                    "valid/loss_structure": np.mean(valid_loss['Str'][-1]),
                    "valid/loss_structure_mae": np.mean(valid_loss['StrMAE'][-1]),
                    "valid/loss_structure_rmsd": np.mean(valid_loss['StrRMSD'][-1]),
                    "valid/loss_structure_pair": np.mean(valid_loss['StrPair'][-1]),
                    "valid/loss_keyatm_attmap": np.mean(valid_loss['KeyatmAttmap'][-1]),
                    # Weighted
                    "valid/loss_structure_w": ep_str_warmup_v * config.losses.w_str * np.mean(valid_loss['Str'][-1]),
                    "valid/loss_structure_pair_w": ep_str_warmup_v * config.losses.w_str_pair * np.mean(valid_loss['StrPair'][-1]),
                    "valid/loss_keyatm_attmap_w": ep_str_warmup_v * config.losses.w_str_attmap * np.mean(valid_loss['KeyatmAttmap'][-1]),
                })

            if len(valid_loss['Screen'][-1]) > 0 and not pretraining_mode:
                valid_metrics.update({
                    "valid/loss_screening_bce": np.mean(valid_loss['Screen'][-1]),
                    "valid/loss_screening_rank": np.mean(valid_loss['ScreenR'][-1]),
                    "valid/loss_screening_contrast": np.mean(valid_loss['ScreenC'][-1]),
                    # Weighted
                    "valid/loss_screening_bce_w": config.losses.w_screen_bce * np.mean(valid_loss['Screen'][-1]),
                    "valid/loss_screening_rank_w": config.losses.w_screen_rank * np.mean(valid_loss['ScreenR'][-1]),
                    "valid/loss_screening_contrast_w": config.losses.w_screen_contrast * np.mean(valid_loss['ScreenC'][-1]),
                })

            if not pretraining_mode:
                for key in ['pdbbind', 'chembl', 'biolip']:
                    if key in auc_valid and len(auc_valid[key]) > 0:
                        valid_metrics[f"valid/auc_{key}"] = auc_valid[key][-1]
                    if key in auc_valid_per_target and len(auc_valid_per_target[key]) > 0:
                        valid_metrics[f"valid/auc_per_target_{key}"] = auc_valid_per_target[key][-1]
                    if key in ef_valid_per_target and len(ef_valid_per_target[key]) > 0:
                        valid_metrics[f"valid/ef1_per_target_{key}"] = ef_valid_per_target[key][-1]

            wandb.log(valid_metrics)

            # Log epoch-level metrics to CSV
            if metrics_logger is not None:
                epoch_metrics = {
                    'loss_total': float(np.mean(valid_loss['total'][-1])),
                    'loss_motif_pos': float(np.mean(valid_loss['MotifP'][-1])),
                    'loss_motif_neg': float(np.mean(valid_loss['MotifN'][-1])),
                    'loss_motif_contrast': float(np.mean(valid_loss['MotifCont'][-1])),
                    'loss_norm_penalty': float(np.mean(valid_loss['NormPenalty'][-1])),
                }
                if len(valid_loss['Str'][-1]) > 0:
                    epoch_metrics.update({
                        'loss_structure': float(np.mean(valid_loss['Str'][-1])),
                        'loss_structure_mae': float(np.mean(valid_loss['StrMAE'][-1])),
                        'loss_structure_rmsd': float(np.mean(valid_loss['StrRMSD'][-1])),
                        'loss_structure_pair': float(np.mean(valid_loss['StrPair'][-1])),
                        'loss_keyatm_attmap': float(np.mean(valid_loss['KeyatmAttmap'][-1])),
                    })
                if len(valid_loss['Screen'][-1]) > 0 and not pretraining_mode:
                    epoch_metrics.update({
                        'loss_screening_bce': float(np.mean(valid_loss['Screen'][-1])),
                        'loss_screening_rank': float(np.mean(valid_loss['ScreenR'][-1])),
                        'loss_screening_contrast': float(np.mean(valid_loss['ScreenC'][-1])),
                    })
                if not pretraining_mode:
                    for key in ['pdbbind', 'chembl', 'biolip']:
                        if key in auc_valid and len(auc_valid[key]) > 0:
                            epoch_metrics[f'auc_{key}'] = float(auc_valid[key][-1])
                        if key in auc_valid_per_target and len(auc_valid_per_target[key]) > 0:
                            epoch_metrics[f'auc_per_target_{key}'] = float(auc_valid_per_target[key][-1])
                        if key in ef_valid_per_target and len(ef_valid_per_target[key]) > 0:
                            epoch_metrics[f'ef1_per_target_{key}'] = float(ef_valid_per_target[key][-1])
                metrics_logger.log_epoch(epoch, 'valid', epoch_metrics)

        print("***SUM***")
        print(f"Train loss | {np.mean(train_loss['total'][-1]):7.4f} | Valid loss | {np.mean(valid_loss['total'][-1]):7.4f}")

        # Step learning rate scheduler if enabled
        if scheduler is not None:
            val_loss_mean = np.mean(valid_loss['total'][-1])
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss_mean)
            else:
                scheduler.step()
            if rank == 0:
                print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        if rank == 0:
            #model_dir = join("models", f"{config.modelname}{config.version}")
            model_dir = join("models", f"{config.model_note}") #{config.modelname}{config.version}")

            # Best model selection depends on mode
            save_best = False
            current_avg_auc_value = -1.0  # Initialize for non-pretraining mode
            if pretraining_mode:
                # In pretraining mode, use structure loss (lower is better)
                current_struct_loss = np.mean(valid_loss['Str'][-1]) if len(valid_loss['Str'][-1]) > 0 else float('inf')
                if current_struct_loss < best_struct_loss:
                    best_struct_loss = current_struct_loss
                    save_best = True
                print(f"Current struct loss: {current_struct_loss:.4f}, Best struct loss: {best_struct_loss:.4f}")
            else:
                # Normal mode: use AUC per target (higher is better)
                current_avg_auc = []
                for key in ['pdbbind', 'chembl', 'biolip']:
                    if key in auc_valid_per_target and len(auc_valid_per_target[key]) > 0:
                        current_avg_auc.append(auc_valid_per_target[key][-1])

                current_avg_auc_value = np.mean(current_avg_auc) if len(current_avg_auc) > 0 else -1.0
                if current_avg_auc_value > best_avg_auc_per_target and current_avg_auc_value > 0:
                    best_avg_auc_per_target = current_avg_auc_value
                    save_best = True
                print(f"Current avg AUC per target: {current_avg_auc_value:.4f}, Best avg AUC per target: {best_avg_auc_per_target:.4f}")

            # Save best model
            if save_best:
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'auc_train': auc_train,
                    'auc_valid': auc_valid,
                }
                if ema is not None:
                    save_dict['ema_state_dict'] = ema.state_dict()
                torch.save(save_dict, join(model_dir, "best.pkl"))
                if pretraining_mode:
                    print(f"Saved best model with struct loss: {best_struct_loss:.4f}")
                else:
                    print(f"Saved best model with avg AUC per target: {current_avg_auc_value:.4f}")

            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'auc_train': auc_train,
                'auc_valid': auc_valid,
            }
            if ema is not None:
                save_dict['ema_state_dict'] = ema.state_dict()
            torch.save(save_dict, join(model_dir, "model.pkl"))

            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'auc_train': auc_train,
                'auc_valid': auc_valid,
            }
            if ema is not None:
                save_dict['ema_state_dict'] = ema.state_dict()
            torch.save(save_dict, join(model_dir, f"epoch{epoch}.pkl"))

    # Flush remaining metrics to CSV at end of training
    if metrics_logger is not None:
        metrics_logger.flush_all()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train MotifScreen-Aff')
    parser.add_argument('--config', type=str, default='common',
                        help='Config name (e.g., common) or path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    parser.add_argument('--version', type=str, help='Override model version (e.g., v1.0, v2.0)')
    parser.add_argument('--model_note', type=str, default='',
                        help='Additional note for the model name in wandb')
    parser.add_argument('--chkpt_name', type=str, default=None,
                        help='Checkpoint name to load (e.g., best, model, epochX)')
    parser.add_argument('--transfer', action='store_true',
                        help='Transfer learning mode: load model weights only, reset epoch/optimizer/loss history')
    args = parser.parse_args()

    # Load configuration
    if args.config.endswith('.yaml'):
        config = load_config(args.config)
    else:
        try:
            config = load_config_with_base(args.config)
        except FileNotFoundError:
            print(f"Config '{args.config}' not found. Using default common config.")
            raise FileNotFoundError(f"Config '{args.config}' not found. Using default common config.")
    if args.debug:
        config.training.debug = True # Access debug from config.training
        config.dataloader.num_workers = 1 # Access num_workers from config.dataloader
    if args.version:
        config.version = args.version
    if args.model_note:
        config.model_note = args.model_note
    if args.chkpt_name:
        config.chkpt_name = args.chkpt_name
    config.transfer = args.transfer
    print(f"DGL version: {dgl.__version__}")
    print(f"Using config: {args.config}")
    print(f"Using model: MSK_{config.version}")
    print(f"Training dropout: {config.dropout_rate}")

    print("\n=== Grouped Parameter Configuration ===")
    # Access parameters from their structured locations
    print(f"Graph: edgemode={config.graph.edgemode}, edgek={config.graph.edgek}, "
          f"edgedist={config.graph.edgedist}, ball_radius={config.graph.ball_radius}")
    print(f"Processing: ntype={config.processing.ntype}, max_subset={config.processing.max_subset}, "
          f"drop_H={config.processing.drop_H}")
    print(f"Augmentation: randomize={config.augmentation.randomize}")
    print(f"Cross-validation: load_cross={config.cross_validation.load_cross}, "
          f"cross_eval_struct={config.cross_validation.cross_eval_struct}")
    print("="*50)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mp.freeze_support()

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Using {world_size} GPUs.." if torch.cuda.is_available() else "Using CPU only..")

    # Access ddp from config.training
    if not torch.cuda.is_available():
        config.training.ddp = False
        print("Disabled DDP for CPU-only execution")

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12347'

    # Access ddp from config.training
    if config.training.ddp:
        mp.spawn(train_model, args=(world_size, config, args.config), nprocs=world_size, join=True)
    else:
        train_model(0, 1, config, args.config)


if __name__ == "__main__":
    main()
