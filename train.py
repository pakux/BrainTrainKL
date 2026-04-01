"""
Main training script for brain MRI classification
"""

import os
import time
import datetime
import random
import copy
import math
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import BrainTrain.config as cfg
from utils import models, distribution, criterions, label_mapping
from utils.dataloaders import dataloader


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

<<<<<<< HEAD

def train_epoch(model, train_loader, criterion, optimizer, device):
=======
def train_epoch(model, train_loader, criterion, optimizer, device, scaler, accum_steps, scheduler=None, scheduler_policy='none'):
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    outputs_list = []
    labels_list = []
    eids_list = []

    # Reset GPU memory stats at start of epoch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
<<<<<<< HEAD

    for eid, images, labels in tqdm(
        train_loader, desc="Training", total=len(train_loader)
    ):
        images = images.to(device)

        if cfg.TASK == "classification":
            labels = labels.float().to(device)
            outputs = model(images).to(device)

            binary_labels = labels[:, 1]
            probs = torch.nn.functional.softmax(outputs, dim=1)
            binary_outputs = probs[:, 1]

            loss = criterion(outputs, labels)

            outputs_list.extend(binary_outputs.tolist())
            labels_list.extend(binary_labels.tolist())

        elif cfg.TASK == "regression":
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images).to(device)

            loss = criterion(outputs, labels)

            outputs_list.extend(outputs.squeeze().tolist())
            labels_list.extend(labels.squeeze().tolist())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
=======
    
    optimizer.zero_grad(set_to_none=True)
    use_amp = cfg.USE_AMP and torch.cuda.is_available()
    autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=use_amp)

    for step_idx, (eid, images, labels) in enumerate(tqdm(train_loader, desc="Training", total=len(train_loader))):
        images = images.to(device)
        
        with autocast_ctx():
            if cfg.TASK == 'classification':
                labels = labels.float().to(device)
                outputs = model(images).to(device)
                class_targets = torch.argmax(labels, dim=1).long()

                loss = criterion(outputs, class_targets)

                if cfg.N_CLASSES == 2:
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    positive_scores = probs[:, 1]
                    outputs_list.extend(positive_scores.tolist())
                    labels_list.extend(class_targets.tolist())
                else:
                    pred_classes = torch.argmax(outputs, dim=1)
                    outputs_list.extend(pred_classes.tolist())
                    labels_list.extend(class_targets.tolist())
            
            elif cfg.TASK == 'regression':
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images).to(device)
                
                loss = criterion(outputs, labels)
                
                outputs_list.extend(outputs.squeeze().tolist())
                labels_list.extend(labels.squeeze().tolist())

        raw_loss = loss
        loss = loss / accum_steps
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
            optimizer_stepped = True
            if use_amp:
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= prev_scale
            else:
                optimizer.step()
            if scheduler is not None and scheduler_policy == 'onecycle' and optimizer_stepped:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        
        running_loss += raw_loss.item()
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
        eids_list.extend(eid)

    # Get peak memory for this epoch
    peak_memory_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    )

    avg_loss = running_loss / len(train_loader)
    return avg_loss, eids_list, outputs_list, labels_list, peak_memory_gb


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    outputs_list = []
    labels_list = []
    eids_list = []
<<<<<<< HEAD

=======
    
    use_amp = cfg.USE_AMP and torch.cuda.is_available()
    autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=use_amp)
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
    with torch.no_grad():
        for eid, images, labels in tqdm(
            val_loader, desc="Validation", total=len(val_loader)
        ):
            images = images.to(device)
<<<<<<< HEAD

            if cfg.TASK == "classification":
                labels = labels.float().to(device)
                outputs = model(images).to(device)

                binary_labels = labels[:, 1]
                probs = torch.nn.functional.softmax(outputs, dim=1)
                binary_outputs = probs[:, 1]

                loss = criterion(outputs, labels)

                outputs_list.extend(binary_outputs.tolist())
                labels_list.extend(binary_labels.tolist())

            elif cfg.TASK == "regression":
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images).to(device)

                loss = criterion(outputs, labels)

                outputs_list.extend(outputs.squeeze().tolist())
                labels_list.extend(labels.squeeze().tolist())

=======
            
            with autocast_ctx():
                if cfg.TASK == 'classification':
                    labels = labels.float().to(device)
                    outputs = model(images).to(device)
                    class_targets = torch.argmax(labels, dim=1).long()

                    loss = criterion(outputs, class_targets)

                    if cfg.N_CLASSES == 2:
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        positive_scores = probs[:, 1]
                        outputs_list.extend(positive_scores.tolist())
                        labels_list.extend(class_targets.tolist())
                    else:
                        pred_classes = torch.argmax(outputs, dim=1)
                        outputs_list.extend(pred_classes.tolist())
                        labels_list.extend(class_targets.tolist())
                
                elif cfg.TASK == 'regression':
                    labels = labels.float().unsqueeze(1).to(device)
                    outputs = model(images).to(device)
                    
                    loss = criterion(outputs, labels)
                    
                    outputs_list.extend(outputs.squeeze().tolist())
                    labels_list.extend(labels.squeeze().tolist())
            
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
            running_loss += loss.item()
            eids_list.extend(eid)

    avg_loss = running_loss / len(val_loader)
    return avg_loss, eids_list, outputs_list, labels_list


def lr_range_test(model, train_loader, criterion, optimizer, device):
    """Run a quick LR range test and return a suggested base LR."""
    min_lr = float(getattr(cfg, "LR_RANGE_MIN", 1e-6))
    max_lr = float(getattr(cfg, "LR_RANGE_MAX", 1e-2))
    num_steps = int(getattr(cfg, "LR_RANGE_STEPS", 100))
    max_batches = getattr(cfg, "LR_RANGE_MAX_BATCHES", None)

    total_steps = min(len(train_loader), num_steps)
    if max_batches is not None:
        total_steps = min(total_steps, int(max_batches))
    if total_steps < 2:
        print("LR range test skipped (not enough batches).")
        return cfg.LEARNING_RATE

    print("\nRunning LR range test...")

    # Save state so the test doesn't affect training.
    model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), total_steps)
    losses = []
    best_loss = float("inf")
    smooth_loss = 0.0
    beta = 0.98

    use_amp = cfg.USE_AMP and torch.cuda.is_available()
    autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=use_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step_idx, (_, images, labels) in enumerate(train_loader):
        if step_idx >= total_steps:
            break

        lr = float(lrs[step_idx])
        for group in optimizer.param_groups:
            group["lr"] = lr

        images = images.to(device)
        with autocast_ctx():
            if cfg.TASK == 'classification':
                labels = labels.float().to(device)
                outputs = model(images).to(device)
                class_targets = torch.argmax(labels, dim=1).long()
                loss = criterion(outputs, class_targets)
            else:
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images).to(device)
                loss = criterion(outputs, labels)

        if not torch.isfinite(loss):
            break

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_val = float(loss.item())
        smooth_loss = beta * smooth_loss + (1 - beta) * loss_val
        smooth_loss_corrected = smooth_loss / (1 - beta ** (step_idx + 1))
        losses.append((lr, smooth_loss_corrected))

        if smooth_loss_corrected < best_loss:
            best_loss = smooth_loss_corrected
        if smooth_loss_corrected > best_loss * 4:
            break

    # Restore state
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

    if not losses:
        print("LR range test failed; using configured LR.")
        return cfg.LEARNING_RATE

    lrs_list, loss_list = zip(*losses)
    min_idx = int(np.argmin(loss_list))
    picked_lr = float(lrs_list[min_idx]) * 0.5
    picked_lr = float(max(min_lr, min(picked_lr, max_lr)))

    print(
        f"LR range test: min loss at {lrs_list[min_idx]:.2e}, "
        f"suggested LR {picked_lr:.2e}"
    )
    return picked_lr


def save_predictions(eids, labels, predictions, save_path):
    """Save predictions to CSV"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = {
        "eid": eids,
        "label": labels,
        "prediction": predictions,
    }
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")


def train():
    """Main training function"""
    device = cfg.DEVICE

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    start_time = time.time()
<<<<<<< HEAD
=======
    
    class_to_index = None
    if cfg.TASK == 'classification':
        class_to_index = label_mapping.resolve_or_create_label_mapping(
            csv_path=cfg.CSV_TRAIN,
            column_name=cfg.COLUMN_NAME,
            mapping_path=cfg.LABEL_MAP_PATH,
            auto_create=getattr(cfg, "LABEL_MAP_AUTO", False),
        )
        if class_to_index is not None:
            mapped_n_classes = len(class_to_index)
            if cfg.N_CLASSES_EXPLICIT and cfg.N_CLASSES != mapped_n_classes:
                raise ValueError(
                    f"N_CLASSES={cfg.N_CLASSES} does not match label map size "
                    f"{mapped_n_classes} at {cfg.LABEL_MAP_PATH}"
                )
            if not cfg.N_CLASSES_EXPLICIT and cfg.N_CLASSES != mapped_n_classes:
                print(f"Updating N_CLASSES from {cfg.N_CLASSES} to {mapped_n_classes} based on label map.")
                cfg.N_CLASSES = mapped_n_classes
            print(f"Using label map: {cfg.LABEL_MAP_PATH}")
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = dataloader.BidsDataset(
        cfg.CSV_TRAIN,
        cfg.TENSOR_DIR,
        cfg.COLUMN_NAME,
        task=cfg.TASK,
<<<<<<< HEAD
        num_classes=cfg.N_CLASSES if cfg.TASK == "classification" else None,
        num_rows=cfg.NROWS,
        bidstags="space-MNI152",
        session="001",
=======
        num_classes=cfg.N_CLASSES if cfg.TASK == 'classification' else None,
        num_rows=cfg.NROWS,
        label_mapping=class_to_index,
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
    )

    val_dataset = dataloader.BidsDataset(
        cfg.CSV_VAL,
        cfg.TENSOR_DIR,
        cfg.COLUMN_NAME,
        task=cfg.TASK,
<<<<<<< HEAD
        num_classes=cfg.N_CLASSES if cfg.TASK == "classification" else None,
        num_rows=cfg.NROWS,
        bidstags="space-MNI152",
        session="001",
    )

    # Check distribution
    if cfg.TASK == "classification":
        train_labels = train_dataset.annotations[cfg.COLUMN_NAME].values.tolist()
        val_labels = val_dataset.annotations[cfg.COLUMN_NAME].values.tolist()
=======
        num_classes=cfg.N_CLASSES if cfg.TASK == 'classification' else None,
        num_rows=cfg.NROWS,
        label_mapping=class_to_index,
    )

    # Check distribution
    if cfg.TASK == 'classification':
        train_labels_raw = train_dataset.annotations[cfg.COLUMN_NAME].values.tolist()
        val_labels_raw = val_dataset.annotations[cfg.COLUMN_NAME].values.tolist()
        if class_to_index is not None:
            train_labels = [class_to_index[str(v).strip()] for v in train_labels_raw]
            val_labels = [class_to_index[str(v).strip()] for v in val_labels_raw]
        else:
            train_labels = [int(v) for v in train_labels_raw]
            val_labels = [int(v) for v in val_labels_raw]
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
        print(f"\nTraining set - {Counter(train_labels)}")
        print(f"Validation set - {Counter(val_labels)}")
    else:
        train_labels = None
        print(f"\nTraining set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
<<<<<<< HEAD
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS
=======
        shuffle=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        num_workers=cfg.NUM_WORKERS, 
        drop_last=True,
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
    )

    # Create model and optimizer
    print("\nInitializing model...")
    model, optimizer = models.create_model(device)

    # Calculate and print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 70)
    print("MODEL PARAMETERS")
    print("=" * 70)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    print("=" * 70)

    # Get criterion
    criterion = criterions.get_criterion(device, train_labels)

<<<<<<< HEAD
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg.SCHEDULER_MODE,
        factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE,
    )
=======
    # Optional LR range test to auto-pick LR
    if getattr(cfg, "LR_RANGE_TEST", False):
        picked_lr = lr_range_test(model, train_loader, criterion, optimizer, device)
        for group in optimizer.param_groups:
            group["lr"] = picked_lr
        cfg.LEARNING_RATE = picked_lr
    
    # Create scheduler
    scheduler_policy = cfg.SCHEDULER_TYPE
    if cfg.SCHEDULER_TYPE == 'onecycle':
        onecycle_max_lr_by_mode = {
            'sfcn': float(getattr(cfg, "ONECYCLE_MAX_LR_SFCN", cfg.LEARNING_RATE)),
            'dense': float(getattr(cfg, "ONECYCLE_MAX_LR_DENSE", cfg.LEARNING_RATE)),
            'swin': float(getattr(cfg, "ONECYCLE_MAX_LR_SWIN", cfg.LEARNING_RATE)),
        }
        if cfg.TRAINING_MODE in onecycle_max_lr_by_mode:
            max_lr = onecycle_max_lr_by_mode[cfg.TRAINING_MODE]
            steps_per_epoch = max(1, math.ceil(len(train_loader) / max(cfg.GRAD_ACCUM_STEPS, 1)))
            total_steps = max(1, steps_per_epoch * cfg.NUM_EPOCHS)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=float(getattr(cfg, "ONECYCLE_PCT_START", 0.3)),
                anneal_strategy=str(getattr(cfg, "ONECYCLE_ANNEAL_STRATEGY", "cos")),
                cycle_momentum=bool(getattr(cfg, "ONECYCLE_CYCLE_MOMENTUM", False)),
                div_factor=float(getattr(cfg, "ONECYCLE_DIV_FACTOR", 25.0)),
                final_div_factor=float(getattr(cfg, "ONECYCLE_FINAL_DIV_FACTOR", 1e4)),
            )
            print(
                f"Using OneCycleLR | mode={cfg.TRAINING_MODE} | max_lr={max_lr:.2e} | "
                f"steps/epoch={steps_per_epoch} | total_steps={total_steps}"
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=cfg.SCHEDULER_MODE,
                factor=cfg.SCHEDULER_FACTOR,
                patience=cfg.SCHEDULER_PATIENCE
            )
            scheduler_policy = 'plateau'
            print(
                f"SCHEDULER_TYPE='onecycle' ignored for mode={cfg.TRAINING_MODE}; "
                "falling back to ReduceLROnPlateau."
            )
    elif cfg.SCHEDULER_TYPE == 'cosine':
        warmup_epochs = max(int(getattr(cfg, "WARMUP_EPOCHS", 0)), 0)
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: float(epoch + 1) / float(warmup_epochs)
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(cfg.NUM_EPOCHS - warmup_epochs, 1),
                eta_min=cfg.MIN_LR
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.NUM_EPOCHS,
                eta_min=cfg.MIN_LR
            )
    elif cfg.SCHEDULER_TYPE == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.SCHEDULER_MODE,
            factor=cfg.SCHEDULER_FACTOR,
            patience=cfg.SCHEDULER_PATIENCE
        )
    elif cfg.SCHEDULER_TYPE == 'none':
        scheduler = None
    else:
        raise ValueError(f"Invalid SCHEDULER_TYPE: {cfg.SCHEDULER_TYPE}")
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d

    # Create log directories
    trainlog_dir = os.path.join(cfg.LOG_DIR, "trainlog", cfg.TRAINING_MODE)
    vallog_dir = os.path.join(cfg.LOG_DIR, "vallog", cfg.TRAINING_MODE)
    timelog_dir = os.path.join(cfg.LOG_DIR, "timelog", cfg.TRAINING_MODE)
    model_dir = os.path.join(cfg.MODEL_DIR, cfg.TRAINING_MODE)

    os.makedirs(trainlog_dir, exist_ok=True)
    os.makedirs(vallog_dir, exist_ok=True)
    os.makedirs(timelog_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
<<<<<<< HEAD

    # Initialize logging
    trainlog_file = os.path.join(trainlog_dir, f"{cfg.EXPERIMENT_NAME}.txt")
    with open(trainlog_file, "w") as log:
        log.write(
            "Epoch, Training Loss, Validation Loss, Learning Rate, Epoch Time (s), Peak GPU Memory (GB)\n"
        )

    # Training loop
    best_val_loss = float("inf")
=======
    
    # Initialize logging and checkpoint path
    trainlog_file = os.path.join(trainlog_dir, f"{cfg.EXPERIMENT_NAME}.txt")
    model_path = os.path.join(model_dir, f"{cfg.EXPERIMENT_NAME}.pth")

    # Training loop state
    best_val_loss = float('inf')
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
    early_stop_counter = 0
    best_val_outputs = None
    best_val_labels = None
    best_val_eids = None
<<<<<<< HEAD

=======
    start_epoch = 0
    
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
    # Track computational metrics
    epoch_times = []
    peak_memories = []

    print("\n" + "=" * 70)
    print("TRAINING LOOP")
<<<<<<< HEAD
    print("=" * 70)

    for epoch in range(cfg.NUM_EPOCHS):
=======
    print("="*70)
    
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP and torch.cuda.is_available())

    # Resume from checkpoint if requested
    if getattr(cfg, "RESUME_CHECKPOINT", None):
        resume_path = cfg.RESUME_CHECKPOINT
        print(f"\nLoading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
        if state_dict is None:
            raise ValueError("Checkpoint missing model weights ('state_dict' or 'model_state_dict').")
        model.load_state_dict(state_dict, strict=False)

        if "optimizer" in checkpoint and not cfg.RESUME_RESET_LR:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif cfg.RESUME_RESET_LR:
            for group in optimizer.param_groups:
                group["lr"] = cfg.LEARNING_RATE

        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint and cfg.USE_AMP and torch.cuda.is_available():
            scaler.load_state_dict(checkpoint["scaler"])

        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_loss = float(checkpoint.get("val_loss", best_val_loss))

        print(
            f"Resumed at epoch {start_epoch}/{cfg.NUM_EPOCHS} "
            f"(best val loss so far: {best_val_loss:.4f})"
        )
        if start_epoch >= cfg.NUM_EPOCHS:
            raise ValueError(
                f"Checkpoint epoch is already at/after NUM_EPOCHS "
                f"({start_epoch} >= {cfg.NUM_EPOCHS}). Increase NUM_EPOCHS to continue."
            )

    # Write or append train log
    if start_epoch > 0 and os.path.exists(trainlog_file):
        with open(trainlog_file, "a") as log:
            log.write(f"# Resumed from {getattr(cfg, 'RESUME_CHECKPOINT', '')}\n")
    else:
        with open(trainlog_file, "w") as log:
            log.write('Epoch, Training Loss, Validation Loss, Learning Rate, Epoch Time (s), Peak GPU Memory (GB)\n')

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
        epoch_start_time = time.time()

        print(f"\n--- Epoch {epoch + 1}/{cfg.NUM_EPOCHS} ---")

        # Train
        train_loss, train_eids, train_preds, train_lbls, peak_memory = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, cfg.GRAD_ACCUM_STEPS,
            scheduler=scheduler, scheduler_policy=scheduler_policy
        )

        # Validate
        val_loss, val_eids, val_preds, val_lbls = validate_epoch(
            model, val_loader, criterion, device
        )
<<<<<<< HEAD

=======
        # Keep a valid fallback even when no new best is found after resuming.
        if best_val_outputs is None:
            best_val_outputs = val_preds
            best_val_labels = val_lbls
            best_val_eids = val_eids
        
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        peak_memories.append(peak_memory)
<<<<<<< HEAD

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
        )
        print(f"Epoch Time: {epoch_time:.1f}s | Peak GPU Memory: {peak_memory:.2f} GB")

=======
        
        if scheduler_policy == 'cosine':
            scheduler.step()
        elif scheduler_policy == 'plateau':
            scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}')
        print(f'Epoch Time: {epoch_time:.1f}s | Peak GPU Memory: {peak_memory:.2f} GB')
        
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
        # Save best model
        if val_loss < best_val_loss:
            print(
                f"✓ New best model! (previous: {best_val_loss:.4f}, current: {val_loss:.4f})"
            )
            best_val_loss = val_loss
            best_val_outputs = val_preds
            best_val_labels = val_lbls
            best_val_eids = val_eids

            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
<<<<<<< HEAD
                "val_loss": val_loss,
            }

            model_path = os.path.join(model_dir, f"{cfg.EXPERIMENT_NAME}.pth")
=======
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if cfg.USE_AMP and torch.cuda.is_available() else None,
                "val_loss": val_loss
            }
            
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
            torch.save(checkpoint, model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement ({early_stop_counter}/{cfg.PATIENCE})")

        # Early stopping
        if early_stop_counter >= cfg.PATIENCE:
            print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
            break

        # Log to file
        with open(trainlog_file, "a") as log:
            log.write(
                f"{epoch + 1}, {train_loss:.4f}, {val_loss:.4f}, {current_lr:.6e}, {epoch_time:.1f}, {peak_memory:.2f}\n"
            )

    # Save final predictions
    print("\nSaving predictions...")
    train_pred_dir = os.path.join(
        cfg.SCORES_DIR, cfg.TRAINING_MODE, "train", cfg.TRAIN_COHORT
    )
    train_pred_path = os.path.join(train_pred_dir, f"{cfg.EXPERIMENT_NAME}.csv")
    val_pred_dir = os.path.join(
        cfg.SCORES_DIR, cfg.TRAINING_MODE, "val", cfg.TRAIN_COHORT
    )
    val_pred_path = os.path.join(val_pred_dir, f"{cfg.EXPERIMENT_NAME}.csv")
    os.makedirs(train_pred_dir, exist_ok=True)
    os.makedirs(val_pred_dir, exist_ok=True)

    save_predictions(train_eids, train_lbls, train_preds, train_pred_path)
    save_predictions(best_val_eids, best_val_labels, best_val_outputs, val_pred_path)

<<<<<<< HEAD
=======
    # Fit and save bias correction coefficients on validation data (regression only)
    if cfg.TASK == 'regression' and getattr(cfg, 'APPLY_BIAS_CORRECTION', False):
        try:
            from test import fit_and_save_bias_correction_coefficients
            fit_and_save_bias_correction_coefficients(
                y_val_true=best_val_labels,
                y_val_pred=best_val_outputs,
                save_path=cfg.BIAS_CORRECTION_COEFFICIENTS_PATH
            )
        except Exception as exc:
            print(f"\n⚠️  Bias correction calibration failed: {exc}")
    
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
    # Calculate computational statistics
    avg_epoch_time = np.mean(epoch_times)
    avg_peak_memory = np.mean(peak_memories)
    max_peak_memory = np.max(peak_memories)

    # Log results
    duration = time.time() - start_time

    vallog_file = os.path.join(vallog_dir, f"{cfg.EXPERIMENT_NAME}.txt")
    with open(vallog_file, "w") as log:
        log.write(f"Training completed\n")
        log.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
        log.write(f"Stopped at epoch: {epoch + 1}\n")

    timelog_file = os.path.join(timelog_dir, f"{cfg.EXPERIMENT_NAME}.txt")
    with open(timelog_file, "w") as log:
        log.write(f"Total duration: {duration:.2f}s ({duration / 60:.2f} min)\n")
        log.write(f"Average epoch time: {avg_epoch_time:.2f}s\n")
        log.write(f"Average peak GPU memory: {avg_peak_memory:.2f} GB\n")
        log.write(f"Max peak GPU memory: {max_peak_memory:.2f} GB\n")
        log.write(f"Start: {datetime.datetime.fromtimestamp(start_time)}\n")
        log.write(f"End: {datetime.datetime.fromtimestamp(time.time())}\n")
        log.write(f"Total params: {total_params:,}\n")
        log.write(f"Trainable params: {trainable_params:,}\n")

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total duration: {duration:.2f}s ({duration / 60:.2f} min)")
    print(f"Average epoch time: {avg_epoch_time:.2f}s")
    print(f"Average peak GPU memory: {avg_peak_memory:.2f} GB")
    print(f"Max peak GPU memory: {max_peak_memory:.2f} GB")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model saved: {model_path}")
    print("=" * 70)

    return model, best_val_loss


def main():
    """Main function"""
    # Setup
    set_seed(cfg.SEED)

    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.DEVICE)

    # Print config
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"Training Mode: {cfg.TRAINING_MODE}")
    print(f"Task: {cfg.TASK}")
    print(f"Cohort: {cfg.TRAIN_COHORT}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Grad Accum Steps: {cfg.GRAD_ACCUM_STEPS}")
    print(f"Effective Batch Size: {cfg.BATCH_SIZE * max(cfg.GRAD_ACCUM_STEPS, 1)}")
    print(f"Learning Rate: {cfg.LEARNING_RATE}")
    print(f"Scheduler: {cfg.SCHEDULER_TYPE}")
    print(f"AMP Enabled: {cfg.USE_AMP}")
    print(f"Epochs: {cfg.NUM_EPOCHS}")
    print(f"Resume Checkpoint: {cfg.RESUME_CHECKPOINT}")
    print(f"Device: {cfg.DEVICE}")
    print(f"Experiment: {cfg.EXPERIMENT_NAME}")
    print("=" * 70)

    # Check data distributions
    print("\n=== Training Data ===")
    distribution.check_data_distribution(cfg.CSV_TRAIN, cfg.COLUMN_NAME, cfg.TASK)

    print("\n=== Validation Data ===")
    distribution.check_data_distribution(cfg.CSV_VAL, cfg.COLUMN_NAME, cfg.TASK)

    # Train model
    train()


if __name__ == "__main__":
    main()
<<<<<<< HEAD

=======
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
