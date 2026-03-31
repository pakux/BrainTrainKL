"""
Main training script for brain MRI classification
"""

import os
import time
import datetime
import random
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import config as cfg
from utils import models, distribution, criterions
from utils.dataloaders import dataloader


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    outputs_list = []
    labels_list = []
    eids_list = []

    # Reset GPU memory stats at start of epoch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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

    with torch.no_grad():
        for eid, images, labels in tqdm(
            val_loader, desc="Validation", total=len(val_loader)
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

            running_loss += loss.item()
            eids_list.extend(eid)

    avg_loss = running_loss / len(val_loader)
    return avg_loss, eids_list, outputs_list, labels_list


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

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = dataloader.BidsDataset(
        cfg.CSV_TRAIN,
        cfg.TENSOR_DIR,
        cfg.COLUMN_NAME,
        task=cfg.TASK,
        num_classes=cfg.N_CLASSES if cfg.TASK == "classification" else None,
        num_rows=cfg.NROWS,
        bidstags="space-MNI152",
        session="001",
    )

    val_dataset = dataloader.BidsDataset(
        cfg.CSV_VAL,
        cfg.TENSOR_DIR,
        cfg.COLUMN_NAME,
        task=cfg.TASK,
        num_classes=cfg.N_CLASSES if cfg.TASK == "classification" else None,
        num_rows=cfg.NROWS,
        bidstags="space-MNI152",
        session="001",
    )

    # Check distribution
    if cfg.TASK == "classification":
        train_labels = train_dataset.annotations[cfg.COLUMN_NAME].values.tolist()
        val_labels = val_dataset.annotations[cfg.COLUMN_NAME].values.tolist()
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
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS
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

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg.SCHEDULER_MODE,
        factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE,
    )

    # Create log directories
    trainlog_dir = os.path.join(cfg.LOG_DIR, "trainlog", cfg.TRAINING_MODE)
    vallog_dir = os.path.join(cfg.LOG_DIR, "vallog", cfg.TRAINING_MODE)
    timelog_dir = os.path.join(cfg.LOG_DIR, "timelog", cfg.TRAINING_MODE)
    model_dir = os.path.join(cfg.MODEL_DIR, cfg.TRAINING_MODE)

    os.makedirs(trainlog_dir, exist_ok=True)
    os.makedirs(vallog_dir, exist_ok=True)
    os.makedirs(timelog_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Initialize logging
    trainlog_file = os.path.join(trainlog_dir, f"{cfg.EXPERIMENT_NAME}.txt")
    with open(trainlog_file, "w") as log:
        log.write(
            "Epoch, Training Loss, Validation Loss, Learning Rate, Epoch Time (s), Peak GPU Memory (GB)\n"
        )

    # Training loop
    best_val_loss = float("inf")
    early_stop_counter = 0
    best_val_outputs = None
    best_val_labels = None
    best_val_eids = None

    # Track computational metrics
    epoch_times = []
    peak_memories = []

    print("\n" + "=" * 70)
    print("TRAINING LOOP")
    print("=" * 70)

    for epoch in range(cfg.NUM_EPOCHS):
        epoch_start_time = time.time()

        print(f"\n--- Epoch {epoch + 1}/{cfg.NUM_EPOCHS} ---")

        # Train
        train_loss, train_eids, train_preds, train_lbls, peak_memory = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_eids, val_preds, val_lbls = validate_epoch(
            model, val_loader, criterion, device
        )

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        peak_memories.append(peak_memory)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
        )
        print(f"Epoch Time: {epoch_time:.1f}s | Peak GPU Memory: {peak_memory:.2f} GB")

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
                "val_loss": val_loss,
            }

            model_path = os.path.join(model_dir, f"{cfg.EXPERIMENT_NAME}.pth")
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
    print(f"Learning Rate: {cfg.LEARNING_RATE}")
    print(f"Epochs: {cfg.NUM_EPOCHS}")
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

