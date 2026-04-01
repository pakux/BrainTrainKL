"""
SSL Training Script - Minimal version, cleaned up from original
"""

import os
import json
import csv
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from monai.utils import set_determinism
from monai.losses import ContrastiveLoss
from monai.data import DataLoader, Dataset
from utils.augmentations.ssl_transforms import double_view_transform
from utils.architectures import sfcn_ssl2

# Import config
import BrainTrain.config as config

# Seeding
set_determinism(seed=config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)
torch.manual_seed(config.SEED)

# Create directories
os.makedirs(config.MODEL_DIR_SSL, exist_ok=True)
os.makedirs(config.LOG_DIR_SSL, exist_ok=True)

# Load data
with open(config.JSON_PATH, "r") as json_f:
    json_data = json.load(json_f)
    train_data = json_data["training"]
    val_data = json_data["validation"]

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# Transforms
train_transforms = double_view_transform(img_size=config.IMG_SIZE)

# DataLoaders
train_ds = Dataset(data=train_data, transform=train_transforms)
train_loader = DataLoader(
    train_ds, batch_size=config.BATCH_SIZE_SSL, shuffle=True, 
    num_workers=config.NUM_WORKERS, drop_last=True
)

val_ds = Dataset(data=val_data, transform=train_transforms)
val_loader = DataLoader(
    val_ds, batch_size=config.BATCH_SIZE_SSL, shuffle=False, 
    num_workers=config.NUM_WORKERS, drop_last=True
)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = sfcn_ssl2.SFCN()
model = torch.nn.DataParallel(model)
model = model.to(device)

# Loss and optimizer
contrastive_loss = ContrastiveLoss(temperature=config.CONTRASTIVE_TEMPERATURE)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR_SSL)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=config.SCHEDULER_FACTOR_SSL, patience=config.SCHEDULER_PATIENCE_SSL
)


# Metrics tracking
epoch_loss_values = []
val_loss_values = []
best_val_loss = float('inf')
patience_counter = 0

# CSV for metrics
csv_path = os.path.join(config.LOG_DIR_SSL, f"metrics_b{config.BATCH_SIZE_SSL}_e{config.MAX_EPOCHS_SSL}.csv")
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate'])

# Training loop
print("\nStarting training...")
for epoch in range(config.MAX_EPOCHS_SSL):
    print(f"\nEpoch {epoch + 1}/{config.MAX_EPOCHS_SSL}")
    print("-" * 50)
    
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        inputs = batch_data["image"].to(device)
        inputs_2 = batch_data["image_2"].to(device)
        
        features1, outputs_v1 = model(inputs)
        features2, outputs_v2 = model(inputs_2)
        
        cl_loss = contrastive_loss(outputs_v1, outputs_v2)
        
        optimizer.zero_grad()
        cl_loss.backward()
        optimizer.step()
        
        epoch_loss += cl_loss.item()
        
        if step % 10 == 0 or step == 1 or step == len(train_loader):
            print(f"  Step {step}/{len(train_loader)}, Loss: {cl_loss.item():.4f}")
    
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"Average training loss: {epoch_loss:.4f}")
    
    # Validation
    if epoch % config.VAL_INTERVAL == 0:
        print("Validating...")
        model.eval()
        total_val_loss = 0
        val_step = 0
        
        with torch.no_grad():
            for val_batch in val_loader:
                val_step += 1
                inputs = val_batch["image"].to(device)
                inputs_2 = val_batch["image_2"].to(device)
                features1, outputs_v1 = model(inputs)
                features2, outputs_v2 = model(inputs_2)
                val_loss = contrastive_loss(outputs_v1, outputs_v2)
                total_val_loss += val_loss.item()
        
        total_val_loss /= val_step
        val_loss_values.append(total_val_loss)
        print(f"Validation loss: {total_val_loss:.4f}")
        
        # Scheduler
        scheduler.step(total_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch + 1, epoch_loss, total_val_loss, current_lr])
        
        # Save best model
        if total_val_loss < best_val_loss:
            print(f"Validation improved from {best_val_loss:.4f} to {total_val_loss:.4f}")
            best_val_loss = total_val_loss
            patience_counter = 0
            
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "train_loss": epoch_loss,
            }
            torch.save(
                checkpoint, 
                os.path.join(config.MODEL_DIR, f"best_model_b{config.BATCH_SIZE_SSL}_e{config.MAX_EPOCHS_SSL}.pt")
            )
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            if patience_counter >= config.PATIENCE_SSL:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_loss_values, label="Train")
        plt.plot(range(0, len(val_loss_values) * config.VAL_INTERVAL_SSL, config.VAL_INTERVAL_SSL), val_loss_values, label="Val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.savefig(os.path.join(config.LOG_DIR, f"loss_plot_b{config.BATCH_SIZE_SSL}_e{config.MAX_EPOCHS_SSL}.png"))
        plt.close()

print("\nTraining completed!")
print(f"Best validation loss: {best_val_loss:.4f}")

# Save final model
final_checkpoint = {
    "epoch": epoch + 1,
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "best_val_loss": best_val_loss,
    "train_loss": epoch_loss,
}
torch.save(
    final_checkpoint, 
    os.path.join(config.MODEL_DIR, f"final_model_b{config.BATCH_SIZE_SSL}_e{config.MAX_EPOCHS_SSL}.pt")
)