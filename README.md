# BrainTrain

Deep learning pipeline for brain MRI analysis.

## Overview

Train and evaluate deep learning models on 3D brain MRI data with support for self-supervised learning (SSL) pretraining, LoRA fine-tuning, and explainability visualizations using GradCAM and saliency maps.

## Features

- **Multiple Training Modes** - SFCN, Densenet, SwinTransformer, FM linear probing, FM full fine-tuning and LoRA finetuning
- **Self-Supervised Learning** - Pretrain on unlabeled brain MRI data
- **Evaluation** - ROC/PRC curves, confusion matrices, bootstrap confidence intervals
- **Explainability** - GradCAM, gradient and saliency maps with additional regional analysis as per AAL atlas
- **Preprocessing** - DICOM to NIfTI conversion, bias correction, registration, skull stripping, resampling npy transformations

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU 

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd BrainTrain

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Format

### CSV Files

Your CSV files should contain:
- `eid` - Subject identifier (first column)
- `label` - Target variable (integer for classification, float for regression)

Example:
```csv
eid,label
sub001,1
sub002,0
```

### Image Files 

(If you have DCMs, run preprocessing.py to get npy tensors first.)
After preprocessing:
- **Format:** NumPy arrays saved as `.npy` files
- **Shape:** `(96, 96, 96)` for 3D MRI volumes or (180, 180, 180)
- **Naming:** Filename must match `eid` in CSV (e.g., `sub001.npy`)

## Quick Start

### 1. Configure Paths

Edit [config.py](config.py) to set all your desired input and output paths:

```python
TRAIN_COHORT = 'your-cohort-name'
TENSOR_DIR = f'../images/{TRAIN_COHORT}/npy96'
CSV_TRAIN = f'../data/{TRAIN_COHORT}/train/data.csv'
```

Make sure to create differnt data and image folders for train and test cohorts.

### 2. Preprocess MRI Data

```bash
python preprocess.py
```

Preprocessing pipeline:
1. DICOM to NIfTI conversion
2. N4 bias field correction
3. Registration to MNI template
4. Skull stripping
5. Resampling to 96×96×96


### 3. Train Model

```bash
# Train with default settings (LoRA fine-tuning)
python train.py

# Train with specific options
python train.py --mode lora --column label --gpu cuda:0
```

**Training modes:**
- `sfcn` - Train SFCN from scratch
- `linear` - Linear probing (frozen SSL backbone)
- `ssl-finetuned` - Fine-tune SSL pretrained model
- `lora` - LoRA adaptation (parameter-efficient)

### 4. Evaluate Model

```bash
python test.py
```

Generates:
- ROC and Precision-Recall curves with bootstrap CI
- Confusion matrix
- Classification metrics (accuracy, precision, recall, F1, AUC)

### 5. Generate Explainability Maps

```bash
python heatmap.py
```

Creates:
- GradCAM or saliency maps overlaid on MRI scans
- Regional attribution analysis using AAL atlas
- Top-N most important brain regions per prediction

### 6. Self-Supervised Pretraining (Optional)

```bash
python ssl_train.py
```

Train a self-supervised backbone on unlabeled brain MRI data. The pretrained model can be used for transfer learning in subsequent steps.

### 7. Features representation

```bash
python features.py
```

Extract features from the self-supervised pretrained model and visualize them as uMaps or tSNEs.


## Project Structure

```
BrainTrain/
├── utils/
│   ├── architectures/      # Neural network models
│   ├── dataloaders/        # Dataset loaders
│   ├── augmentations/      # SSL augmentations
│   └── ...
├── preprocess.py           # MRI preprocessing
├── ssl_train.py            # Self-supervised pretraining
├── train.py                # Model training
├── test.py                 # Model evaluation
├── heatmap.py              # Explainability visualization
└── config.py               # Configuration
```

## Outputs

Results can be saved to parent directory (or whatever locaton is suitable):
- `../models/` - Model checkpoints
- `../scores/` - Prediction scores
- `../logs/` - Training logs
- `../evaluations/` - Evaluation plots
- `../explainability/` - Heatmaps
