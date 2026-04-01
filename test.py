import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, average_precision_score, 
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score)
from sklearn.preprocessing import label_binarize
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.dataloaders import dataloader
from utils.architectures import sfcn_cls, sfcn_ssl2, head, lora_layers
from utils import models, label_mapping
import BrainTrain.config as cfg
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

#%%
def bootstrap_auc(y_true, y_score, curve="roc", n_bootstraps=1000, seed=42):
    """Calculate AUC with bootstrap confidence intervals"""
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        if curve == "roc":
            fpr, tpr, _ = roc_curve(y_true[indices], y_score[indices])
            score = auc(fpr, tpr)
        elif curve == "prc":
            precision, recall, _ = precision_recall_curve(y_true[indices], y_score[indices])
            score = auc(recall, precision)
        
        bootstrapped_scores.append(score)
    
    lower = np.percentile(bootstrapped_scores, 2.5)
    upper = np.percentile(bootstrapped_scores, 97.5)
    return np.mean(bootstrapped_scores), lower, upper
#%%
def plot_roc_curve(y_true, y_score, test_cohort, save_path=None):
    """Plot ROC curve with confidence intervals"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    roc_mean, roc_lower, roc_upper = bootstrap_auc(y_true, y_score, curve="roc")
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2,
             label=f"ROC (AUC = {roc_auc:.2f} [{roc_lower:.2f}–{roc_upper:.2f}])")
    plt.plot([0, 1], [0, 1], lw=1, ls="--", color="gray")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(f"ROC Curve — {cfg.TRAINING_MODE} on {test_cohort}", fontsize=14)
    plt.legend(loc="lower right", frameon=False, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    plt.close()

def plot_prc_curve(y_true, y_score, test_cohort, save_path=None):
    """Plot Precision-Recall curve with confidence intervals"""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    prc_auc = auc(recall, precision)
    prc_mean, prc_lower, prc_upper = bootstrap_auc(y_true, y_score, curve="prc")
    pos_rate = y_true.mean()
    
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, lw=2,
             label=f"PRC (AUC = {prc_auc:.2f} [{prc_lower:.2f}–{prc_upper:.2f}])")
    plt.hlines(pos_rate, 0, 1, colors="gray", linestyles="--",
               label=f"Baseline = {pos_rate:.3f}")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title(f"PRC Curve — {cfg.TRAINING_MODE} on {test_cohort}", fontsize=14)
    plt.legend(loc="lower left", frameon=False, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PRC curve saved to {save_path}")        
    plt.close()
#%%
def plot_confusion_matrix(y_true, y_score, threshold='youden', save_path=None):
    """Plot confusion matrix at specified threshold"""
    if threshold == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        threshold_value = thresholds[np.argmax(tpr - fpr)]
    elif isinstance(threshold, (int, float)):
        threshold_value = threshold
    else:
        threshold_value = 0.5
    
    y_pred = (y_score >= threshold_value).astype(int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = (tp + tn) / cm.sum()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)
    
    print(f"\nThreshold: {threshold_value:.3f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")
    print(f"Specificity={spec:.3f}, F1={f1:.3f}, Balanced Acc={bacc:.3f}")
    
    disp = ConfusionMatrixDisplay(cm, display_labels=["0", "1"])
    disp.plot(cmap="Blues", colorbar=False, values_format="d")
    plt.grid(False)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()
    
    return {
        'threshold': threshold_value,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'specificity': spec, 'f1': f1, 'balanced_accuracy': bacc
    }


def plot_multiclass_confusion_matrix(y_true, y_pred, n_classes, save_path=None):
    """Plot multiclass confusion matrix and return macro metrics."""
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(f"Confusion Matrix — {cfg.TRAINING_MODE} on {cfg.TEST_COHORT}")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

#%%
def find_optimal_thresholds(y_true, y_score):
    """
    Find optimal thresholds using multiple methods
    
    Returns:
    --------
    dict with all threshold methods and their key metrics
    """
    # Method 1: Youden's Index (maximizes sensitivity + specificity - 1)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_index = tpr - fpr
    youden_idx = np.argmax(youden_index)
    youden_threshold = thresholds[youden_idx]
    youden_sensitivity = tpr[youden_idx]
    youden_specificity = 1 - fpr[youden_idx]
    
    # Method 2: Closest to Top-Left (minimizes distance to (0,1))
    distances = np.sqrt((1 - tpr)**2 + fpr**2)
    topleft_idx = np.argmin(distances)
    topleft_threshold = thresholds[topleft_idx]
    topleft_sensitivity = tpr[topleft_idx]
    topleft_specificity = 1 - fpr[topleft_idx]
    
    # Method 3: Balanced Accuracy (maximizes (sensitivity + specificity) / 2)
    balanced_acc = (tpr + (1 - fpr)) / 2
    balanced_idx = np.argmax(balanced_acc)
    balanced_threshold = thresholds[balanced_idx]
    balanced_sensitivity = tpr[balanced_idx]
    balanced_specificity = 1 - fpr[balanced_idx]
    
    # Method 4: F1 Score
    from sklearn.metrics import precision_recall_curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = np.zeros(len(precision))
    for i in range(len(precision)):
        if precision[i] + recall[i] > 0:
            f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    f1_idx = np.argmax(f1_scores)
    f1_threshold = pr_thresholds[f1_idx] if f1_idx < len(pr_thresholds) else 1.0
    f1_precision = precision[f1_idx]
    f1_recall = recall[f1_idx]
    
    return {
        'youden_threshold': youden_threshold,
        'youden_sensitivity': youden_sensitivity,
        'youden_specificity': youden_specificity,
        'youden_index': youden_index[youden_idx],
        'topleft_threshold': topleft_threshold,
        'topleft_sensitivity': topleft_sensitivity,
        'topleft_specificity': topleft_specificity,
        'balanced_threshold': balanced_threshold,
        'balanced_sensitivity': balanced_sensitivity,
        'balanced_specificity': balanced_specificity,
        'balanced_accuracy': balanced_acc[balanced_idx],
        'f1_threshold': f1_threshold,
        'f1_precision': f1_precision,
        'f1_recall': f1_recall,
        'f1_score': f1_scores[f1_idx]
    }

#%%
def plot_kaplan_meier(time_to_event, event_observed, prediction_scores, 
                      test_cohort, threshold, save_path=None):
    """
    Plot Kaplan-Meier curve stratified by DL model predictions
    
    Parameters:
    -----------
    time_to_event : array-like
        Time until event or censoring (in months)
    event_observed : array-like
        Binary labels (0: not progressing/censored, 1: progressing/event)
    prediction_scores : array-like
        DL model prediction scores (probabilities)
    test_cohort : str
        Name of test cohort for plot title
    threshold : float
        Threshold to stratify high-risk vs low-risk groups
    save_path : str
        Path to save the figure
    """
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time_to_event,
        'event': event_observed,
        'risk_score': prediction_scores
    })
    
    # Stratify by model predictions
    df['risk_group'] = (df['risk_score'] >= threshold).astype(int)
    
    # Initialize Kaplan-Meier fitter
    kmf = KaplanMeierFitter()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot KM curves for each risk group
    colors = ['#2ecc71', '#e74c3c']  # Green for low risk, red for high risk
    
    for idx, group in enumerate([0, 1]):
        mask = df['risk_group'] == group
        label = f'Low Risk (n={mask.sum()})' if group == 0 else f'High Risk (n={mask.sum()})'
        
        kmf.fit(df.loc[mask, 'time'], 
                df.loc[mask, 'event'], 
                label=label)
        
        kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[idx], 
                                   linewidth=2.5, alpha=0.8)
    
    # Perform log-rank test
    low_risk = df[df['risk_group'] == 0]
    high_risk = df[df['risk_group'] == 1]
    
    results = logrank_test(
        low_risk['time'], high_risk['time'],
        low_risk['event'], high_risk['event']
    )
    
    # Add labels and title
    ax.set_xlabel('Time (months)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Progression-Free Probability', fontsize=14, fontweight='bold')
    ax.set_title(f'Kaplan-Meier Curve — {cfg.TRAINING_MODE} on {test_cohort}',
                fontsize=16, fontweight='bold', pad=20)
    
    # Add log-rank test results
    p_value = results.p_value
    test_stat = results.test_statistic
    
    textstr = f'Log-rank test:\np = {p_value:.4f}\nχ² = {test_stat:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props)
    
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Kaplan-Meier curve saved to {save_path}")
    plt.close()
    
    # Return metrics
    km_metrics = {
        'threshold': threshold,
        'n_low_risk': int((df['risk_group'] == 0).sum()),
        'n_high_risk': int((df['risk_group'] == 1).sum()),
        'events_low_risk': int(low_risk['event'].sum()),
        'events_high_risk': int(high_risk['event'].sum()),
        'logrank_p_value': p_value,
        'logrank_chi2': test_stat
    }
    
    return km_metrics

#%%
def plot_regression_scatter(y_true, y_pred, test_cohort, save_path=None):
    """Plot scatter plot for regression predictions with fit line"""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    
    # Fit a line through the points
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.array([y_true.min(), y_true.max()])
    y_line = p(x_line)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Fit line
    ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'g--', linewidth=2, 
            label='Perfect prediction', alpha=0.7)
    
    # Labels and title
    ax.set_xlabel('Actual Values', fontsize=14)
    ax.set_ylabel('Predicted Values', fontsize=14)
    ax.set_title(f'Regression Predictions — {cfg.TRAINING_MODE} on {test_cohort}', fontsize=14)
    
    # Add metrics to plot
    metrics_text = f'R² = {r2:.3f}\nPearson r = {pearson_r:.3f} (p={pearson_p:.4f})\nSpearman r = {spearman_r:.3f} (p={spearman_p:.4f})\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax.legend(loc='lower right', fontsize=12, frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Regression scatter plot saved to {save_path}")
    plt.close()
    
    return {
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'fit_slope': z[0],
        'fit_intercept': z[1]
    }

def plot_regression_residuals(y_true, y_pred, test_cohort, save_path=None):
    """Plot residuals analysis for regression"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Values', fontsize=12)
    axes[0, 0].set_ylabel('Residuals', fontsize=12)
    axes[0, 0].set_title('Residuals vs Predicted', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals vs Actual
    axes[0, 1].scatter(y_true, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Actual Values', fontsize=12)
    axes[0, 1].set_ylabel('Residuals', fontsize=12)
    axes[0, 1].set_title('Residuals vs Actual', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Residual Value', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Distribution of Residuals', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Residual Analysis — {cfg.TRAINING_MODE} on {test_cohort}', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Regression residuals plot saved to {save_path}")
    plt.close()

#%%
def fit_and_save_bias_correction_coefficients(y_val_true, y_val_pred, save_path):
    """
    VALIDATION PHASE FUNCTION: Fit bias correction model on validation data.
    
    This function should be called DURING TRAINING on validation set predictions,
    NOT during testing. The fitted coefficients are then saved and applied during
    testing to prevent information leakage.
    
    USAGE IN TRAIN.PY:
    ------------------
    After validation epoch, if cfg.TASK == 'regression' and cfg.APPLY_BIAS_CORRECTION:
    
        from test import fit_and_save_bias_correction_coefficients
        
        fit_and_save_bias_correction_coefficients(
            y_val_true=val_labels,      # True validation labels
            y_val_pred=val_predictions,  # Model predictions on validation set
            save_path=cfg.BIAS_CORRECTION_COEFFICIENTS_PATH
        )
    
    Parameters:
    -----------
    y_val_true : array-like
        True labels from VALIDATION set
    y_val_pred : array-like
        Model predictions on VALIDATION set
    save_path : str
        Path to save the coefficients JSON file
    
    Returns:
    --------
    dict
        Coefficients dictionary with 'intercept' and 'slope'
    """
    from sklearn.linear_model import LinearRegression
    import json
    
    print("\n" + "="*70)
    print("FITTING BIAS CORRECTION MODEL (ON VALIDATION SET)")
    print("="*70)
    
    # Fit linear regression on validation data: true_val = β₀ + β₁ × pred_val
    X = np.array(y_val_pred).reshape(-1, 1)
    y_true = np.array(y_val_true)
    
    lr_model = LinearRegression()
    lr_model.fit(X, y_true)
    
    intercept = lr_model.intercept_
    slope = lr_model.coef_[0]
    r2_val = lr_model.score(X, y_true)
    
    print(f"Fitted on validation set ({len(y_val_true)} samples)")
    print(f"  Relationship: True = {intercept:.4f} + {slope:.4f} × Predicted")
    print(f"  R² on validation set: {r2_val:.4f}")
    
    # Save coefficients
    coefficients = {
        'intercept': float(intercept),
        'slope': float(slope),
        'r2_validation': float(r2_val),
        'n_samples_validation': int(len(y_val_true)),
        'description': 'Bias correction coefficients fitted on validation set. Apply to test predictions only.',
        'methodology': 'Linear regression: true_value = intercept + slope * predicted_value'
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(coefficients, f, indent=4)
    
    print(f"✓ Coefficients saved to {save_path}")
    print("="*70)
    
    return coefficients

#%%
def save_bias_correction_coefficients(intercept, slope, save_path):
    """
    Save bias correction coefficients to a JSON file for later use.
    
    Parameters:
    -----------
    intercept : float
        Intercept (β₀) from linear regression
    slope : float
        Slope (β₁) from linear regression
    save_path : str
        Path to save the JSON file
    """
    import json
    
    coefficients = {
        'intercept': float(intercept),
        'slope': float(slope),
        'description': 'Bias correction coefficients fitted on validation set. Use to correct test predictions.'
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(coefficients, f, indent=4)
    
    print(f"Bias correction coefficients saved to {save_path}")
    return coefficients

def load_or_calibrate_bias_correction_coefficients(coeff_path, validation_data=None):
    """
    Load pre-fitted bias correction coefficients, or calibrate on validation set if needed.
    
    Priority:
    1. Load from saved file if it exists
    2. Calibrate on validation set if file doesn't exist
    3. Return None if neither is available
    
    Parameters:
    -----------
    coeff_path : str
        Path to saved coefficients JSON file
    validation_data : tuple, optional
        (y_val_true, y_val_pred) if calibration is needed
    
    Returns:
    --------
    dict or None
        Dictionary with 'intercept' and 'slope' keys, or None
    """
    import json
    
    # Try to load from file first
    if os.path.exists(coeff_path):
        with open(coeff_path, 'r') as f:
            coefficients = json.load(f)
        print(f"\n✓ Loaded pre-fitted bias correction coefficients from:")
        print(f"  {coeff_path}")
        print(f"  Intercept: {coefficients['intercept']:.4f}")
        print(f"  Slope: {coefficients['slope']:.4f}")
        return coefficients
    
    # If file doesn't exist and validation data is provided, calibrate
    if validation_data is not None:
        y_val_true, y_val_pred = validation_data
        print(f"\n Coefficient file not found: {coeff_path}")
        print(f"   Calibrating bias correction on validation set ({len(y_val_true)} samples)...")
        
        from sklearn.linear_model import LinearRegression
        
        X = np.array(y_val_pred).reshape(-1, 1)
        y_true = np.array(y_val_true)
        
        lr_model = LinearRegression()
        lr_model.fit(X, y_true)
        
        intercept = lr_model.intercept_
        slope = lr_model.coef_[0]
        r2_val = lr_model.score(X, y_true)
        
        coefficients = {
            'intercept': float(intercept),
            'slope': float(slope),
            'r2_validation': float(r2_val),
            'n_samples_validation': int(len(y_val_true)),
            'description': 'Bias correction coefficients fitted on validation set during test phase.',
            'methodology': 'Linear regression: true_value = intercept + slope * predicted_value'
        }
        
        # Save for future use
        os.makedirs(os.path.dirname(coeff_path), exist_ok=True)
        with open(coeff_path, 'w') as f:
            json.dump(coefficients, f, indent=4)
        
        print(f"✓ Calibrated and saved to: {coeff_path}")
        print(f"  Intercept: {intercept:.4f}")
        print(f"  Slope: {slope:.4f}")
        print(f"  R² on validation: {r2_val:.4f}")
        
        return coefficients
    
    # No file and no validation data
    print(f"\n⚠️  No bias correction coefficients available:")
    print(f"   File not found: {coeff_path}")
    print(f"   Validation data not provided")
    print(f"   Skipping bias correction.")
    return None

def get_validation_predictions(model, val_loader, device):
    """
    Get predictions on validation set for bias correction calibration.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    val_loader : DataLoader
        Validation data loader
    device : str
        Device to use (cuda:0, cpu, etc.)
    
    Returns:
    --------
    tuple
        (val_labels, val_predictions)
    """
    model.eval()
    val_outputs = []
    val_labels = []
    
    print("Getting validation predictions for bias correction calibration...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            eid, images, labels = batch
            val_labels.extend(labels.tolist())
            
            images = images.to(device)
            outputs = model(images)

            # Keep regression outputs as a flat 1D list even for batch_size=1.
            if isinstance(outputs, torch.Tensor):
                outputs_list = outputs.detach().reshape(-1).cpu().tolist()
            else:
                outputs_list = torch.as_tensor(outputs).reshape(-1).cpu().tolist()
            val_outputs.extend(outputs_list)
    
    return val_labels, val_outputs

def apply_linear_bias_correction(y_true, y_pred, coefficients=None, fit_on_data=False):
    """
    Apply linear regression-based bias correction to predictions.
    
    IMPORTANT: To avoid information leakage, bias correction should be:
    - FITTED on validation/calibration data (during training phase)
    - APPLIED to test data (during testing phase)
    
    Fits a linear model: true_value = β₀ + β₁ × predicted_value
    Then corrects predictions: corrected = (predicted - β₀) / β₁
    
    Parameters:
    -----------
    y_true : array-like
        True values (ground truth). Used only if fit_on_data=True.
    y_pred : array-like
        Model predictions (before correction)
    coefficients : dict, optional
        Pre-fitted coefficients from validation set. If None, will attempt to load.
        Should contain keys: 'intercept', 'slope'
    fit_on_data : bool, default=False
        DEPRECATED - kept for backward compatibility only.
        If True: fits correction model on test set (causes information leakage!)
        If False: uses pre-fitted coefficients from validation set (correct approach)
    
    Returns:
    --------
    y_pred_corrected : array
        Bias-corrected predictions
    correction_model : dict
        Dictionary containing correction coefficients and statistics
    """
    from sklearn.linear_model import LinearRegression
    
    if not fit_on_data and coefficients is None:
        # Try to load from config path
        if hasattr(cfg, 'BIAS_CORRECTION_COEFFICIENTS_PATH'):
            coefficients = load_bias_correction_coefficients(cfg.BIAS_CORRECTION_COEFFICIENTS_PATH)
        
        if coefficients is None:
            raise ValueError(
                "Bias correction coefficients not provided and could not be loaded from config path. "
                "Coefficients must be fitted on validation set during training phase and provided here."
            )
    
    if fit_on_data:
        print("\nWARNING: Fitting bias correction on test data!")
        print("    This causes information leakage and inflates performance metrics.")
        print("    For correct methodology, fit on validation data only.")
        print("    Proceeding with current data for backward compatibility...\n")
        
        # Fit linear regression: y_true = β₀ + β₁ × y_pred
        X = y_pred.reshape(-1, 1)
        lr_model = LinearRegression()
        lr_model.fit(X, y_true)
        
        intercept = lr_model.intercept_
        slope = lr_model.coef_[0]
        r2_correction = lr_model.score(X, y_true)
        
        correction_info = {
            'intercept': intercept,
            'slope': slope,
            'r2_correction_model': r2_correction,
            'fitted_on': 'test_set (INFORMATION LEAKAGE WARNING)'
        }
    else:
        # Use pre-fitted coefficients
        intercept = coefficients['intercept']
        slope = coefficients['slope']
        
        # Calculate r2 of correction model on current data (for reference only)
        X = y_pred.reshape(-1, 1)
        y_pred_from_correction = intercept + slope * y_pred
        r2_correction = r2_score(y_true, y_pred_from_correction)
        
        correction_info = {
            'intercept': intercept,
            'slope': slope,
            'r2_correction_model': r2_correction,
            'fitted_on': 'validation_set (no information leakage)'
        }
    
    # Guard against invalid slope to avoid inf/NaN
    if not np.isfinite(slope) or abs(slope) < 1e-12:
        print("⚠️  Bias correction skipped: slope is non-finite or too small.")
        correction_info['bias_correction_skipped'] = True
        return y_pred.copy(), correction_info

    # Apply correction: corrected = (predicted - β₀) / β₁
    y_pred_corrected = (y_pred - intercept) / slope
    
    print(f"\nBias Correction Model (Linear Regression):")
    print(f"  Fitted on: {correction_info['fitted_on']}")
    print(f"  Relationship: True = {intercept:.4f} + {slope:.4f} × Predicted")
    print(f"  R² of correction model: {r2_correction:.4f}")
    print(f"  Correction formula: Corrected = (Predicted - {intercept:.4f}) / {slope:.4f}")
    
    return y_pred_corrected, correction_info

def plot_bias_correction_comparison(y_true, y_pred_raw, y_pred_corrected, test_cohort, save_path=None):
    """
    Plot comparison of predictions before and after bias correction.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred_raw : array-like
        Raw predictions (before correction)
    y_pred_corrected : array-like
        Bias-corrected predictions
    test_cohort : str
        Name of test cohort for plot title
    save_path : str
        Path to save the figure
    """
    r2_raw = r2_score(y_true, y_pred_raw)
    r2_corrected = r2_score(y_true, y_pred_corrected)
    mae_raw = mean_absolute_error(y_true, y_pred_raw)
    mae_corrected = mean_absolute_error(y_true, y_pred_corrected)
    pearson_r_raw, _ = pearsonr(y_true, y_pred_raw)
    pearson_r_corrected, _ = pearsonr(y_true, y_pred_corrected)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before correction
    axes[0].scatter(y_true, y_pred_raw, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    z = np.polyfit(y_true, y_pred_raw, 1)
    p = np.poly1d(z)
    x_line = np.array([y_true.min(), y_true.max()])
    axes[0].plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'g--', linewidth=2, label='Perfect prediction', alpha=0.7)
    axes[0].set_xlabel('Actual Values', fontsize=12)
    axes[0].set_ylabel('Predicted Values', fontsize=12)
    axes[0].set_title('Before Bias Correction', fontsize=12, fontweight='bold')
    metrics_text = f'R² = {r2_raw:.4f}\nMAE = {mae_raw:.4f}\nr = {pearson_r_raw:.4f}'
    axes[0].text(0.05, 0.95, metrics_text, transform=axes[0].transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), family='monospace')
    axes[0].legend(loc='lower right', fontsize=11, frameon=False)
    axes[0].grid(True, alpha=0.3)
    
    # After correction
    axes[1].scatter(y_true, y_pred_corrected, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    z = np.polyfit(y_true, y_pred_corrected, 1)
    p = np.poly1d(z)
    axes[1].plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'g--', linewidth=2, label='Perfect prediction', alpha=0.7)
    axes[1].set_xlabel('Actual Values', fontsize=12)
    axes[1].set_ylabel('Corrected Predictions', fontsize=12)
    axes[1].set_title('After Bias Correction', fontsize=12, fontweight='bold')
    metrics_text = f'R² = {r2_corrected:.4f}\nMAE = {mae_corrected:.4f}\nr = {pearson_r_corrected:.4f}'
    axes[1].text(0.05, 0.95, metrics_text, transform=axes[1].transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), family='monospace')
    axes[1].legend(loc='lower right', fontsize=11, frameon=False)
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Bias Correction Comparison — {cfg.TRAINING_MODE} on {test_cohort}', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bias correction comparison plot saved to {save_path}")
    plt.close()
    
    print(f"\nBias Correction Results:")
    print(f"  R² improvement: {r2_raw:.4f} → {r2_corrected:.4f} (Δ = {r2_corrected - r2_raw:+.4f})")
    print(f"  MAE improvement: {mae_raw:.4f} → {mae_corrected:.4f} (Δ = {mae_corrected - mae_raw:+.4f})")
    print(f"  Pearson r improvement: {pearson_r_raw:.4f} → {pearson_r_corrected:.4f} (Δ = {pearson_r_corrected - pearson_r_raw:+.4f})")


#%%
def test(model_path, output_dir, log_dir):
    """Main test function for both classification and regression tasks"""
    device = cfg.DEVICE
    print("\n" + "="*70)
    print(f"TESTING ({cfg.TASK.upper()})")
    print("="*70)
    print(f"Test cohort: {cfg.TEST_COHORT}")
    print(f"Test CSV: {cfg.CSV_TEST}")
    print(f"Task: {cfg.TASK}")
    
    # Define subdirectory structure
    subdir = f'{cfg.TRAINING_MODE}/test/{cfg.TEST_COHORT}'
    
    # Create all necessary directories
    roc_dir = os.path.join(log_dir, 'roc', subdir)
    prc_dir = os.path.join(log_dir, 'prc', subdir)
    cm_dir = os.path.join(log_dir, 'cm', subdir)
    metrics_dir = os.path.join(log_dir, 'metrics', subdir)
    summary_dir = os.path.join(log_dir, 'summary', subdir)
    scores_dir = os.path.join(output_dir, subdir)
    
    directories = [metrics_dir, summary_dir, scores_dir]
    
    # Task-specific directories
    if cfg.TASK == 'classification':
        directories.extend([roc_dir, prc_dir, cm_dir])
    elif cfg.TASK == 'regression':
        regression_dir = os.path.join(log_dir, 'regression', subdir)
        directories.append(regression_dir)
    
    # Add Kaplan-Meier directory if enabled
    if hasattr(cfg, 'KAPLAN_MEIER') and cfg.KAPLAN_MEIER:
        km_dir = os.path.join(log_dir, 'kaplan_meier', subdir)
        directories.append(km_dir)
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"\nDirectories created:")
    if cfg.TASK == 'classification':
        print(f"  ROC: {roc_dir}")
        print(f"  PRC: {prc_dir}")
        print(f"  Confusion Matrix: {cm_dir}")
    elif cfg.TASK == 'regression':
        print(f"  Regression: {regression_dir}")
    print(f"  Metrics: {metrics_dir}")
    print(f"  Summary: {summary_dir}")
    print(f"  Scores: {scores_dir}")
    if hasattr(cfg, 'KAPLAN_MEIER') and cfg.KAPLAN_MEIER:
        print(f"  Kaplan-Meier: {km_dir}")
    
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

    # Load model
    model = models.load_model(model_path, device)
    
    # Create test dataset
    test_dataset = dataloader.BidsDataset(
        csv_file=cfg.CSV_TEST,
        root_dir=cfg.TENSOR_DIR_TEST,
        column_name=cfg.COLUMN_NAME,
        num_rows=cfg.NROWS,
        num_classes=cfg.N_CLASSES,
<<<<<<< HEAD
        task='classification',
        bidstags="space-MNI152",
        session="001"

=======
        task=cfg.TASK,
        label_mapping=class_to_index,
>>>>>>> 703e27df398d14156844978b8dfa6dd177d77e7d
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        drop_last=True)
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create validation loader if task is regression (for bias correction calibration)
    val_loader = None
    if cfg.TASK == 'regression' and os.path.exists(cfg.CSV_VAL):
        val_dataset = dataloader.BrainDataset(
            csv_file=cfg.CSV_VAL,
            root_dir=cfg.TENSOR_DIR,
            column_name=cfg.COLUMN_NAME,
            num_rows=cfg.NROWS,
            num_classes=cfg.N_CLASSES,
            task=cfg.TASK,
            label_mapping=class_to_index,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.NUM_WORKERS,
            drop_last=False)
        print(f"Validation dataset size: {len(val_dataset)} (for bias correction calibration)")
    
    # Run inference
    if cfg.TASK == 'classification':
        return test_classification(model, test_loader, test_dataset, 
                                  output_dir, log_dir, subdir, device)
    elif cfg.TASK == 'regression':
        return test_regression(model, test_loader, test_dataset,
                             output_dir, log_dir, subdir, device, val_loader=val_loader)

def test_classification(model, test_loader, test_dataset, output_dir, log_dir, subdir, device):
    """Test function for classification task (binary and multiclass)."""

    test_prob_batches = []
    test_labels = []
    test_eids = []

    print("\nRunning inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            eid, images, labels = batch
            true_classes = torch.argmax(labels, dim=1)
            test_labels.extend(true_classes.tolist())

            test_eids.extend(eid)
            images = images.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            test_prob_batches.append(probs.cpu().numpy())

    y_true = np.array(test_labels).astype(int)
    y_prob = np.concatenate(test_prob_batches, axis=0).astype(float)

    return test_classification_metrics(
        y_true, y_prob, test_eids, test_labels, output_dir, log_dir, subdir
    )

def test_regression(model, test_loader, test_dataset, output_dir, log_dir, subdir, device, val_loader=None):
    """Test function for regression task"""
    
    # Run inference
    test_outputs = []
    test_labels = []
    test_eids = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            eid, images, labels = batch
            test_labels.extend(labels.tolist())
            
            test_eids.extend(eid)
            images = images.to(device)
            labels = labels.float().to(device)
            
            outputs = model(images)

            # Keep regression outputs as a flat 1D list even for batch_size=1.
            if isinstance(outputs, torch.Tensor):
                outputs_list = outputs.detach().reshape(-1).cpu().tolist()
            else:
                outputs_list = torch.as_tensor(outputs).reshape(-1).cpu().tolist()
            test_outputs.extend(outputs_list)
    
    # Convert to numpy
    y_true = np.array(test_labels).astype(float)
    y_pred = np.array(test_outputs).astype(float)
    
    return test_regression_metrics(y_true, y_pred, test_eids, test_labels,
                                   output_dir, log_dir, subdir, 
                                   val_loader=val_loader, device=device, model=model)

def test_classification_metrics(y_true, y_prob, test_eids, test_labels, output_dir, log_dir, subdir):
    """Calculate and save classification metrics (binary and multiclass)."""

    print("\nCalculating metrics...")
    n_classes = y_prob.shape[1]
    y_pred = np.argmax(y_prob, axis=1).astype(int)

    roc_dir = os.path.join(log_dir, 'roc', subdir)
    prc_dir = os.path.join(log_dir, 'prc', subdir)
    cm_dir = os.path.join(log_dir, 'cm', subdir)
    metrics_dir = os.path.join(log_dir, 'metrics', subdir)
    summary_dir = os.path.join(log_dir, 'summary', subdir)
    scores_dir = os.path.join(output_dir, subdir)

    for directory in [roc_dir, prc_dir, cm_dir, metrics_dir, summary_dir, scores_dir]:
        os.makedirs(directory, exist_ok=True)

    predictions_df = pd.DataFrame({
        'eid': test_eids,
        'label': test_labels,
        'pred_class': y_pred
    })
    for cls_idx in range(n_classes):
        predictions_df[f'prob_class_{cls_idx}'] = y_prob[:, cls_idx]

    pred_path = os.path.join(scores_dir, f'{cfg.EXPERIMENT_NAME}.csv')
    predictions_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to {pred_path}")

    if n_classes == 2:
        y_score = y_prob[:, 1]
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

        bootstrapped_auroc = []
        bootstrapped_auprc = []
        rng = np.random.RandomState(42)

        for _ in range(1000):
            indices = rng.randint(0, len(y_true), len(y_true))
            y_true_sample = y_true[indices]
            y_score_sample = y_score[indices]

            if len(np.unique(y_true_sample)) < 2:
                continue

            bootstrapped_auroc.append(roc_auc_score(y_true_sample, y_score_sample))
            bootstrapped_auprc.append(average_precision_score(y_true_sample, y_score_sample))

        ci_auroc_lower = np.percentile(bootstrapped_auroc, 2.5)
        ci_auroc_upper = np.percentile(bootstrapped_auroc, 97.5)
        ci_auprc_lower = np.percentile(bootstrapped_auprc, 2.5)
        ci_auprc_upper = np.percentile(bootstrapped_auprc, 97.5)

        print("\n" + "="*70)
        print("CLASSIFICATION RESULTS (BINARY)")
        print("="*70)
        print(f"AUROC: {auroc:.3f} (95% CI: {ci_auroc_lower:.3f}–{ci_auroc_upper:.3f})")
        print(f"AUPRC: {auprc:.3f} (95% CI: {ci_auprc_lower:.3f}–{ci_auprc_upper:.3f})")
        print("="*70)

        summary_df = pd.DataFrame([{
            'test_cohort': cfg.TEST_COHORT,
            'AUROC': auroc,
            'AUROC_CI_lower': ci_auroc_lower,
            'AUROC_CI_upper': ci_auroc_upper,
            'AUPRC': auprc,
            'AUPRC_CI_lower': ci_auprc_lower,
            'AUPRC_CI_upper': ci_auprc_upper
        }])
        summary_path = os.path.join(summary_dir, f'{cfg.EXPERIMENT_NAME}.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary metrics saved to {summary_path}")

        roc_path = os.path.join(roc_dir, f'{cfg.EXPERIMENT_NAME}.png')
        plot_roc_curve(y_true, y_score, cfg.TEST_COHORT, save_path=roc_path)

        prc_path = os.path.join(prc_dir, f'{cfg.EXPERIMENT_NAME}.png')
        plot_prc_curve(y_true, y_score, cfg.TEST_COHORT, save_path=prc_path)

        cm_path = os.path.join(cm_dir, f'{cfg.EXPERIMENT_NAME}.png')
        cm_metrics = plot_confusion_matrix(y_true, y_score, threshold='youden', save_path=cm_path)

        cm_df = pd.DataFrame([cm_metrics])
        cm_metrics_path = os.path.join(metrics_dir, f'{cfg.EXPERIMENT_NAME}.csv')
        cm_df.to_csv(cm_metrics_path, index=False)
        print(f"Confusion matrix metrics saved to {cm_metrics_path}")

        print("\n" + "="*70)
        print("FINDING OPTIMAL THRESHOLDS")
        print("="*70)
        thresholds_dict = find_optimal_thresholds(y_true, y_score)
        thresholds_df = pd.DataFrame([thresholds_dict])
        thresholds_path = os.path.join(metrics_dir, f'{cfg.EXPERIMENT_NAME}_thresholds.csv')
        thresholds_df.to_csv(thresholds_path, index=False)
        print(f"Optimal thresholds saved to {thresholds_path}")

        print("\nOptimal Thresholds:")
        for method, value in thresholds_dict.items():
            if 'threshold' in method.lower():
                print(f"  {method}: {value:.4f}")

        if hasattr(cfg, 'KAPLAN_MEIER') and cfg.KAPLAN_MEIER:
            print("\n" + "="*70)
            print("KAPLAN-MEIER ANALYSIS")
            print("="*70)

            km_dir = os.path.join(log_dir, 'kaplan_meier', subdir)
            os.makedirs(km_dir, exist_ok=True)
            test_csv = pd.read_csv(cfg.CSV_TEST)
            km_data = test_csv.merge(predictions_df, on='eid', how='inner')

            if 'time' in km_data.columns:
                time_to_event = km_data['time'].values
                event_observed = km_data['label'].values
                prediction_scores = km_data['prob_class_1'].values
                km_threshold = thresholds_dict['youden_threshold']
                print(f"Using Youden's threshold: {km_threshold:.4f}")

                km_path = os.path.join(km_dir, f'{cfg.EXPERIMENT_NAME}.png')
                km_metrics = plot_kaplan_meier(
                    time_to_event,
                    event_observed,
                    prediction_scores,
                    cfg.TEST_COHORT,
                    threshold=km_threshold,
                    save_path=km_path
                )

                from lifelines import CoxPHFitter
                cox_df = pd.DataFrame({
                    'time': time_to_event,
                    'event': event_observed,
                    'high_risk': (prediction_scores >= km_threshold).astype(int)
                })

                try:
                    cph = CoxPHFitter()
                    cph.fit(cox_df, duration_col='time', event_col='event')

                    hr = np.exp(cph.params_['high_risk'])
                    hr_ci_lower = np.exp(cph.confidence_intervals_['95% lower-bound']['high_risk'])
                    hr_ci_upper = np.exp(cph.confidence_intervals_['95% upper-bound']['high_risk'])
                    hr_p_value = cph.summary.loc['high_risk', 'p']

                    km_metrics['hazard_ratio'] = hr
                    km_metrics['hr_ci_lower'] = hr_ci_lower
                    km_metrics['hr_ci_upper'] = hr_ci_upper
                    km_metrics['hr_p_value'] = hr_p_value

                    print(f"\nHazard Ratio (High Risk vs Low Risk):")
                    print(f"  HR = {hr:.3f} (95% CI: {hr_ci_lower:.3f} - {hr_ci_upper:.3f})")
                    print(f"  p-value = {hr_p_value:.4f}")
                except Exception as e:
                    print(f"\nWarning: Could not calculate Hazard Ratio: {e}")

                km_metrics_df = pd.DataFrame([km_metrics])
                km_metrics_path = os.path.join(metrics_dir, f'{cfg.EXPERIMENT_NAME}_km.csv')
                km_metrics_df.to_csv(km_metrics_path, index=False)
                print(f"Kaplan-Meier metrics saved to {km_metrics_path}")
            else:
                print(f"Warning: 'time' column not found in CSV")
                print(f"  Available columns: {km_data.columns.tolist()}")
                print("Skipping Kaplan-Meier analysis")

            print("="*70)
    else:
        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        try:
            auroc_macro_ovr = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except ValueError:
            auroc_macro_ovr = np.nan

        y_true_binarized = label_binarize(y_true, classes=np.arange(n_classes))
        try:
            auprc_macro = average_precision_score(y_true_binarized, y_prob, average='macro')
        except ValueError:
            auprc_macro = np.nan

        print("\n" + "="*70)
        print(f"CLASSIFICATION RESULTS (MULTICLASS, {n_classes} CLASSES)")
        print("="*70)
        print(f"Accuracy: {acc:.3f}")
        print(f"Balanced Accuracy: {bacc:.3f}")
        print(f"Precision (macro): {prec_macro:.3f}")
        print(f"Recall (macro): {rec_macro:.3f}")
        print(f"F1 (macro): {f1_macro:.3f}")
        print(f"AUROC (macro, OVR): {auroc_macro_ovr:.3f}" if np.isfinite(auroc_macro_ovr) else "AUROC (macro, OVR): nan")
        print(f"AUPRC (macro): {auprc_macro:.3f}" if np.isfinite(auprc_macro) else "AUPRC (macro): nan")
        print("="*70)

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(metrics_dir, f'{cfg.EXPERIMENT_NAME}_report.csv')
        report_df.to_csv(report_path)
        print(f"Per-class report saved to {report_path}")

        cm_path = os.path.join(cm_dir, f'{cfg.EXPERIMENT_NAME}.png')
        cm_metrics = plot_multiclass_confusion_matrix(y_true, y_pred, n_classes, save_path=cm_path)

        summary_dict = {
            'test_cohort': cfg.TEST_COHORT,
            'n_classes': n_classes,
            'accuracy': acc,
            'balanced_accuracy': bacc,
            'precision_macro': prec_macro,
            'recall_macro': rec_macro,
            'f1_macro': f1_macro,
            'auroc_macro_ovr': auroc_macro_ovr,
            'auprc_macro': auprc_macro,
            **cm_metrics,
        }
        metrics_df = pd.DataFrame([summary_dict])
        metrics_path = os.path.join(metrics_dir, f'{cfg.EXPERIMENT_NAME}.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")

        summary_df = pd.DataFrame([summary_dict])
        summary_path = os.path.join(summary_dir, f'{cfg.EXPERIMENT_NAME}.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary metrics saved to {summary_path}")

    print("\nTesting completed!")
    return summary_df

def test_regression_metrics(y_true, y_pred, test_eids, test_labels, output_dir, log_dir, subdir, val_loader=None, device=None, model=None):
    """Calculate and save regression metrics"""
    print("\nCalculating regression metrics...")

    # Filter out non-finite values to avoid sklearn errors
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.all(finite_mask):
        num_bad = int(np.size(finite_mask) - np.sum(finite_mask))
        print(f"Warning: Found {num_bad} non-finite predictions/labels; filtering for metrics.")
        y_true = y_true[finite_mask]
        y_pred = y_pred[finite_mask]
        test_eids = np.array(test_eids)[finite_mask].tolist()
        test_labels = np.array(test_labels)[finite_mask].tolist()
        if y_true.size == 0:
            raise ValueError("All regression predictions/labels are non-finite after filtering.")
    
    # Calculate raw regression metrics (before correction)
    mse_raw = mean_squared_error(y_true, y_pred)
    rmse_raw = np.sqrt(mse_raw)
    mae_raw = mean_absolute_error(y_true, y_pred)
    r2_raw = r2_score(y_true, y_pred)
    pearson_r_raw, pearson_p_raw = pearsonr(y_true, y_pred)
    spearman_r_raw, spearman_p_raw = spearmanr(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error) - raw
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape_raw = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / np.abs(y_true[non_zero_mask]))) * 100
    else:
        mape_raw = np.nan
    
    print("\n" + "="*70)
    print("REGRESSION RESULTS (RAW PREDICTIONS)")
    print("="*70)
    print(f"R² Score: {r2_raw:.4f}")
    print(f"RMSE: {rmse_raw:.4f}")
    print(f"MAE: {mae_raw:.4f}")
    print(f"MSE: {mse_raw:.4f}")
    if not np.isnan(mape_raw):
        print(f"MAPE: {mape_raw:.4f}%")
    print(f"Pearson Correlation: r={pearson_r_raw:.4f}, p={pearson_p_raw:.4e}")
    print(f"Spearman Correlation: ρ={spearman_r_raw:.4f}, p={spearman_p_raw:.4e}")
    print("="*70)
    
    # Apply bias correction (optional)
    y_pred_corrected = y_pred.copy()
    correction_info = None
    if getattr(cfg, 'APPLY_BIAS_CORRECTION', False):
        print("\n" + "="*70)
        print("APPLYING LINEAR BIAS CORRECTION")
        print("="*70)
        
        # Try to load pre-fitted coefficients, or calibrate on validation set
        bias_correction_coeffs = None
        
        if hasattr(cfg, 'BIAS_CORRECTION_COEFFICIENTS_PATH'):
            coeff_path = cfg.BIAS_CORRECTION_COEFFICIENTS_PATH
            
            # Check if we need to calibrate
            if not os.path.exists(coeff_path) and val_loader is not None and model is not None and device is not None:
                # Get validation predictions for calibration
                print(f"Calibrating bias correction using validation set ({cfg.CSV_VAL})...")
                val_labels, val_preds = get_validation_predictions(model, val_loader, device)
                bias_correction_coeffs = load_or_calibrate_bias_correction_coefficients(
                    coeff_path, 
                    validation_data=(val_labels, val_preds)
                )
            else:
                # Try to load pre-fitted coefficients
                bias_correction_coeffs = load_or_calibrate_bias_correction_coefficients(coeff_path)
        
        if bias_correction_coeffs is not None:
            y_pred_corrected, correction_info = apply_linear_bias_correction(
                y_true, y_pred, coefficients=bias_correction_coeffs, fit_on_data=False
            )
        else:
            print("No bias correction coefficients available.")
            print("   Skipping bias correction for this test run.\n")
    else:
        print("\nBias correction disabled (cfg.APPLY_BIAS_CORRECTION=False).")
    
    # Filter again after correction (correction can introduce non-finite values)
    finite_mask_corrected = np.isfinite(y_true) & np.isfinite(y_pred_corrected)
    if not np.all(finite_mask_corrected):
        num_bad = int(np.size(finite_mask_corrected) - np.sum(finite_mask_corrected))
        print(f"Warning: Found {num_bad} non-finite corrected predictions; filtering for metrics/plots.")
        y_true = y_true[finite_mask_corrected]
        y_pred = y_pred[finite_mask_corrected]
        y_pred_corrected = y_pred_corrected[finite_mask_corrected]
        test_eids = np.array(test_eids)[finite_mask_corrected].tolist()
        test_labels = np.array(test_labels)[finite_mask_corrected].tolist()
        if y_true.size == 0:
            raise ValueError("All corrected predictions are non-finite after filtering.")

    # Calculate corrected metrics
    mse_corrected = mean_squared_error(y_true, y_pred_corrected)
    rmse_corrected = np.sqrt(mse_corrected)
    mae_corrected = mean_absolute_error(y_true, y_pred_corrected)
    r2_corrected = r2_score(y_true, y_pred_corrected)
    pearson_r_corrected, pearson_p_corrected = pearsonr(y_true, y_pred_corrected)
    spearman_r_corrected, spearman_p_corrected = spearmanr(y_true, y_pred_corrected)
    
    # Calculate MAPE - corrected
    non_zero_mask_corrected = y_true != 0
    if non_zero_mask_corrected.sum() > 0:
        mape_corrected = np.mean(
            np.abs(
                (y_true[non_zero_mask_corrected] - y_pred_corrected[non_zero_mask_corrected])
                / np.abs(y_true[non_zero_mask_corrected])
            )
        ) * 100
    else:
        mape_corrected = np.nan
    
    print("\n" + "="*70)
    print("REGRESSION RESULTS (BIAS-CORRECTED PREDICTIONS)")
    print("="*70)
    print(f"R² Score: {r2_corrected:.4f}")
    print(f"RMSE: {rmse_corrected:.4f}")
    print(f"MAE: {mae_corrected:.4f}")
    print(f"MSE: {mse_corrected:.4f}")
    if not np.isnan(mape_corrected):
        print(f"MAPE: {mape_corrected:.4f}%")
    print(f"Pearson Correlation: r={pearson_r_corrected:.4f}, p={pearson_p_corrected:.4e}")
    print(f"Spearman Correlation: ρ={spearman_r_corrected:.4f}, p={spearman_p_corrected:.4e}")
    print("="*70)
    
    # Create directory structure for regression
    regression_dir = os.path.join(log_dir, 'regression', subdir)
    metrics_dir = os.path.join(log_dir, 'metrics', subdir)
    summary_dir = os.path.join(log_dir, 'summary', subdir)
    scores_dir = os.path.join(output_dir, subdir)
    
    for directory in [regression_dir, metrics_dir, summary_dir, scores_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Save predictions (both raw and corrected)
    predictions_df = pd.DataFrame({
        'eid': test_eids,
        'actual': test_labels,
        'predicted_raw': y_pred,
        'predicted_corrected': y_pred_corrected
    })
    
    pred_path = os.path.join(scores_dir, f'{cfg.EXPERIMENT_NAME}.csv')
    predictions_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to {pred_path}")
    
    # Save summary metrics (use corrected metrics as primary)
    summary_dict = {
        'test_cohort': cfg.TEST_COHORT,
        # Raw metrics
        'r2_score_raw': r2_raw,
        'rmse_raw': rmse_raw,
        'mae_raw': mae_raw,
        'mse_raw': mse_raw,
        'pearson_r_raw': pearson_r_raw,
        'pearson_p_raw': pearson_p_raw,
        'spearman_r_raw': spearman_r_raw,
        'spearman_p_raw': spearman_p_raw,
        # Corrected metrics
        'r2_score': r2_corrected,
        'rmse': rmse_corrected,
        'mae': mae_corrected,
        'mse': mse_corrected,
        'pearson_r': pearson_r_corrected,
        'pearson_p': pearson_p_corrected,
        'spearman_r': spearman_r_corrected,
        'spearman_p': spearman_p_corrected,
    }
    
    # Add bias correction info if available
    if correction_info is not None:
        summary_dict.update({
            'bias_correction_intercept': correction_info['intercept'],
            'bias_correction_slope': correction_info['slope'],
            'bias_correction_r2': correction_info['r2_correction_model'],
            'bias_correction_fitted_on': correction_info['fitted_on'],
        })
    
    if not np.isnan(mape_raw):
        summary_dict['mape_raw'] = mape_raw
    if not np.isnan(mape_corrected):
        summary_dict['mape'] = mape_corrected
    
    summary_df = pd.DataFrame([summary_dict])
    summary_path = os.path.join(summary_dir, f'{cfg.EXPERIMENT_NAME}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary metrics saved to {summary_path}")
    
    # Plot scatter plot with predictions (raw)
    scatter_path_raw = os.path.join(regression_dir, f'{cfg.EXPERIMENT_NAME}_scatter_raw.png')
    scatter_metrics_raw = plot_regression_scatter(y_true, y_pred, cfg.TEST_COHORT, save_path=scatter_path_raw)
    
    # Plot scatter plot with predictions (corrected)
    scatter_path_corrected = os.path.join(regression_dir, f'{cfg.EXPERIMENT_NAME}_scatter_corrected.png')
    scatter_metrics_corrected = plot_regression_scatter(y_true, y_pred_corrected, cfg.TEST_COHORT, save_path=scatter_path_corrected)
    
    # Plot bias correction comparison
    comparison_path = os.path.join(regression_dir, f'{cfg.EXPERIMENT_NAME}_bias_correction_comparison.png')
    plot_bias_correction_comparison(y_true, y_pred, y_pred_corrected, cfg.TEST_COHORT, save_path=comparison_path)
    
    # Save detailed metrics (including corrected)
    metrics_dict = summary_dict.copy()
    metrics_dict.update(scatter_metrics_corrected)
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_path = os.path.join(metrics_dir, f'{cfg.EXPERIMENT_NAME}.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Detailed metrics saved to {metrics_path}")
    
    # Plot residuals analysis (on corrected predictions)
    residuals_path = os.path.join(regression_dir, f'{cfg.EXPERIMENT_NAME}_residuals.png')
    plot_regression_residuals(y_true, y_pred_corrected, cfg.TEST_COHORT, save_path=residuals_path)
    
    print("\nTesting completed!")
    return summary_df
#%%
def main():
    """Main function"""
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.DEVICE)
    torch.manual_seed(42)
    
    model_path = f'{cfg.MODEL_DIR}/{cfg.TRAINING_MODE}/{cfg.EXPERIMENT_NAME}.pth'

    # Run testing
    test(model_path, cfg.SCORES_DIR, cfg.EVALUATION_DIR)    
    print("\n✓ All done!")

if __name__ == "__main__":
    main()
