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
    precision_score, recall_score, f1_score, balanced_accuracy_score)
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.dataloaders import dataloader
from utils.architectures import sfcn_cls, sfcn_ssl2, head, lora_layers
from utils import models
import config as cfg
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
def test(model_path, output_dir, log_dir):
    """Main test function"""
    device = cfg.DEVICE
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    print(f"Test cohort: {cfg.TEST_COHORT}")
    print(f"Test CSV: {cfg.CSV_TEST}")
    
    # Define subdirectory structure
    subdir = f'{cfg.TRAINING_MODE}/test/{cfg.TEST_COHORT}'
    
    # Create all necessary directories
    roc_dir = os.path.join(log_dir, 'roc', subdir)
    prc_dir = os.path.join(log_dir, 'prc', subdir)
    cm_dir = os.path.join(log_dir, 'cm', subdir)
    metrics_dir = os.path.join(log_dir, 'metrics', subdir)
    summary_dir = os.path.join(log_dir, 'summary', subdir)
    scores_dir = os.path.join(output_dir, subdir)
    
    directories = [roc_dir, prc_dir, cm_dir, metrics_dir, summary_dir, scores_dir]
    
    # Add Kaplan-Meier directory if enabled
    if hasattr(cfg, 'KAPLAN_MEIER') and cfg.KAPLAN_MEIER:
        km_dir = os.path.join(log_dir, 'kaplan_meier', subdir)
        directories.append(km_dir)
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"\nDirectories created:")
    print(f"  ROC: {roc_dir}")
    print(f"  PRC: {prc_dir}")
    print(f"  Confusion Matrix: {cm_dir}")
    print(f"  Metrics: {metrics_dir}")
    print(f"  Summary: {summary_dir}")
    print(f"  Scores: {scores_dir}")
    if hasattr(cfg, 'KAPLAN_MEIER') and cfg.KAPLAN_MEIER:
        print(f"  Kaplan-Meier: {km_dir}")
    
    # Load model
    model = models.load_model(model_path, device)
    
    # Create test dataset
    test_dataset = dataloader.BidsDataset(
        csv_file=cfg.CSV_TEST,
        root_dir=cfg.TENSOR_DIR_TEST,
        column_name=cfg.COLUMN_NAME,
        num_rows=None,
        num_classes=cfg.N_CLASSES,
        task='classification',
        bidstags="space-MNI152",
        session="001"

    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        drop_last=False)
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Run inference
    test_outputs_binary = []
    test_labels = []
    test_eids = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for eid, images, labels in tqdm(test_loader, desc="Testing"):
            test_eids.extend(eid)
            images = images.to(device)
            labels = labels.float().to(device)
            binary_labels = labels[:, 1]
            test_labels.extend(binary_labels.tolist())
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            binary_outputs = probs[:, 1]
            test_outputs_binary.extend(binary_outputs.tolist())
    
    # Convert to numpy
    y_true = np.array(test_labels).astype(int)
    y_score = np.array(test_outputs_binary).astype(float)
    
    # Calculate metrics with bootstrapping
    print("\nCalculating metrics...")
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
    print("RESULTS")
    print("="*70)
    print(f"AUROC: {auroc:.3f} (95% CI: {ci_auroc_lower:.3f}–{ci_auroc_upper:.3f})")
    print(f"AUPRC: {auprc:.3f} (95% CI: {ci_auprc_lower:.3f}–{ci_auprc_upper:.3f})")
    print("="*70)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'eid': test_eids,
        'label': test_labels,
        'prediction': test_outputs_binary
    })

    pred_path = os.path.join(scores_dir, f'{cfg.EXPERIMENT_NAME}.csv')
    predictions_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to {pred_path}")
    
    # Save summary metrics
    summary_df = pd.DataFrame([{
        'test_cohort': cfg.TEST_COHORT,
        'AUROC': auroc,
        'AUROC_CI_lower': ci_auroc_lower,
        'AUROC_CI_upper': ci_auroc_upper,
        'AUPRC': auprc,
        'AUPRC_CI_lower': ci_auprc_lower,
        'AUPRC_CI_upper': ci_auprc_upper}])

    summary_path = os.path.join(summary_dir, f'{cfg.EXPERIMENT_NAME}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary metrics saved to {summary_path}")
    
    # Plot ROC curve
    roc_path = os.path.join(roc_dir, f'{cfg.EXPERIMENT_NAME}.png')
    plot_roc_curve(y_true, y_score, cfg.TEST_COHORT, save_path=roc_path)
    
    # Plot PRC curve
    prc_path = os.path.join(prc_dir, f'{cfg.EXPERIMENT_NAME}.png')
    plot_prc_curve(y_true, y_score, cfg.TEST_COHORT, save_path=prc_path)
    
    # Plot confusion matrix
    cm_path = os.path.join(cm_dir, f'{cfg.EXPERIMENT_NAME}.png')
    cm_metrics = plot_confusion_matrix(y_true, y_score, threshold='youden', save_path=cm_path)
    
    # Save confusion matrix metrics
    cm_df = pd.DataFrame([cm_metrics])
    cm_metrics_path = os.path.join(metrics_dir, f'{cfg.EXPERIMENT_NAME}.csv')
    cm_df.to_csv(cm_metrics_path, index=False)
    print(f"Confusion matrix metrics saved to {cm_metrics_path}")
    
    # Find and save optimal thresholds
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
    
    # Plot Kaplan-Meier curve if enabled
    if hasattr(cfg, 'KAPLAN_MEIER') and cfg.KAPLAN_MEIER:
        print("\n" + "="*70)
        print("KAPLAN-MEIER ANALYSIS")
        print("="*70)
        
        # Read the original CSV to get survival data
        test_csv = pd.read_csv(cfg.CSV_TEST)
        
        # Merge with predictions
        km_data = test_csv.merge(predictions_df, on='eid', how='inner')
        
        # Check if required columns exist ('time' and cfg.COLUMN_NAME)
        if 'time' in km_data.columns:
            time_to_event = km_data['time'].values
            event_observed = km_data['label'].values  # Use label from predictions_df (same as cfg.COLUMN_NAME)
            prediction_scores = km_data['prediction'].values
            
            # Use Youden's threshold
            km_threshold = thresholds_dict['youden_threshold']
            print(f"Using Youden's threshold: {km_threshold:.4f}")
            
            # Plot Kaplan-Meier curve
            km_path = os.path.join(km_dir, f'{cfg.EXPERIMENT_NAME}.png')
            km_metrics = plot_kaplan_meier(
                time_to_event, 
                event_observed, 
                prediction_scores,
                cfg.TEST_COHORT,
                threshold=km_threshold,
                save_path=km_path
            )
            
            # Calculate Hazard Ratio
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
                
                # Add HR to metrics
                km_metrics['hazard_ratio'] = hr
                km_metrics['hr_ci_lower'] = hr_ci_lower
                km_metrics['hr_ci_upper'] = hr_ci_upper
                km_metrics['hr_p_value'] = hr_p_value
                
                print(f"\nHazard Ratio (High Risk vs Low Risk):")
                print(f"  HR = {hr:.3f} (95% CI: {hr_ci_lower:.3f} - {hr_ci_upper:.3f})")
                print(f"  p-value = {hr_p_value:.4f}")
                
            except Exception as e:
                print(f"\nWarning: Could not calculate Hazard Ratio: {e}")
            
            # Save KM metrics (now includes HR if calculated)
            km_metrics_df = pd.DataFrame([km_metrics])
            km_metrics_path = os.path.join(metrics_dir, f'{cfg.EXPERIMENT_NAME}_km.csv')
            km_metrics_df.to_csv(km_metrics_path, index=False)
            print(f"Kaplan-Meier metrics saved to {km_metrics_path}")
            
            print(f"\nKaplan-Meier Results:")
            print(f"  Threshold: {km_metrics['threshold']:.4f}")
            print(f"  Low Risk: n={km_metrics['n_low_risk']}, events={km_metrics['events_low_risk']}")
            print(f"  High Risk: n={km_metrics['n_high_risk']}, events={km_metrics['events_high_risk']}")
            print(f"  Log-rank p-value: {km_metrics['logrank_p_value']:.4f}")
        else:
            print(f"Warning: 'time' column not found in CSV")
            print(f"  Available columns: {km_data.columns.tolist()}")
            print("Skipping Kaplan-Meier analysis")
        
        print("="*70)
    
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
