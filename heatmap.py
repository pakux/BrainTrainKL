import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
from utils.dataloaders import dataloader
from utils.architectures import sfcn_cls, sfcn_ssl2, head, lora_layers

# import configuration
import config as cfg
from utils import atlas_labels, anatomical_classes, models


def find_last_conv_layer(model):
    """Find the last convolutional layer in the model"""
    last_conv = None
    last_conv_name = None

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            last_conv = module
            last_conv_name = name

    if last_conv is None:
        raise ValueError("No Conv3d layer found in model!")

    print(f"Using layer for GradCAM: {last_conv_name}")
    return last_conv


def compute_gradcam(
    model, image, target_layer, target="logit_diff", class_idx=None, mode="magnitude"
):
    """
    Compute GradCAM attention map using intermediate layer activations.

    Args:
        model: PyTorch model
        image: Input image tensor [1,1,D,H,W]
        target_layer: Layer to compute gradients for
        target: 'logit_diff' (default), 'pred', or 'target_class'
        class_idx: Target class index (only used if target='target_class')
        mode: 'magnitude' (ReLU applied) or 'signed' (preserve sign)

    Returns:
        gradcam: Attention map [D,H,W]
        pred_class: Predicted class
        confidence: Prediction confidence
    """
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Register hooks
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward pass
        model.zero_grad(set_to_none=True)
        logits = model(image)
        probs = torch.softmax(logits, dim=1)

        # Determine target and get predictions
        if target == "logit_diff" and logits.shape[1] == 2:
            score = logits[0, 1] - logits[0, 0]
            pred_class = int(probs[0, 1] > probs[0, 0])
            conf = probs[0, pred_class].item()
        elif target == "pred":
            pred_class = torch.argmax(probs, dim=1).item()
            score = logits[0, pred_class]
            conf = probs[0, pred_class].item()
        elif target == "target_class":
            if class_idx is None:
                raise ValueError("Must specify class_idx when target='target_class'")
            pred_class = torch.argmax(probs, dim=1).item()
            score = logits[0, class_idx]
            conf = probs[0, class_idx].item()
        else:
            raise ValueError(f"Unknown target: {target}")

        # Backward pass
        score.backward()

        # Compute GradCAM
        act = activations[0]  # [1, C, D, H, W]
        grad = gradients[0]  # [1, C, D, H, W]

        # Global average pooling of gradients
        weights = grad.mean(dim=(2, 3, 4), keepdim=True)  # [1, C, 1, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * act).sum(dim=1, keepdim=True)  # [1, 1, D, H, W]

        # Apply mode
        if mode == "magnitude":
            cam = F.relu(cam)
        # else: keep signed values (mode == 'signed')

        cam = cam.squeeze().cpu().numpy()  # [D, H, W]

        # Normalize
        if mode == "magnitude":
            # Normalize to [0, 1]
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:  # signed
            # Normalize to [-1, 1] while preserving sign
            max_abs = np.abs(cam).max()
            if max_abs > 0:
                cam = cam / max_abs

        return cam, pred_class, conf

    finally:
        handle_f.remove()
        handle_b.remove()


def quantify_regions(heatmap, atlas_path, label_dict, signed=False):
    """
    Quantify attention per brain region.

    Parameters:
    -----------
    signed : bool
        If True, compute signed mean (can be negative)
        If False, compute magnitude mean (always positive)
    """
    atlas = nib.load(atlas_path).get_fdata().astype(int)
    region_scores = {}

    for region_id in np.unique(atlas):
        if region_id == 0:
            continue
        mask = atlas == region_id
        vals = heatmap[mask]
        if vals.size == 0:
            continue

        if signed:
            heat = float(vals.mean())
        else:
            heat = float(vals.mean())  # Already magnitude values

        region_scores[label_dict.get(region_id, f"Region_{region_id}")] = heat

    return region_scores


def quantify_major_regions(heatmap, atlas_path, label_dict, signed=False):
    """
    Quntify major brain regions

    """

    # Identify all labels for the major atlas regions
    anatomical_categories = anatomical_classes.ANATOMICAL_CATEGORIES
    labels = {value: key for key, value in atlas_labels.LABEL_DICT.items()}

    major_region_labels = {}

    for majorname in anatomical_categories.keys():
        subregions = anatomical_categories[majorname]["keywords"]
        major_region_labels[majorname] = [labels[reg] for reg in subregions]

    region_scores = {}
    atlas = nib.load(atlas_path).get_fdata().astype(int)

    for majorname in anatomical_categories.keys():
        mask = np.zeros(atlas.shape)

        for region_id in major_region_labels[majorname]:
            mask[atlas == region_id] = 1

        vals = heatmap[mask]
        if vals.size == 0:
            continue

        heat = float(vals.mean())
        region_scores[majorname] = heat

    return region_scores


def compute_saliency(
    model, image, target="logit_diff", class_idx=None, mode="magnitude"
):
    """
    Compute saliency map (gradient of output w.r.t. input).

    Args:
        model: PyTorch model
        image: Input image tensor [1,1,D,H,W]
        target: 'logit_diff' (default), 'pred', or 'target_class'
        class_idx: Target class index (only used if target='target_class')
        mode: 'magnitude' (absolute value) or 'signed' (preserve sign)

    Returns:
        saliency: Attention map [D,H,W]
        pred_class: Predicted class
        confidence: Prediction confidence
    """
    image.requires_grad = True
    model.zero_grad(set_to_none=True)

    logits = model(image)
    probs = torch.softmax(logits, dim=1)

    # Determine target
    if target == "logit_diff" and logits.shape[1] == 2:
        score = logits[0, 1] - logits[0, 0]
        pred_class = int(probs[0, 1] > probs[0, 0])
        conf = probs[0, pred_class].item()
    elif target == "pred":
        pred_class = torch.argmax(probs, dim=1).item()
        score = logits[0, pred_class]
        conf = probs[0, pred_class].item()
    elif target == "target_class":
        if class_idx is None:
            raise ValueError("Must specify class_idx when target='target_class'")
        pred_class = torch.argmax(probs, dim=1).item()
        score = logits[0, class_idx]
        conf = probs[0, class_idx].item()
    else:
        raise ValueError(f"Unknown target: {target}")

    # Backward
    score.backward()

    # Get saliency map
    if mode == "magnitude":
        saliency = image.grad.data.abs().squeeze().cpu().numpy()  # [D, H, W]
        # Normalize to [0, 1]
        if saliency.max() > 0:
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    else:  # signed
        saliency = image.grad.data.squeeze().cpu().numpy()  # [D, H, W]
        # Normalize to [-1, 1] while preserving sign
        max_abs = np.abs(saliency).max()
        if max_abs > 0:
            saliency = saliency / max_abs

    image.requires_grad = False
    return saliency, pred_class, conf


def load_image(path, device):
    """Load and preprocess image"""
    img_data = np.load(path)

    # Display version (8-bit for overlays)
    p1, p99 = np.percentile(img_data, (1, 99))
    img_np = np.clip(img_data, p1, p99)
    img_np = (
        (img_np - img_np.min()) / max(img_np.max() - img_np.min(), 1e-8) * 255
    ).astype(np.uint8)

    # Model input version
    img_t = torch.from_numpy(img_data).float().unsqueeze(0).unsqueeze(0).to(device)
    return img_t, img_np


def save_visualization(heatmap, image, name, output_dir, signed=False, affine=None):
    """Save heatmap visualization as overlays AND NIfTI files"""
    import nibabel as nib

    os.makedirs(output_dir, exist_ok=True)

    # Default affine if none provided
    if affine is None:
        affine = np.eye(4)

    # Save brain image as NIfTI
    brain_nifti_path = os.path.join(output_dir, f"{name}_brain.nii.gz")
    brain_img = nib.Nifti1Image(image, affine)
    nib.save(brain_img, brain_nifti_path)
    print(f"Saved brain NIfTI: {brain_nifti_path}")

    # Save heatmap as NIfTI
    heatmap_nifti_path = os.path.join(output_dir, f"{name}_heatmap.nii.gz")
    heatmap_img = nib.Nifti1Image(heatmap, affine)
    nib.save(heatmap_img, heatmap_nifti_path)
    print(f"Saved heatmap NIfTI: {heatmap_nifti_path}")

    # === PNG visualization (unchanged) ===
    D, H, W = image.shape
    slices = {"axial": D // 2, "coronal": H // 2, "sagittal": W // 2}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (view, slice_idx) in enumerate(slices.items()):
        ax = axes[idx]

        if view == "axial":
            img_slice = image[slice_idx, :, :]
            heat_slice = heatmap[slice_idx, :, :]
        elif view == "coronal":
            img_slice = image[:, slice_idx, :]
            heat_slice = heatmap[:, slice_idx, :]
        else:  # sagittal
            img_slice = image[:, :, slice_idx]
            heat_slice = heatmap[:, :, slice_idx]

        ax.imshow(img_slice.T, cmap="gray", origin="lower")

        if signed:
            im = ax.imshow(
                heat_slice.T,
                cmap="RdBu_r",
                alpha=0.5,
                vmin=-np.abs(heat_slice).max(),
                vmax=np.abs(heat_slice).max(),
                origin="lower",
            )
        else:
            im = ax.imshow(
                heat_slice.T,
                cmap="hot",
                alpha=0.5,
                vmin=0,
                vmax=heat_slice.max(),
                origin="lower",
            )

        ax.set_title(f"{view.capitalize()} (slice {slice_idx})", fontsize=12)
        ax.axis("off")

    plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    plt.suptitle(name, fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PNG: {save_path}")


def generate_heatmaps(
    heatmap_dir,
    attention_method="gradcam",
    attention_mode="magnitude",
    mode="top_individual",
    top_n=10,
    attention_target="logit_diff",
    attention_class_idx=None,
    atlas_path=None,
    label_dict=None,
):
    """
    Main function to generate heatmaps
    """
    device = cfg.DEVICE

    print("\n" + "=" * 70)
    print("HEATMAP GENERATION")
    print("=" * 70)
    print(f"Test cohort: {cfg.TEST_COHORT}")
    print(f"Attention method: {attention_method}")
    print(f"Attention mode: {attention_mode}")
    print(f"Visualization mode: {mode}")
    print(f"Top N: {top_n}")

    # Create output directories
    os.makedirs(heatmap_dir, exist_ok=True)
    model_path = f"{cfg.MODEL_DIR}/{cfg.TRAINING_MODE}/{cfg.EXPERIMENT_NAME}.pth"

    # Load model
    model = models.load_model(model_path, device)

    # Get target layer for GradCAM
    if attention_method == "gradcam":
        target_layer = find_last_conv_layer(model)

    # Load CSV data
    df = pd.read_csv(cfg.CSV_TEST)
    print(f"Test dataset size: {len(df)}")

    # Create test dataset
    test_dataset = dataloader.BrainDataset(
        csv_file=cfg.CSV_TEST,
        root_dir=cfg.TENSOR_DIR_TEST,
        column_name=cfg.COLUMN_NAME,
        num_rows=None,
        num_classes=cfg.N_CLASSES,
        task="classification",
        # bidstags="space-MNI152",
        # session="001",
        # modality="T1w",
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Process images
    results = []
    all_region_scores = []  # ✅ ADD this
    all_major_region_scores = []

    signed = attention_mode == "signed"

    print("\nGenerating heatmaps...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Get eid from row first
        eid_raw = row["eid"]

        # Handle both numeric and string eids
        if isinstance(eid_raw, (int, float)):
            eid = str(int(eid_raw))  # Convert numeric to string (removing .0)
        else:
            eid = str(eid_raw)  # Already a string

        label = int(row[cfg.COLUMN_NAME])

        # Construct image path
        image_path = os.path.join(cfg.TENSOR_DIR_TEST, f"{eid}.npy")

        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        # Load image
        image_t, image_np = load_image(image_path, device)

        # Compute attention
        if attention_method == "gradcam":
            att_map, pred_class, confidence = compute_gradcam(
                model,
                image_t,
                target_layer,
                target=attention_target,
                class_idx=attention_class_idx,
                mode=attention_mode,
            )
        else:  # saliency
            att_map, pred_class, confidence = compute_saliency(
                model,
                image_t,
                target=attention_target,
                class_idx=attention_class_idx,
                mode=attention_mode,
            )

        results.append(
            {
                "heatmap": att_map,
                "image": image_np,
                "confidence": confidence,
                "pred_class": pred_class,
                "true_class": label,
                "eid": eid,
            }
        )

        # QUANTIFY REGIONS (if atlas provided)
        if atlas_path and label_dict:
            region_scores = quantify_regions(
                att_map, atlas_path, label_dict, signed=signed
            )
            region_scores["eid"] = eid
            all_region_scores.append(region_scores)

            major_region_scores = quantify_major_regions(
                att_map, atlas_path, label_dict, signed=signed
            )
            major_region_scores["eid"] = eid
            all_major_region_scores.append(region_scores)

        if mode == "single":
            break

    if all_major_region_scores:
        regional_df = pd.DataFrame(all_major_region_scores)
        regional_df.set_index("eid", inplace=True)
        regional_csv_path = os.path.join(heatmap_dir, "major_regional_scores.csv")
        regional_df.to_csv(regional_csv_path)
        print(f"\nRegional scores saved to: {regional_csv_path}")

    # SAVE REGIONAL SCORES CSV
    if all_region_scores:
        regional_df = pd.DataFrame(all_region_scores)
        regional_df.set_index("eid", inplace=True)
        regional_csv_path = os.path.join(heatmap_dir, "regional_scores.csv")
        regional_df.to_csv(regional_csv_path)
        print(f"\nRegional scores saved to: {regional_csv_path}")

    # Generate visualizations
    print("\nCreating visualizations...")

    if mode == "single":
        result = results[0]
        save_visualization(
            result["heatmap"],
            result["image"],
            f"single_{result['eid']}_pred{result['pred_class']}_conf{result['confidence']:.3f}",
            heatmap_dir,
            signed=signed,
        )

    elif mode == "average":
        top_results = sorted(results, key=lambda x: x["confidence"], reverse=True)[
            :top_n
        ]
        avg_heatmap = np.mean([r["heatmap"] for r in top_results], axis=0)
        save_visualization(
            avg_heatmap,
            results[0]["image"],
            f"average_top{top_n}",
            heatmap_dir,
            signed=signed,
        )

    elif mode == "top_individual":
        positive_results = [r for r in results if r["pred_class"] == 1]

        if not positive_results:
            print(
                f"WARNING: No predictions for class 1 found! Using all predictions instead."
            )
            positive_results = results

        top_positive = sorted(
            positive_results, key=lambda x: x["confidence"], reverse=True
        )[:top_n]
        # ... rest of code
        for i, result in enumerate(top_positive):
            save_visualization(
                result["heatmap"],
                result["image"],
                f"top{i + 1}_{result['eid']}_conf{result['confidence']:.3f}",
                heatmap_dir,
                signed=signed,
            )

    # Save summary
    summary_df = pd.DataFrame(
        [
            {
                "eid": r["eid"],
                "pred_class": r["pred_class"],
                "true_class": r["true_class"],
                "confidence": r["confidence"],
            }
            for r in results
        ]
    )

    summary_path = os.path.join(heatmap_dir, "heatmap_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Heatmaps saved to: {heatmap_dir}")
    print("\nHeatmap generation completed!")

    return results


def main():
    """Main function"""
    # Setup
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.DEVICE)
    torch.manual_seed(42)

    print("\n" + "=" * 70)
    print("HEATMAP CONFIGURATION")
    print("=" * 70)
    print(f"Training mode: {cfg.TRAINING_MODE}")
    print(f"Model: {cfg.MODEL_DIR}")
    print(f"Test cohort: {cfg.TEST_COHORT}")
    print(f"Test CSV: {cfg.CSV_TEST}")
    print("=" * 70)

    # Set heatmap parameters
    attention_method = cfg.ATTENTION_METHOD
    attention_mode = cfg.ATTENTION_MODE
    mode = cfg.HEATMAP_MODE
    top_n = cfg.HEATMAP_TOP_N
    explainability_path = f"{cfg.COLUMN_NAME}/{cfg.TEST_COHORT}/{cfg.TRAINING_MODE}/{cfg.ATTENTION_METHOD}/{cfg.ATTENTION_MODE}/{cfg.EXPERIMENT_NAME}"

    # Create output directories
    heatmap_dir = os.path.join(cfg.EXPLAINABILITY_DIR, explainability_path)

    # ADD ATLAS PATH (adjust to your actual atlas location)
    atlas_path = cfg.ATLAS_PATH  # Update this!
    label_dict = atlas_labels.LABEL_DICT  # Make sure this is defined in config

    # Generate heatmaps with regional analysis
    generate_heatmaps(
        heatmap_dir=heatmap_dir,
        attention_method=attention_method,
        attention_mode=attention_mode,
        mode=mode,
        top_n=top_n,
        attention_target="logit_diff",
        attention_class_idx=None,
        atlas_path=atlas_path,
        label_dict=label_dict,
    )

    print("\n✓ Heatmap generation done!")

    # NOW RUN REGIONAL ANALYSIS
    print("\n" + "=" * 70)
    print("REGIONAL ANALYSIS")
    print("=" * 70)

    config_for_analysis = {
        "regional_csv": os.path.join(heatmap_dir, "regional_scores.csv"),
        "cohort": cfg.TEST_COHORT,
    }

    save_fig_path = os.path.join(heatmap_dir, "regional_analysis.png")
    analyze_regions(config_for_analysis, top_k=cfg.N_REGIONS, save_path=save_fig_path)

    print("\n✓ All done!")


# ============================================================================
# REGIONAL ANALYSIS
# ============================================================================

anatomical_categories = anatomical_classes.ANATOMICAL_CATEGORIES


def categorize_region(region_name):
    """Categorize a brain region based on its name."""
    for category, info in anatomical_categories.items():
        for keyword in info["keywords"]:
            if keyword.lower() in region_name.lower():
                return category
    return "Other"


def analyze_regions(config, top_k=100, save_path=None):
    """Analyze regional attention patterns with anatomical coloring."""

    try:
        df = pd.read_csv(config["regional_csv"], index_col="eid")
        region_means = df.mean().sort_values(ascending=False)

        top_regions = region_means.head(top_k)

        categories = [categorize_region(region) for region in top_regions.index]
        colors = [
            anatomical_categories.get(cat, {"color": "#95a5a6"})["color"]
            for cat in categories
        ]

        fig, ax = plt.subplots(figsize=(max(30, 0.15 * len(top_regions)), 8))

        x_pos = np.arange(len(top_regions))
        bars = ax.bar(
            x_pos, top_regions.values, color=colors, alpha=0.8, edgecolor="none"
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(top_regions.index, fontsize=12, rotation=60, ha="right")
        ax.set_ylabel("Mean Attention", fontsize=13, fontweight="bold")
        ax.set_title(
            f"Top Brain Regions — {config['cohort'].upper()}",
            fontsize=15,
            fontweight="bold",
            pad=20,
        )

        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

        legend_elements = [
            plt.Rectangle(
                (0, 0), 1, 1, facecolor=info["color"], edgecolor="none", label=category
            )
            for category, info in anatomical_categories.items()
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=16,
            framealpha=0.9,
            title="Brain Region",
            title_fontsize=18,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {save_path}")

        plt.show()

        print("\n" + "=" * 60)
        print("REGIONAL SUMMARY BY ANATOMICAL CATEGORY")
        print("=" * 60)

        category_stats = {}
        for region, score in top_regions.items():
            cat = categorize_region(region)
            if cat not in category_stats:
                category_stats[cat] = {"scores": [], "count": 0}
            category_stats[cat]["scores"].append(score)
            category_stats[cat]["count"] += 1

        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            mean_score = np.mean(stats["scores"])
            print(
                f"\n{cat:15s} | Count: {stats['count']:3d} | Mean Score: {mean_score:+.6f}"
            )

            cat_regions = [
                (r, s) for r, s in top_regions.items() if categorize_region(r) == cat
            ]
            for i, (region, score) in enumerate(cat_regions[:3], 1):
                print(f"  {i}. {region:40s} : {score:+.6f}")

        return region_means

    except FileNotFoundError:
        print(f"Regional CSV not found at {config['regional_csv']}. Run main() first.")
        return None


if __name__ == "__main__":
    main()

