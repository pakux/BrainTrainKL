#! /usr/bin/env python

import nilearn


import os
import bids
import argparse
import torchio as tio
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from rich.progress import track
from os.path import basename, splitext, join, exists
import logging
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import torch

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Atlasfile")
    parser.add_argument(
        "--img_size",
        type=int,
        default=180,
        help="Image size for cropping/padding (default: 180)",
    )
    parser.add_argument("--atlas_name", type=str, default="AAL")
    parser.add_argument("--outfile", type=str, default="atlas.nii.gz")

    return parser.parse_args()


def transform_and_save_npy(nii_path, output_path, crop, norm):
    img = nib.load(nii_path)
    data = img.get_fdata()
    tensor_data = torch.tensor(data).unsqueeze(0)
    crop_data = crop(tensor_data)
    norm_data = norm(crop_data).squeeze(0)
    np.save(output_path, norm_data)


def transform_and_save_npy(nii_path, output_path, transforms):
    subject = tio.Subject(img=tio.ScalarImage(nii_path))
    subject = transforms(subject)
    data = subject.img.data.squeeze(0).numpy()  # Remove channel dimension
    np.save(output_path, data)


def process_nifti_files(root_dir, npy_folder, transforms):
    nii_files = [f for f in os.listdir(root_dir) if f.endswith("_deskulled.nii.gz")]
    for nii_file in nii_files:
        nii_path = os.path.join(root_dir, nii_file)
        npy_file = nii_file.replace("_deskulled.nii.gz", "") + ".npy"
        output_path = os.path.join(npy_folder, npy_file)

        if os.path.exists(output_path):
            print(f"Skipping {npy_file}, already exists.")
            continue

        transform_and_save_npy(nii_path, output_path, transforms)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    args = parse_args()

    input_folder = (
        args.input_folder
        or f"/mnt/bulk-neptune/radhika/project/images/{args.cohort}/nifti_deskull/"
    )
    output_folder = (
        args.output_folder
        or f"/mnt/bulk-neptune/radhika/project/images/{args.cohort}/npy{args.img_size}/"
    )
    os.makedirs(output_folder, exist_ok=True)

    # Full transform pipeline
    transforms = tio.Compose(
        [
            tio.Resample((1, 1, 1)),  # Resample to 1mm isotropic
            tio.CropOrPad(
                (args.crop_size, args.crop_size, args.crop_size)
            ),  # Crop/Pad to 180³
            tio.Resize(
                (args.img_size, args.img_size, args.img_size)
            ),  # Downscale to 96³
            tio.ZNormalization(),  # Normalize intensity
        ]
    )

    # Process and save files
    if args.bids:
        query = {}
        bids_fields = ["subject", "session", "space", "label", "suffix", "extension"]
        for f in bids_fields:
            val = getattr(args, f, None)
            if val is not None:
                query[f] = val

        process_bids_dir(
            input_folder, query, npy_folder=output_folder, transforms=transforms
        )
    else:
        process_nifti_files(input_folder, output_folder, transforms)
    # Process
    # process_nifti_files(input_folder, output_folder, transforms)

    # Report
    npy_count = len([f for f in os.listdir(output_folder) if f.endswith(".npy")])
    print(f"Total .npy files in {output_folder}: {npy_count}")
