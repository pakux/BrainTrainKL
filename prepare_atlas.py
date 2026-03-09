#! /usr/bin/env python

import nilearn
import nilearn.datasets
import re
import argparse
import torchio as tio
import numpy as np
import nibabel as nib
from os.path import basename, splitext, join, exists, isfile
import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Atlasfile")
    parser.add_argument(
        "--size",
        type=int,
        default=180,
        help="Image size for cropping/padding (default: 180)",
    )
    parser.add_argument("--atlas_name", type=str, default="AAL")
    parser.add_argument("--outfile", type=str, default="atlas.nii.gz")

    return parser.parse_args()


def transform_and_save_nii(nii_path, transforms, output_path=None, affine=None):
    subject = tio.Subject(img=tio.ScalarImage(nii_path))
    subject = transforms(subject)
    data = subject.img.data.squeeze(0).numpy()  # Remove channel dimension

    # Default affine if none provided
    if affine is None:
        affine = np.eye(4)

    # Save image as NIfTI
    img = nib.Nifti1Image(data, affine)

    # Save image if output_path given
    if not output_path is None:
        nib.save(img, output_path)

    return img


def download_atlas(atlas_name: str) -> str:
    """
    Download an atlas from given String from

    Returns: file_path

    """
    name = atlas_name.strip().lower()

    # Harvard-Oxford cortical probability / maxprob at 2mm
    if name in ("harvard-oxford", "harvardoxford", "harvard_oxford"):
        b = nilearn.datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        return b["maps"] if "maps" in b else b.maps

    if name in ("aal", "aal3"):
        b = nilearn.datasets.fetch_atlas_aal()
        return b["maps"] if "maps" in b else b.maps

    if name in ("msdl",):
        b = nilearn.datasets.fetch_atlas_msdl()
        return b["maps"] if "maps" in b else b.maps

    if name in ("yeo7", "yeo_7", "yeo 7"):
        b = nilearn.datasets.fetch_atlas_yeo_2011(networks=7)
        return b["thick_7"] if "thick_7" in b else b.maps

    if name in ("yeo17", "yeo_17", "yeo 17"):
        b = nilearn.datasets.fetch_atlas_yeo_2011(networks=17)
        return b["thick_17"] if "thick_17" in b else b.maps

    # Schaefer: parse number of parcels
    m = re.search(r"schaefer[_\-\s]?(\d+)", name)
    if m:
        n = int(m.group(1))
        b = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=n, yeo_networks=7)
        return b["maps"] if "maps" in b else b.maps

    if name in ("destrieux", "desikan", "desikan_killiany", "desikan-killiany"):
        b = nilearn.datasets.fetch_atlas_surf_destrieux()
        # surf destrieux returns 'map' paths for surface; try 'map' or 'maps'
        return b.get("map", b.get("maps", None)) or b.map

    # Fallback: try a few other fetchers
    try:
        b = nilearn.datasets.fetch_atlas_basc_multiscale()
        # returns 'symmetry' or 'scale' maps; pick first available nii
        for k in ("maps", "symmetry", "scale_imgs"):
            if k in b:
                val = b[k]
                return (
                    val
                    if isinstance(val, str)
                    else (val[0] if isinstance(val, (list, tuple)) else val)
                )
    except Exception:
        pass

    raise ValueError(f"Unknown or unsupported atlas name: {atlas_name}")


def main(
    atlas_name: str = "AAL", out_file: str = "atlas.nii.gz", size: int | list = 96
):
    """
    Fetch Atlas
    """

    if isfile(atlas_name):
        logging.info(f"Found given atlas as file {atlas_name}")
        atlas_file = atlas_name
    else:
        logging.info(f"No file named {atlas_name} found. Trying to use nilearn fetch")
        atlas_file = download_atlas(atlas_name)

    if type(size) is int:
        size = [size, size, size]

    logging.info(f"resizing to {size}")

    transforms = tio.Compose(
        [
            tio.Resample((1, 1, 1)),  # Resample to 1mm isotropic
            tio.CropOrPad(180),  # Crop/Pad to 180³
            tio.Resize(
                size,
                image_interpolation="nearest",
                label_interpolation="nearest",
            ),  # Downscale to 96³
        ]
    )
    transform_and_save_nii(atlas_file, transforms, out_file)


if __name__ == "__main__":
    args = parse_args()
    main(atlas_name=args.atlas_name, out_file=args.outfile, size=args.size)
