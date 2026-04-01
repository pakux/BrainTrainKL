import os
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Iterable
import ants
import torchio as tio
import numpy as np
import BrainTrain.config as cfg

# ============================================================
# Globals
# ============================================================

fixed = ants.image_read(cfg.TEMPLATE_PATH)

# ============================================================
# Helpers
# ============================================================

def _strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def _safe_unlink(path_like):
    try:
        p = Path(path_like)
        if p.exists() and p.is_file():
            p.unlink()
            print(f"[CLEAN] Removed {p}")
    except Exception as e:
        print(f"[WARN][CLEAN] Could not remove {path_like}: {e}")


def _cleanup_files(paths: Iterable[str]):
    for p in paths:
        _safe_unlink(p)


def _hd_bet_paths(out_path: Path):
    base = _strip_nii_suffix(out_path.name)
    return {
        "final": out_path,
        "bet": out_path.parent / f"{base}_bet.nii.gz",
        "mask": out_path.parent / f"{base}_mask.nii.gz",
    }


def cleanup_deskull_artifacts():
    folder = Path(cfg.DESKULL_FOLDER)
    if not folder.exists():
        return

    # Remove mask sidecars left by hd-bet.
    for mask_file in folder.glob("*_mask.nii.gz"):
        _safe_unlink(mask_file)

    # Normalize older hd-bet naming to this pipeline's expected output name.
    for bet_file in folder.glob("*_deskulled_bet.nii.gz"):
        final_file = folder / bet_file.name.replace("_deskulled_bet.nii.gz", "_deskulled.nii.gz")
        try:
            if final_file.exists():
                _safe_unlink(bet_file)
            else:
                bet_file.rename(final_file)
                print(f"[CLEAN][BET] Renamed {bet_file.name} -> {final_file.name}")
        except Exception as e:
            print(f"[WARN][CLEAN] Could not normalize {bet_file}: {e}")


def list_nifti_raw():
    files = []
    for p in sorted(Path(cfg.INPUT_FOLDER).iterdir()):
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            files.append(p.name)
    return files

def list_nifti_registered():
    files = []
    for p in sorted(Path(cfg.REG_FOLDER).iterdir()):
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            files.append(p.name)
    return files

def list_nifti_deskulled():
    files = []
    for p in sorted(Path(cfg.DESKULL_FOLDER).iterdir()):
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            files.append(p.name)
    return files


# ============================================================
# Stage 1: DICOM → NIfTI
# ============================================================

def dcm_to_nifti(dcm_dir):
    try:
        dcm_dir = Path(dcm_dir)
        out_dir = Path(cfg.INPUT_FOLDER)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_nii = out_dir / f"{dcm_dir.name}.nii.gz"

        if out_nii.exists():
            print(f"[SKIP][DCM] {out_nii.name}")
            return out_nii.name

        print(f"[INFO][DCM] {dcm_dir.name}")
        cmd = [
            cfg.DCM2NIIX,
            "-z", "y",
            "-f", dcm_dir.name,
            "-o", str(out_dir),
            str(dcm_dir)
        ]
        subprocess.run(cmd, check=True)
        return out_nii.name

    except Exception as e:
        print(f"[ERROR][DCM] {dcm_dir}: {e}")


# ============================================================
# Stage 2: N4 Bias Correction
# ============================================================

def bias_correct(filename):
    try:
        in_path  = Path(cfg.INPUT_FOLDER) / filename
        out_name = f"{_strip_nii_suffix(filename)}_n4.nii.gz"
        out_path = Path(cfg.N4_FOLDER) / out_name

        if out_path.exists():
            print(f"[SKIP][N4] {out_name}")
            return out_name

        print(f"[INFO][N4] {filename}")
        img = ants.image_read(str(in_path))
        corrected = ants.n4_bias_field_correction(img)
        ants.image_write(corrected, str(out_path))
        return out_name

    except Exception as e:
        print(f"[ERROR][N4] {filename}: {e}")


# ============================================================
# Stage 3: Registration
# ============================================================

def register(n4_name):
    try:
        if not n4_name:
            return

        in_path  = Path(cfg.N4_FOLDER) / n4_name
        out_name = n4_name.replace("_n4.nii.gz", "_registered.nii.gz")
        out_path = Path(cfg.REG_FOLDER) / out_name

        if out_path.exists():
            print(f"[SKIP][REG] {out_name}")
            return out_name

        print(f"[INFO][REG] {n4_name}")
        moving = ants.image_read(str(in_path))
        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=cfg.REGISTRATION_TYPE
        )
        ants.image_write(reg["warpedmovout"], str(out_path))

        if getattr(cfg, "CLEANUP_ANTS_TRANSFORMS", True):
            transform_files = []
            transform_files.extend(reg.get("fwdtransforms", []))
            transform_files.extend(reg.get("invtransforms", []))
            _cleanup_files(transform_files)

        return out_name

    except Exception as e:
        print(f"[ERROR][REG] {n4_name}: {e}")


# ============================================================
# Stage 4: Deskulling
# ============================================================

def deskull(reg_name, gpu_id=0):
    try:
        if not reg_name:
            return

        in_path  = Path(cfg.REG_FOLDER) / reg_name
        out_name = reg_name.replace("_registered.nii.gz", "_deskulled.nii.gz")
        out_path = Path(cfg.DESKULL_FOLDER) / out_name

        if out_path.exists():
            print(f"[SKIP][BET] {out_name}")
            return out_name

        print(f"[INFO][BET] {reg_name} (GPU {gpu_id})")
        cmd = f'hd-bet -i "{in_path}" -o "{out_path}" -device cuda:{gpu_id}'
        subprocess.run(cmd, shell=True, check=True)

        bet_paths = _hd_bet_paths(out_path)
        # Some hd-bet versions write *_bet.nii.gz even when a full filename is given.
        if not bet_paths["final"].exists() and bet_paths["bet"].exists():
            bet_paths["bet"].rename(bet_paths["final"])
            print(f"[CLEAN][BET] Renamed {bet_paths['bet'].name} -> {bet_paths['final'].name}")

        if getattr(cfg, "CLEANUP_HD_BET_MASK", True):
            _safe_unlink(bet_paths["mask"])

        return out_name

    except Exception as e:
        print(f"[ERROR][BET] {reg_name}: {e}")


# ============================================================
# Stage 5: NIfTI → NPY (TorchIO)
# ============================================================

def nifti_to_npy(nii_name):
    try:
        nii_path = Path(cfg.DESKULL_FOLDER) / nii_name
        npy_path = Path(cfg.NPY_FOLDER) / nii_name.replace("_deskulled.nii.gz", ".npy")

        if npy_path.exists():
            print(f"[SKIP][NPY] {npy_path.name}")
            return npy_path.name

        transforms = tio.Compose([
            tio.Resample((1, 1, 1)),
            tio.CropOrPad((cfg.CROP_SIZE, cfg.CROP_SIZE, cfg.CROP_SIZE)),
            tio.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.IMG_SIZE)),
            tio.ZNormalization()
        ])

        subject = tio.Subject(img=tio.ScalarImage(str(nii_path)))
        subject = transforms(subject)

        data = subject.img.data.squeeze(0).numpy()
        np.save(npy_path, data)

        print(f"[SAVE][NPY] {npy_path.name}")
        return npy_path.name

    except Exception as e:
        print(f"[ERROR][NPY] {nii_name}: {e}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Create folders
    for folder in [
        cfg.DCM_FOLDER,
        cfg.INPUT_FOLDER,
        cfg.N4_FOLDER,
        cfg.REG_FOLDER,
        cfg.DESKULL_FOLDER,
        cfg.NPY_FOLDER,
    ]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # ---- DICOM → NIfTI ----
    start_stage = cfg.PREPROCESS_START.lower()
    run_n4 = start_stage in {"dcm", "dicom", "nifti", "nifti_raw", "nifti-raw"}
    run_reg = run_n4
    run_deskull = start_stage in {
        "dcm", "dicom", "nifti", "nifti_raw", "nifti-raw", "nifti_reg", "nifti-reg"
    }

    if start_stage in {"dcm", "dicom"}:
        print("\n=== DICOM → NIfTI ===")
        dcm_dirs = [
            Path(cfg.DCM_FOLDER) / d
            for d in os.listdir(cfg.DCM_FOLDER)
            if (Path(cfg.DCM_FOLDER) / d).is_dir()
        ]
        print(f"Total DICOM dirs: {len(dcm_dirs)}")

        with mp.Pool(cfg.N4_PROCESSES) as pool:
            current_files = list(filter(None, pool.map(dcm_to_nifti, dcm_dirs)))
        print(f"\n[DONE] Total NIfTIs: {len(current_files)}")
    elif start_stage in {"nifti", "nifti_raw", "nifti-raw"}:
        print("\n=== NIfTI Raw (skip DICOM) ===")
        current_files = list_nifti_raw()
        print(f"Total NIfTIs: {len(current_files)}")
    elif start_stage in {"nifti_reg", "nifti-reg"}:
        print("\n=== NIfTI Registered (skip DICOM and N4) ===")
        current_files = list_nifti_registered()
        print(f"Total Registered NIfTIs: {len(current_files)}")
    elif start_stage in {"nifti_deskulled", "nifti-deskulled"}:
        print("\n=== NIfTI Deskulled (skip DICOM, N4, Registration, Deskulling) ===")
        current_files = list_nifti_deskulled()
        print(f"Total Deskulled NIfTIs: {len(current_files)}")
    else:
        raise ValueError(f"Unsupported PREPROCESS_START: {cfg.PREPROCESS_START}")

    # ---- N4 ----
    if run_n4:
        print("\n=== N4 Bias Correction ===")
        with mp.Pool(cfg.N4_PROCESSES) as pool:
            current_files = list(filter(None, pool.map(bias_correct, current_files)))
        print(f"\n[DONE] Total N4 Corrected: {len(current_files)}")
    else:
        print("\n=== N4 Bias Correction (skipped) ===")

    # ---- Registration ----
    if run_reg:
        print("\n=== Registration ===")
        with mp.Pool(cfg.REG_PROCESSES) as pool:
            current_files = list(filter(None, pool.map(register, current_files)))
        print(f"\n[DONE] Total Registered: {len(current_files)}")
    else:
        print("\n=== Registration (skipped) ===")

    # ---- Deskulling ----
    if run_deskull:
        print("\n=== Deskulling ===")
        deskulled_files = []
        for f in current_files:
            out = deskull(f, gpu_id=cfg.GPU_ID)
            if out:
                deskulled_files.append(out)
        current_files = deskulled_files
        print(f"\n[DONE] Total Deskulled: {len(current_files)}")
    else:
        print("\n=== Deskulling (skipped) ===")

    # ---- TorchIO → NPY ----
    print("\n=== TorchIO → NPY ===")
    with mp.Pool(cfg.NUM_WORKERS) as pool:
        npy_files = list(filter(None, pool.map(nifti_to_npy, current_files)))

    print(f"\n[DONE] Total NPYS: {len(npy_files)}")

    if getattr(cfg, "CLEANUP_HD_BET_MASK", True):
        cleanup_deskull_artifacts()
