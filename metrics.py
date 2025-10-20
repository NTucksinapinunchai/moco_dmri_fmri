# -*- coding: utf-8 -*-
"""
Metric evaluation script for motion-corrected results.
This script evaluates motion correction quality for dMRI and fMRI data
by computing quantitative metrics before and after registration.

Features:
------------
- Automatically locates the moving, fixed, warped, mask, and segmentation files
  using patterns defined in config.yaml.
- Computes:
    • 4D RMSE (Root Mean Squared Error) per timepoint inside the mask
    • 3D Dice coefficient between segmentations (before vs. after)
    • fMRI-only metrics: temporal SNR (tSNR) and DVARS
- Aggregates all subject/session metrics into a summary CSV

Usage:
------
In terminal/command line:

    python metrics.py /path/to/dataset mode

Arguments:
    /path/to/dataset : path to dmri_dataset or fmri_dataset folder
    mode          : 'dmri' or 'fmri' depending on dataset type
"""

import os
import sys
import csv
import glob
import numpy as np
import nibabel as nib
import torch
torch.set_float32_matmul_precision("medium")

from monai.metrics import DiceMetric
from moco_main import normalize_volume
from config_loader import config

# -----------------------------
# Helper functions
# -----------------------------
def load_nifti(path):
    """
       Load a NIfTI image as numpy array.
    """
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data, img.affine, img.header

def to_tensor(arr, add_dims=True):
    """
        Convert a numpy array to a torch tensor with batch/channel dimensions.
    """
    t = torch.tensor(arr, dtype=torch.float32)
    if add_dims:
        t = t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
    return t

def compute_rmse(pred, target, mask):
    """
    Compute Root Mean Squared Error (RMSE) within a binary mask.
    """
    diff = (pred - target) ** 2
    masked = diff * mask
    denom = mask.sum()
    return torch.sqrt(masked.sum() / (denom + 1e-6)).item()

def compute_tsnr(data, mask):
    """
    Compute temporal Signal-to-Noise Ratio (tSNR).
    """
    mean_t = np.mean(data, axis=-1)
    std_t = np.std(data, axis=-1)
    tsnr_map = np.divide(mean_t, std_t, out=np.zeros_like(mean_t), where=std_t > 0)
    mean_tsnr = np.mean(tsnr_map[mask > 0])
    return tsnr_map, mean_tsnr

def compute_dvars(data, mask):
    """
    Compute DVARS time series and mean value.
    """
    masked = data[mask > 0]
    norm_data = masked / np.mean(masked, axis=0, keepdims=True)
    diff = np.diff(norm_data, axis=1)
    dvars = np.sqrt(np.mean(diff ** 2, axis=0))
    mean_dvars = np.mean(dvars)
    return dvars, mean_dvars

# -----------------------------
# Main evaluation
# -----------------------------
def main(data_dir, mode):
    """
        Main evaluation routine for computing RMSE, Dice, tSNR, and DVARS.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if mode not in ["dmri", "fmri"]:
        raise ValueError("Mode must be either 'dmri' or 'fmri'")

    patterns = config[mode]
    subjects = sorted(glob.glob(os.path.join(data_dir, "sub-*")))
    print(f"Found {len(subjects)} subjects in {mode} dataset")

    results = []  # store per-subject metrics

    for sub in subjects:
        sessions = sorted(glob.glob(os.path.join(sub, "ses-*")))
        targets = sessions if sessions else [sub]

        for target in targets:
            subdir = os.path.join(target, patterns["subdir"])
            prefix = os.path.basename(target.rstrip("/"))

            print(f"\nProcessing {target}")

            raw_files = glob.glob(os.path.join(subdir, patterns["raw"]))
            moving_files = glob.glob(os.path.join(subdir, patterns["moving"]))
            warped_files = glob.glob(os.path.join(subdir, patterns["moco"]))
            mask_files = glob.glob(os.path.join(subdir, patterns["mask"]))

            seg_r_files = glob.glob(os.path.join(subdir, "sub-*_seg.nii.gz"))
            seg_m_files = glob.glob(os.path.join(subdir, "aug_*_seg.nii.gz"))
            seg_w_files = glob.glob(os.path.join(subdir, "moco_*_seg.nii.gz"))

            if not (moving_files and raw_files and warped_files and mask_files):
                print(f"Missing required files in {target}, skipping.")
                continue

            # -----------------------------
            # Load data
            # -----------------------------
            raw, _, _ = load_nifti(raw_files[0])
            moving, _, _ = load_nifti(moving_files[0])
            warped, aff, hdr = load_nifti(warped_files[0])
            mask, _, _ = load_nifti(mask_files[0])

            mask_t = to_tensor(mask)

            # -----------------------------
            # RMSE (root mean squared error across timepoints; per-t reference + normalization)
            # -----------------------------
            T = moving.shape[-1]
            rmse_before, rmse_after = [], []

            for t in range(T):
                mov_t = to_tensor(moving[..., t])
                war_t = to_tensor(warped[..., t])
                raw_t = to_tensor(raw[..., t])

                # percentile-normalize each volume inside the mask
                mov_t_n = normalize_volume(mov_t, mask_t)
                war_t_n = normalize_volume(war_t, mask_t)
                raw_t_n = normalize_volume(raw_t, mask_t)

                rb = compute_rmse(mov_t_n, raw_t_n, mask_t)
                ra = compute_rmse(war_t_n, raw_t_n, mask_t)
                rmse_before.append(rb)
                rmse_after.append(ra)

            mean_rmse_before = np.mean(rmse_before)
            mean_rmse_after = np.mean(rmse_after)
            print(f"Mean RMSE before={mean_rmse_before:.4f}, after={mean_rmse_after:.4f}")

            # -----------------------------
            # DICE
            # -----------------------------
            dice_before, dice_after = np.nan, np.nan
            if seg_m_files and seg_r_files and seg_w_files:
                seg_r, _, _ = load_nifti(seg_r_files[0])
                seg_m, _, _ = load_nifti(seg_m_files[0])
                seg_w, _, _ = load_nifti(seg_w_files[0])

                dice_metric = DiceMetric(include_background=True, reduction="mean")
                seg_r_t = to_tensor(seg_r)
                seg_m_t = to_tensor(seg_m)
                seg_w_t = to_tensor(seg_w)

                dice_before = dice_metric(seg_m_t, seg_r_t).item()
                dice_after  = dice_metric(seg_w_t, seg_r_t).item()

                print(f"DICE before={dice_before:.4f}, after={dice_after:.4f}")
            else:
                print("No segmentation files found, skipping DICE.")

            # -----------------------------
            # fMRI metrics (tSNR, DVARS)
            # -----------------------------
            if mode == "fmri":
                print("Computing tSNR and DVARS ...")
                _, tsnr_before = compute_tsnr(moving, mask)
                _, tsnr_after  = compute_tsnr(warped, mask)
                _, dvars_before = compute_dvars(moving, mask)
                _, dvars_after  = compute_dvars(warped, mask)

                print(f"tSNR before={tsnr_before:.2f}, after={tsnr_after:.2f}")
                print(f"DVARS before={dvars_before:.4f}, after={dvars_after:.4f}")
            else:
                tsnr_before = tsnr_after = np.nan
                dvars_before = dvars_after = np.nan

            # -----------------------------
            # Store subject metrics
            # -----------------------------
            results.append({
                "subject": prefix,
                "mean_rmse_before": mean_rmse_before,
                "mean_rmse_after": mean_rmse_after,
                "dice_before": dice_before,
                "dice_after": dice_after,
                "tsnr_before": tsnr_before,
                "tsnr_after": tsnr_after,
                "dvars_before": dvars_before,
                "dvars_after": dvars_after,
            })

    # -----------------------------
    # Save combined summary CSV
    # -----------------------------
    summary_csv = os.path.join(data_dir, f"{mode}_metrics.csv")
    print(f"\nSaving combined metrics summary → {summary_csv}")

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "mean_rmse_before", "mean_rmse_after",
                "dice_before", "dice_after",
                "tsnr_before", "tsnr_after",
                "dvars_before", "dvars_after",
            ],
        )
        for r in results:
            for k, v in r.items():
                if isinstance(v, (float, np.floating)):
                    if np.isnan(v):
                        r[k] = "nan"
                    else:
                        r[k] = f"{v:.4f}"

        writer.writeheader()
        writer.writerows(results)

    print(f"Combined metrics saved → {summary_csv}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python metrics.py <dataset_dir> <mode>")
        sys.exit(1)

    data_dir = sys.argv[1]
    mode = sys.argv[2].lower()
    main(data_dir, mode)