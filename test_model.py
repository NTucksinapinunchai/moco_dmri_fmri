# -*- coding: utf-8 -*-
"""
Inference script used to generate the motion-corrected 4D volumes from DenseRigidDSRReg model.

This script loads raw NIfTI files (moving, fixed, mask) directly and applies the trained
DenseRigidDSRReg checkpoint to perform motion correction. It outputs the corrected
4D volume as well as the translation parameters (Tx, Ty)

Features:
---------
- Supports both dMRI and fMRI datasets (detected automatically from input path).
- Loads moving/fixed/mask volumes from BIDS-like folder structure.
- Performs inference with the trained checkpoint.
- Saves motion-corrected volumes and displacement fields as NIfTI files.
- Reports inference timing statistics across subjects.

Outputs:
---------
- Motion-corrected 4D NIfTI (moco_*)
- 2 rigid translation parameter: Tx, Ty  (each [D,T])
- Timing statistics

Usage:
------
In terminal/command line:

    python test_model.py /path/to/data /path/to/trained_weight.ckpt

Arguments:
    /path/to/data                   : directory of prepared dataset
    /path/to/trained_weight.ckpt    : path directly to trained checkpoint file
"""

import os
import sys
import glob
import time
import torch

torch.set_float32_matmul_precision("medium")

import numpy as np
import nibabel as nib

from config_loader import config
from moco_main import DenseRigidReg, RigidWarp
from skimage.exposure import match_histograms

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Main inference
# -----------------------------
def main(data_dir, ckpt_path):
    """
    Run inference on all subjects/sessions in the dataset and save
    motion-corrected 4D NIfTI volumes along with translation maps (Tx, Ty).
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    mode = "dmri" if "dmri_dataset" in data_dir else "fmri"
    patterns = config[mode]

    # -----------------------------
    # Load model
    # -----------------------------
    model = DenseRigidReg.load_from_checkpoint(ckpt_path, map_location=device)
    model = model.to(device)
    model.eval()
    model.warp = RigidWarp(mode="bilinear")

    # -----------------------------
    # Detect subjects and files automatically
    # -----------------------------
    subjects = sorted(glob.glob(os.path.join(data_dir, "sub-*")))
    timings = []

    for sub in subjects:
        sessions = sorted(glob.glob(os.path.join(sub, "ses-*")))
        targets = sessions if sessions else [sub]

        for target in targets:
            print(f"\nProcessing {target} ...")

            raw_files = glob.glob(os.path.join(target, patterns["subdir"], patterns["raw"]))
            moving_files = glob.glob(os.path.join(target, patterns["subdir"], patterns["moving"]))
            fixed_files = glob.glob(os.path.join(target, patterns["subdir"], patterns["fixed"]))
            mask_files = glob.glob(os.path.join(target, patterns["subdir"], patterns["mask"]))
            subdir = patterns["subdir"]
            suffix = patterns["suffix"]

            if not (raw_files and moving_files and fixed_files and mask_files):
                print(f"Missing files in {target}, skipping.")
                continue

            # -----------------------------
            # Load NIfTI data properly
            # -----------------------------
            raw_img = nib.load(raw_files[0])
            moving_img = nib.load(moving_files[0])
            fixed_img = nib.load(fixed_files[0])
            mask_img = nib.load(mask_files[0])

            # keep NIfTI image object for header/affine
            affine = raw_img.affine
            header = raw_img.header

            # extract float data arrays
            ref_data = raw_img.get_fdata().astype(np.float32)
            moving_np = moving_img.get_fdata().astype(np.float32)
            fixed_np = fixed_img.get_fdata().astype(np.float32)
            mask_np = mask_img.get_fdata().astype(np.float32)

            # Add batch/channel dims
            moving = torch.from_numpy(moving_np).unsqueeze(0).unsqueeze(0).to(device)
            fixed = torch.from_numpy(fixed_np).unsqueeze(0).unsqueeze(0).to(device)
            mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

            # -----------------------------
            # Run model inference
            # -----------------------------
            start_time = time.time()
            with torch.no_grad():
                warped_all, Tx_all, Ty_all = model(moving, fixed, mask)

            warped = warped_all.squeeze(0).squeeze(0).cpu().numpy()  # (H,W,D,T)
            Tx = Tx_all.squeeze().cpu().numpy()
            Ty = Ty_all.squeeze().cpu().numpy()

            import cv2
            H, W, D, T = warped.shape
            sharpened = np.copy(warped)
            for t in range(T):
                for d in range(D):
                    img_warped = warped[..., d, t]
                    img_raw = ref_data[..., d, t]
                    mask_slice = mask_np[..., d]

                    if np.count_nonzero(mask_slice) == 0:
                        sharpened[..., d, t] = img_warped
                        continue

                    raw_smooth = cv2.GaussianBlur(img_raw, (0, 0), 0.6)     # 0.6 for dmri, 0.5 for fmri
                    texture = img_raw - raw_smooth
                    out = img_warped + 1.3 * texture
                    lo, hi = np.percentile(img_raw[mask_slice > 0], [0.5, 99.5])
                    out = np.clip(out, lo, hi)

                    sharpened[..., d, t] = out

            sharpened[mask_np == 0] = ref_data[mask_np == 0]

            # # -----------------------------
            # # Build 5D displacement field (H, W, D, T, 3)
            # # -----------------------------
            # H, W, D, T = warped.shape
            # disp_field = np.zeros((H, W, D, T, 3), dtype=np.float32)
            #
            # for t in range(T):
            #     for d in range(D):
            #         # Constant 2D translation per slice
            #         disp_x = np.full((H, W), Tx[d, t], dtype=np.float32)  # shift in x (width)
            #         disp_y = np.full((H, W), Ty[d, t], dtype=np.float32)  # shift in y (height)
            #         disp_z = np.zeros((H, W), dtype=np.float32)
            #
            #         # Assign
            #         disp_field[..., d, t, 0] = disp_x
            #         disp_field[..., d, t, 1] = disp_y
            #         disp_field[..., d, t, 2] = disp_z
            #
            # voxel_size = header.get_zooms()[:3]  # (sx, sy, sz) in mm
            # disp_field_mm = np.zeros_like(disp_field)
            # disp_field_mm[..., 0] = disp_field[..., 0] * voxel_size[0]
            # disp_field_mm[..., 1] = disp_field[..., 1] * voxel_size[1]
            # disp_field_mm[..., 2] = disp_field[..., 2] * voxel_size[2]
            #
            # Histogram matching
            matched = np.zeros_like(sharpened)
            for t in range(warped.shape[-1]):
                if ref_data.ndim == 4:
                    matched[..., t] = match_histograms(sharpened[..., t], ref_data[..., t])
                else:
                    matched[..., t] = match_histograms(sharpened[..., t], ref_data)

            # -----------------------------
            # Save the output
            # -----------------------------
            out_dir = os.path.join(target, subdir)
            prefix = os.path.basename(target)
            nib.save(nib.Nifti1Image(matched, affine, header=header), os.path.join(out_dir, f"moco_{prefix}_{suffix}_ne.nii.gz"))
            nib.save(nib.Nifti1Image(Tx[np.newaxis, np.newaxis, ...], affine, header=header), os.path.join(out_dir, f"{prefix}_Tx.nii.gz"))
            nib.save(nib.Nifti1Image(Ty[np.newaxis, np.newaxis, ...], affine, header=header), os.path.join(out_dir, f"{prefix}_Ty.nii.gz"))
            print(f"Saved outputs to: {out_dir}")

            # -----------------------------
            # Save per-timepoint displacement fields into a folder
            # -----------------------------
            # disp_dir = os.path.join(out_dir, "dispfield")
            # os.makedirs(disp_dir, exist_ok=True)
            #
            # for t in range(T):
            #     disp_t = disp_field_mm[..., t, :]  # (H, W, D, 3)
            #     disp_img = nib.Nifti1Image(disp_t, affine, header=header)
            #     disp_img.header.set_intent('vector', (), '')
            #
            #     nib.save(disp_img, os.path.join(disp_dir, f"{prefix}_dispfield_t{t:04d}.nii.gz"))
            #
            # print(f"Saved displacement fields to: {disp_dir}")

            elapsed = time.time() - start_time
            timings.append(elapsed)
            print(f"Inference completed in {elapsed:.2f} sec")

    # -----------------------------
    # Summary time
    # -----------------------------
    if timings:
        print("\n=== Inference Timing Summary ===")
        print(f"Samples: {len(timings)}")
        print(f"Mean:    {np.mean(timings):.2f} sec")
        print(f"Std:     {np.std(timings):.2f} sec")
        print(f"Min:     {np.min(timings):.2f} sec")
        print(f"Max:     {np.max(timings):.2f} sec")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_model.py <data_dir> <trained_weight>")
        sys.exit(1)

    data_dir = sys.argv[1]  # path to testing dataset
    ckpt_path = sys.argv[2]  # full path to .ckpt file (trained-weight)
    main(data_dir, ckpt_path)