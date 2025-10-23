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

    # Override warping for inference
    model.warp = RigidWarp(mode="nearest")      # use nearest-neighbor interpolation for evaluation, Preserves original voxel intensities

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

            if not (moving_files and fixed_files and mask_files):
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
            # Run inference
            # -----------------------------
            start_time = time.time()
            with torch.no_grad():
                warped_all, Tx_all, Ty_all = model(moving, fixed, mask)

            # Convert to numpy
            warped = warped_all.squeeze(0).squeeze(0).cpu().numpy()  # (H,W,D,T)
            Tx = Tx_all.squeeze().cpu().numpy()  # (1,1,D,T)
            Ty = Ty_all.squeeze().cpu().numpy()  # (1,1,D,T)

            # Histogram matching
            matched = np.zeros_like(warped)
            for t in range(warped.shape[-1]):
                if ref_data.ndim == 4:
                    matched[..., t] = match_histograms(warped[..., t], ref_data[..., t])
                else:
                    matched[..., t] = match_histograms(warped[..., t], ref_data)

            # -----------------------------
            # Save the output
            # -----------------------------
            out_dir = os.path.join(target, subdir)
            prefix = os.path.basename(target)
            nib.save(nib.Nifti1Image(matched, affine, header=header), os.path.join(out_dir, f"moco_{prefix}_{suffix}.nii.gz"))
            nib.save(nib.Nifti1Image(Tx[np.newaxis, np.newaxis, ...], affine, header=header), os.path.join(out_dir, f"{prefix}_Tx.nii.gz"))
            nib.save(nib.Nifti1Image(Ty[np.newaxis, np.newaxis, ...], affine, header=header), os.path.join(out_dir, f"{prefix}_Ty.nii.gz"))
            print(f"Saved outputs to: {out_dir}")

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

    data_dir = sys.argv[1]          # path to testing dataset
    ckpt_path = sys.argv[2]         # full path to .ckpt file (trained-weight)
    main(data_dir, ckpt_path)