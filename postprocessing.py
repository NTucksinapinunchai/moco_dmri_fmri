# -*- coding: utf-8 -*-

"""
Post-processing script for preparing metric evaluation inputs. This script prepares the necessary intermediate files for computing
evaluation metrics (RMSE, Dice, tSNR, DVARS) after motion correction using functions from the Spinal Cord Toolbox (SCT).

Pipeline Overview:
------------------
For dMRI data:
    - Separate b0 and dwi volumes to get mean dwi
    - Perform spinal cord segmentation in each data type (raw, augmented, motion-corrected)

For fMRI data:
    - Compute mean across time to get mean fmri
    - Perform spinal cord segmentation in each data type (raw, augmented, motion-corrected)

Usage:
------
In terminal/command line:

    python postprocessing.py /path/to/data mode

Arguments:
    /path/to/data : data directory for training dataset preparation
    mode          : 'dmri' or 'fmri' depending on dataset type
"""

import os
import sys
import glob
import subprocess
import nibabel as nib

from config_loader import config

def process_subject(target, mode):
    """
        Process a single subject or session directory to generate mean
        and segmentation files required for evaluation metrics.
    """
    print(f"\nProcessing {target} in {mode.upper()} mode ...")

    patterns = config[mode]
    subdir = patterns["subdir"]

    # Get three files: raw, moving, moco
    file_types = ["raw", "moving", "moco"]
    nii_dict = {}

    for ftype in file_types:
        nii_files = glob.glob(os.path.join(target, subdir, patterns[ftype]))
        if not nii_files:
            print(f"No {ftype} NIfTI found in {target}")
            continue
        nii_dict[ftype] = nii_files[0]

    if not nii_dict:
        print(f"No valid NIfTI files found for {target}")
        return

    for ftype, input_img in nii_dict.items():
        print(f"\n=== Processing {ftype.upper()} file ===")
        nii = nib.load(input_img)
        T = nii.shape[3] if nii.ndim == 4 else 1
        print(f"Found NIfTI: {input_img}, shape: {nii.shape}")

        # Prefix extraction
        root = os.path.basename(input_img)
        parts = root.split("_")
        if len(parts) > 1 and parts[1].startswith("ses-"):
            prefix = "_".join(parts[:2])
        else:
            prefix = parts[0]

        out_dir = os.path.join(target, subdir)
        # -----------------------------
        # dMRI processing
        # -----------------------------
        if mode == "dmri":
            files = patterns[ftype].split("*")[0]

            # Find bvec/bval
            bvec_files = glob.glob(os.path.join(target, subdir, "*.bvec"))
            bval_files = glob.glob(os.path.join(target, subdir, "*.bval"))
            if not bvec_files or not bval_files:
                print(f"Missing bvec/bval in {target}")
                continue
            bvec_file, bval_file = bvec_files[0], bval_files[0]

            # Separate b0 and dwi to get mean dwi
            SCT_SEP = [
                "sct_dmri_separate_b0_and_dwi",
                "-i", input_img,
                "-bvec", bvec_file,
                "-bval", bval_file,
                "-ofolder", out_dir
            ]

            subprocess.run(SCT_SEP, check=True)
            print("Separation Success!")

            # Remove specific unnecessary files
            remove_patterns = [
                "*_dwi_b0.nii.gz",
                "*_dwi_dwi.nii.gz"
            ]

            for pattern in remove_patterns:
                for f in glob.glob(os.path.join(out_dir, pattern)):
                    try:
                        os.remove(f)
                        print(f"Removed: {os.path.basename(f)}")
                    except Exception as e:
                        print(f"Could not remove {f}: {e}")
            print("Clean-up Success!")

            mean_files = glob.glob(os.path.join(out_dir, f"{files}*_dwi_mean.nii.gz"))
            if not mean_files:
                print(f"No mean DWI found for {ftype}")
                continue
            mean_img = mean_files[0]

            # Segment spinal cord
            SCT_SEG = [
                "sct_deepseg",
                "spinalcord",
                "-i", mean_img
            ]

            subprocess.run(SCT_SEG, check=True)
            print("Segmentation Success!")

        # -----------------------------
        # fMRI processing
        # -----------------------------
        elif mode == "fmri":
            base = os.path.basename(input_img)
            if base.endswith(".nii.gz"):
                files = base[:-7]  # remove 7 characters for ".nii.gz"
            else:
                files = os.path.splitext(base)[0]
            mean_img = os.path.join(out_dir, f"{files}_mean.nii.gz")
            seg_img = os.path.join(out_dir, f"{files}_seg.nii.gz")

            # Mean over time
            SCT_MATH = [
                "sct_maths",
                "-i", input_img,
                "-mean", "t",
                "-o", mean_img
            ]
            subprocess.run(SCT_MATH, check=True, cwd=out_dir)
            print("Mean Success!")

            # Segment spinal cord
            SCT_SEG = [
                "sct_deepseg",
                "sc_epi",
                "-i", mean_img,
                "-o", seg_img
            ]

            subprocess.run(SCT_SEG, check=True)
            print("Segmentation Success!")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python postprocessing.py <data_dir> <dmri|fmri>")
        sys.exit(1)

    data_dir = sys.argv[1]
    mode = sys.argv[2].lower()

    # Get all subdirectories inside data_dir
    subjects = sorted([
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not subjects:
        print(f"No subject directories found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(subjects)} subjects in {data_dir}")
    for subj in subjects:
        process_subject(subj, mode)