"""
Pre-processing script for dMRI and fMRI data using function of Spinal Cord Toolbox (SCT)
For dMRI data   - Separate b0 and dwi
                - Segmentation
                - Mask creation
                and duplicate mean b0 and mean dwi volumes into 4D (H,W,D,T)
For fMRI data   - Mean across time
                - Segmentation
                - Mask creation

Usage: in terminal/command line
    python preprocess_data.py /path/to/data mode
    - /path/to/data --> directory of dataset
    - mode --> dmri or fmri :depending on dataset
"""

import os
import sys
import glob
import subprocess
import numpy as np
import nibabel as nib

def process_subject(target, mode):
    print(f"\nProcessing {target} in {mode.upper()} mode ...")

    if mode == "dmri":
        # ---------------------------
        # Find NIfTI + bvec + bval
        # ---------------------------
        nii_files = glob.glob(os.path.join(target, "dwi", "sub-*dwi.nii.gz"))
        if not nii_files:
            print(f"No dwi NIfTI found in {target}")
            return
        input_img = nii_files[0]
        nii = nib.load(input_img)
        T = nii.shape[3]
        print("Found NIfTI:", input_img)

        bvec_files = glob.glob(os.path.join(target, "dwi", "*.bvec"))
        if not bvec_files:
            print(f"No bvec file found in {target}")
            return
        bvec_file = bvec_files[0]

        bval_files = glob.glob(os.path.join(target, "dwi", "*.bval"))
        if not bval_files:
            print(f"No bval file found in {target}")
            return
        bval_file = bval_files[0]

        out_dir = os.path.join(target, "dwi")

        # ---------------------------
        # Separate b0 and dwi
        # ---------------------------
        SCT_SEP = [
            "sct_dmri_separate_b0_and_dwi",
            "-i", input_img,
            "-bvec", bvec_file,
            "-bval", bval_file,
            "-ofolder", out_dir
        ]
        subprocess.run(SCT_SEP, check=True)
        print("Separation Success!")

        # mean dwi
        nii_files = glob.glob(os.path.join(target, "dwi", "sub-*dwi_mean.nii.gz"))
        if not nii_files:
            print(f"No mean dwi found in {target}")
            return
        mean_img = nii_files[0]
        print("Found mean dwi:", mean_img)

    elif mode == "fmri":
        # ---------------------------
        # Find NIfTI
        # ---------------------------
        nii_files = glob.glob(os.path.join(target, "func", "sub-*bold.nii.gz"))
        if not nii_files:
            print(f"No fMRI NIfTI found in {target}")
            return
        input_img = nii_files[0]
        nii = nib.load(input_img)
        T = nii.shape[3]
        print("Found NIfTI:", input_img)

        root = os.path.basename(input_img)
        parts = root.split("_")
        if len(parts) > 1 and parts[1].startswith("ses-"):
            prefix = "_".join(parts[:2])
        else:
            prefix = parts[0]

        func_dir = os.path.join(target, "func")
        mean_img = os.path.join(func_dir, f"{prefix}_fmri_mean.nii.gz")
        out_dir = os.path.join(target, "func")

        # ---------------------------
        # Mean fMRI across time
        # ---------------------------
        SCT_MATH = [
            "sct_maths",
            "-i", input_img,
            "-mean", "t",
            "-o", mean_img
        ]
        subprocess.run(SCT_MATH, check=True, cwd=func_dir)
        print("Mean Success!")

    else:
        raise ValueError("Mode must be 'dmri' or 'fmri'")

    # ---------------------------
    # Segmentation spinal cord
    # ---------------------------
    SCT_SEG = [
        "sct_deepseg",
        "spinalcord",
        "-i", mean_img,
    ]

    try:
        subprocess.run(SCT_SEG, check=True)
        print("Segmentation Success!")
    except subprocess.CalledProcessError as e:
        print("Error", e)

    # segmentation output
    seg_files = glob.glob(os.path.join(os.path.dirname(mean_img), "*seg.nii.gz"))
    if not seg_files:
        print(f"No seg file produced for {target}")
        return
    seg_img = seg_files[0]

    # ---------------------------
    # Create MASK along spinal cord
    # ---------------------------
    SCT_MASK = [
        "sct_create_mask",
        "-i", mean_img,
        "-p", f"centerline,{seg_img}",
        "-size", "35mm"
    ]

    try:
        subprocess.run(SCT_MASK, check=True, cwd=out_dir)
        print("Mask Success!")
    except subprocess.CalledProcessError as e:
        print("Error", e)

    # ---------------------------
    # Duplicate along time point
    # ---------------------------
    if mode == "dmri":
        nii_files = glob.glob(os.path.join(target, "dwi", "sub-*b0_mean.nii.gz"))
        if not nii_files:
            print(f"No mean b0 found in {target}")
            return
        mean_b0_img = nii_files[0]

        data = nii.get_fdata()
        affine = nii.affine
        header = nii.header

        mean_b0_data = nib.load(mean_b0_img).get_fdata()
        mean_dwi_data = nib.load(mean_img).get_fdata()
        bvals = np.loadtxt(bval_file)

        corrected = np.zeros_like(data)
        for t in range(T):
            if bvals[t] < 50:  # b0
                corrected[..., t] = mean_b0_data
            else:  # dwi
                corrected[..., t] = mean_dwi_data

        basename_noext = os.path.basename(input_img).replace("_dwi.nii.gz", "")
        out_corrected = os.path.join(target, "dwi", f"dup_{basename_noext}_fixed.nii.gz")
        nib.save(nib.Nifti1Image(corrected, affine, header), out_corrected)
        print("Saved corrected file:", out_corrected)

    elif mode == "fmri":
        affine = nii.affine
        header = nii.header
        mean_fmri_data = nib.load(mean_img).get_fdata()

        # Save only the mean fMRI as 3D fixed reference
        out_fixed = os.path.join(target, "func", f"{prefix}_fixed.nii.gz")
        nib.save(nib.Nifti1Image(mean_fmri_data, affine, header), out_fixed)
        print("Saved fixed file:", out_fixed)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_data.py <data_dir> <dwi|fmri>")
        sys.exit(1)

    data_dir = sys.argv[1]
    mode = sys.argv[2].lower()
    subfolders = sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])

    for sub in subfolders:
        # session-aware
        sessions = sorted([f.path for f in os.scandir(sub) if f.is_dir() and f.name.startswith("ses-")])
        targets = sessions if sessions else [sub]

        for target in targets:
            process_subject(target, mode)