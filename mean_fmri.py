import os
import glob
import subprocess
import numpy as np
import nibabel as nib

# ---------------------------
# Load data
# ---------------------------
data_dir = "/Users/kenggkkeng/Desktop/fmri_dataset/"

subfolders = sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])
# subfolders = [os.path.join(data_dir, "")]

for sub in subfolders:
    sessions = sorted([f.path for f in os.scandir(sub) if f.is_dir() and f.name.startswith("ses-")])
    targets = sessions if sessions else [sub]

    for target in targets:
        print(f"Processing {target} ...")

        # Find NIfTI file
        nii_files = glob.glob(os.path.join(target, "func", "sub-*bold.nii.gz"))
        if len(nii_files) == 0:
            raise FileNotFoundError("No NIfTI file found in folder.")
        input_img = nii_files[0]
        nii = nib.load(input_img)
        T = nii.shape[3]
        print("Found NIfTI:", input_img)

        root = os.path.basename(input_img)
        parts = root.split("_")
        if parts[1].startswith("ses-"):
            prefix = "_".join(parts[:2])
        else:
            prefix = parts[0]

        func_dir = os.path.join(target, "func")
        mean_img = os.path.join(func_dir, f"{prefix}_fmri_mean.nii.gz")

        # ---------------------------
        # Mean fMRI
        # ---------------------------
        SCT_MATH = [
            "sct_maths",
            "-i", input_img,
            "-mean", "t",
            "-o", mean_img
        ]

        try:
            subprocess.run(SCT_MATH, check=True, cwd=func_dir)
            print("Mean Success!")
        except subprocess.CalledProcessError as e:
            print("Error", e)

        # ---------------------------
        # Segmentation spinal cord
        # ---------------------------
        nii_files = glob.glob(os.path.join(target, "anat", "sub-*w.nii.gz"))
        if len(nii_files) == 0:
            raise FileNotFoundError("No NIfTI file found in folder.")
        t2_img = nii_files[0]
        print("Found NIfTI:", t2_img)

        SCT_SEG = [
            "sct_deepseg",
            "spinalcord",
            "-i", t2_img,
        ]

        try:
            subprocess.run(SCT_SEG, check=True)
            print("Segmentation Success!")
        except subprocess.CalledProcessError as e:
            print("Error", e)

        # ---------------------------
        # Registration Multimodal
        # ---------------------------
        nii_files = glob.glob(os.path.join(target, "anat", "sub-*_seg.nii.gz"))
        if len(nii_files) == 0:
            raise FileNotFoundError("No NIfTI file found in folder.")
        seg_img = nii_files[0]
        print("Found NIfTI:", seg_img)

        anat_dir = os.path.join(target, "anat")

        SCT_REG = [
            "sct_register_multimodal",
            "-i", seg_img,
            "-d", mean_img,
            "-identity", "1",
            "-ofolder", anat_dir
        ]

        try:
            subprocess.run(SCT_REG, check=True)
            print("Register Success!")
        except subprocess.CalledProcessError as e:
            print("Error", e)

        # ---------------------------
        # MASK
        # ---------------------------
        nii_files = glob.glob(os.path.join(target, "anat", "sub-*seg_reg.nii.gz"))
        if len(nii_files) == 0:
            raise FileNotFoundError("No NIfTI file found in folder.")
        seg_reg_img = nii_files[0]
        print("Found NIfTI:", seg_reg_img)

        SCT_MASK = [
            "sct_create_mask",
            "-i", input_img,
            "-p", f"centerline,{seg_reg_img}",
            "-size", "35mm"
        ]

        try:
            subprocess.run(SCT_MASK, check=True, cwd=func_dir)
            print("Mask Success!")
        except subprocess.CalledProcessError as e:
            print("Error", e)

        # ---------------------------
        # Duplicate fmri_mean along time point
        # ---------------------------
        nii = nib.load(input_img)
        data = nii.get_fdata()
        affine = nii.affine
        header = nii.header

        # Load mean_dwi
        mean_fmri_nii = nib.load(mean_img)
        mean_fmri_data = mean_fmri_nii.get_fdata()

        # Replace volumes according to bval
        corrected = np.zeros_like(data)
        for t in range(T):
            corrected[..., t] = mean_fmri_data

        # Save single corrected 4D file
        out_corrected = os.path.join(target, "func", f"dup_{prefix}_mean_fixed.nii.gz")
        nib.save(nib.Nifti1Image(corrected, affine, header), out_corrected)
        print("Saved:", out_corrected)