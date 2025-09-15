import os
import glob
import subprocess
import numpy as np
import nibabel as nib

# ---------------------------
# Load data
# ---------------------------
data_dir = "/Users/kenggkkeng/Desktop/single/"

# subfolders = sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])
subfolders = [os.path.join(data_dir, "sub-perform")]

for subfolder in subfolders:
    print(f"Processing {subfolder} ...")

    # Find NIfTI file
    nii_files = glob.glob(os.path.join(subfolder, "dwi", "sub-*dwi.nii.gz"))
    if len(nii_files) == 0:
        raise FileNotFoundError("No NIfTI file found in folder.")
    input_img = nii_files[0]
    nii = nib.load(input_img)
    T = nii.shape[3]
    print("Found NIfTI:", input_img)

    # Find bvec file
    bvec_files = glob.glob(os.path.join(subfolder, "dwi", "*.bvec"))
    if len(bvec_files) == 0:
        raise FileNotFoundError("No bvec file found in folder.")
    bvec_file = bvec_files[0]
    print("Found bvec:", bvec_file)

    # Find bvec file
    bval_files = glob.glob(os.path.join(subfolder, "dwi", "*.bval"))
    if len(bval_files) == 0:
        raise FileNotFoundError("No bval file found in folder.")
    bval_file = bval_files[0]
    print("Found bval:", bvec_file)

    # basename = os.path.basename(input_img)
    out_dir = os.path.join(subfolder, "dwi")

    # ---------------------------
    # Separate b0 ans dwi
    # ---------------------------
    SCT_SEP = [
        "sct_dmri_separate_b0_and_dwi",
        "-i", input_img,
        "-bvec", bvec_file,
        "-bval", bval_file,
        "-ofolder", out_dir
    ]

    try:
        subprocess.run(SCT_SEP, check=True)
        print("Separation Success!")
    except subprocess.CalledProcessError as e:
        print("Error", e)

    # ---------------------------
    # Segmentation spinal cord
    # ---------------------------
    nii_files = glob.glob(os.path.join(subfolder, "dwi", "sub-*dwi_mean.nii.gz"))
    if len(nii_files) == 0:
        raise FileNotFoundError("No NIfTI file found in folder.")
    mean_img = nii_files[0]
    print("Found NIfTI:", mean_img)

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

    # ---------------------------
    # MASK
    # ---------------------------
    nii_files = glob.glob(os.path.join(subfolder, "dwi", "sub-*mean_seg.nii.gz"))
    if len(nii_files) == 0:
        raise FileNotFoundError("No NIfTI file found in folder.")
    seg_img = nii_files[0]
    print("Found NIfTI:", seg_img)

    SCT_MASK = [
        "sct_create_mask",
        "-i", mean_img,
        "-p", f"centerline,{seg_img}",
        "-size", "25mm"
    ]

    try:
        subprocess.run(SCT_MASK, check=True, cwd=out_dir)
        print("Mask Success!")
    except subprocess.CalledProcessError as e:
        print("Error", e)

    # ---------------------------
    # Duplicate dwi_mean along time point
    # ---------------------------
    mean_img = nib.load(mean_img)
    mean_data = mean_img.get_fdata()
    mean_4d = np.repeat(mean_data[..., np.newaxis], T, axis=3)
    dup_nii = nib.Nifti1Image(mean_4d, affine=mean_img.affine, header=mean_img.header)

    basename_noext = os.path.basename(input_img).replace(".nii.gz", "")
    out_4d_mean = os.path.join(subfolder, "dwi", f"dup_{basename_noext}_mean.nii.gz")
    nib.save(dup_nii, out_4d_mean)
    print("Saved:", out_4d_mean)