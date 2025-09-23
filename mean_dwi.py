import os
import glob
import subprocess
import numpy as np
import nibabel as nib

# ---------------------------
# Load data
# ---------------------------
data_dir = "/Users/kenggkkeng/Desktop/sourcedata/"

subfolders = sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])
# subfolders = [os.path.join(data_dir, "sub-douglas")]

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

    nii_files = glob.glob(os.path.join(subfolder, "dwi", "sub-*b0_mean.nii.gz"))
    if len(nii_files) == 0:
        raise FileNotFoundError("No NIfTI file found in folder.")
    mean_b0_img = nii_files[0]
    print("Found NIfTI:", mean_b0_img)

    # ---------------------------
    # Duplicate dwi_mean along time point
    # ---------------------------
    nii = nib.load(input_img)
    data = nii.get_fdata()
    affine = nii.affine
    header = nii.header

    # Load mean_b0
    mean_b0_nii = nib.load(mean_b0_img)
    mean_b0_data = mean_b0_nii.get_fdata()

    # Load mean_dwi
    mean_dwi_nii = nib.load(mean_img)
    mean_dwi_data = mean_dwi_nii.get_fdata()

    # Load bval to know which volumes are b0 and which are dwi
    bvals = np.loadtxt(bval_file)

    # Replace volumes according to bval
    corrected = np.zeros_like(data)
    for t in range(T):
        if bvals[t] < 50:  # identify b0
            corrected[..., t] = mean_b0_data
        else:  # identify dwi
            corrected[..., t] = mean_dwi_data

    # Save single corrected 4D file
    basename_noext = os.path.basename(input_img).replace("_dwi.nii.gz", "")
    out_corrected = os.path.join(subfolder, "dwi", f"dup_{basename_noext}_fixed.nii.gz")
    nib.save(nib.Nifti1Image(corrected, affine, header), out_corrected)
    print("Saved:", out_corrected)