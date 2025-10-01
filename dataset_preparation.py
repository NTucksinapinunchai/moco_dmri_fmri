"""
Preparation script used to create dataset.json of dMRI and fMRI dataset for training/validation the DL model.
- the script will convert nifti file into tensor in .pt for training and validation purposes

Usage: in terminal/command line
    python dataset_preparation.py /path/to/dataset mode
    - /path/to/data --> directory of dataset
    - mode --> dmri or fmri :depending on dataset
"""

import os
import sys
import json
import random
import torch
import shutil
import numpy as np
import nibabel as nib

from glob import glob

# -----------------------------
# Helper: load NIfTI → torch
# -----------------------------
def load_as_tensor(nii_path, add_channel=True):
    img = nib.load(nii_path)
    data = img.get_fdata().astype(np.float32)
    tensor = torch.from_numpy(data)
    if add_channel:  # add leading channel dimension
        tensor = tensor.unsqueeze(0)
    return tensor, img.affine.astype(np.float32)

# -----------------------------
# Build dataset entries
# -----------------------------
def build_entries(subject_list, split_name, base_dir, target_dir, mode):
    entries = []
    dataset_root = f"{mode}_dataset"
    for sub_ses in subject_list:
        if "_ses-" in sub_ses:
            sub, ses = sub_ses.split("_", 1)
            if mode == "dmri":
                data_folder = os.path.join(base_dir, sub, ses, "dwi")
            else: # fmri
                data_folder = os.path.join(base_dir, sub, ses, "func")
            out_sub = sub_ses
        else:
            sub = sub_ses
            if mode == "dmri":
                data_folder = os.path.join(base_dir, sub, "dwi")
            else: # fmri
                data_folder = os.path.join(base_dir, sub, "func")
            out_sub = sub

        # ---------------------------
        # File patterns
        # ---------------------------
        if mode == "dmri":
            moving_files = glob(os.path.join(data_folder, "aug_*dwi.nii.gz"))
        else:  # fmri
            moving_files = glob(os.path.join(data_folder, "aug_*bold.nii.gz"))

        for moving_path in moving_files:
            if mode == "dmri":
                fixed_path = moving_path.replace("aug_", "dup_").replace("_dwi.nii.gz", "_fixed.nii.gz")
                mask_files = glob(os.path.join(data_folder, "mask_*.nii.gz"))
                mask_path = mask_files[0]
                out_folder = os.path.join(target_dir, "dmri_dataset", split_name, out_sub, "dwi")
                extra_pattern = os.path.join(data_folder, f"{sub}_dwi.nii.gz")
            else:  # fmri
                fixed_path = moving_path.replace("aug_", "").replace("_bold.nii.gz", "_fixed.nii.gz")
                mask_files = glob(os.path.join(data_folder, "mask_*.nii.gz"))
                mask_path = mask_files[0]
                out_folder = os.path.join(target_dir, "fmri_dataset", split_name, out_sub, "func")
                extra_pattern = os.path.join(data_folder, f"{sub}_*_bold.nii.gz")

            os.makedirs(out_folder, exist_ok=True)

            # Convert and save
            moving, affine = load_as_tensor(moving_path, add_channel=True)
            fixed, _ = load_as_tensor(fixed_path, add_channel=True)
            mask, _ = load_as_tensor(mask_path, add_channel=True)

            for src in [moving_path, fixed_path, mask_path]:
                if src and os.path.exists(src):
                    shutil.copy2(src, out_folder)

            # ---------------------------
            # Extra file (only for testing)
            # ---------------------------
            if split_name == "testing" and os.path.exists(extra_pattern):
                shutil.copy2(extra_pattern, out_folder)
                print(f"Copied extra file: {extra_pattern}")

            # Save as .pt
            pt_filename = os.path.basename(moving_path).replace("aug_", "paired_").replace(".nii.gz", ".pt")
            pt_path = os.path.join(out_folder, pt_filename)
            torch.save({"moving": moving, "fixed": fixed, "mask": mask, "affine": affine}, pt_path)

            # JSON entry
            rel_to_dataset = os.path.relpath(pt_path, os.path.join(target_dir, dataset_root))
            entry = {"data": rel_to_dataset}
            entries.append(entry)
    entries = sorted(entries, key=lambda e: e["data"])
    return entries

def main(base_dir, mode):
    target_dir = os.path.join(base_dir, "prepared")
    os.makedirs(target_dir, exist_ok=True)

    # -----------------------------
    # Collect subjects
    # -----------------------------
    all_subjects = []
    for sub in os.listdir(base_dir):
        sub_path = os.path.join(base_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        ses_folders = [d for d in os.listdir(sub_path) if d.startswith("ses-")]
        if ses_folders:
            for ses in ses_folders:
                all_subjects.append(f"{sub}_{ses}")
        else:
            all_subjects.append(sub)

    all_subjects.sort()
    random.shuffle(all_subjects)

    # -----------------------------
    # Split train/val/test
    # -----------------------------
    train_ratio = 0.80
    val_ratio = 0.10
    n = len(all_subjects)
    n_train = round(train_ratio * n)
    n_val = round(val_ratio * n)
    train_subjects = all_subjects[:n_train]
    val_subjects = all_subjects[n_train:n_train + n_val]
    test_subjects = all_subjects[n_train + n_val:]

    # -----------------------------
    # Build dataset dict
    # -----------------------------
    dataset_dict = {
        "training": build_entries(train_subjects, "training", base_dir, target_dir, mode),
        "validation": build_entries(val_subjects, "validation", base_dir, target_dir, mode),
        "testing": build_entries(test_subjects, "testing", base_dir, target_dir, mode),
    }

    # -----------------------------
    # Save JSON
    # -----------------------------
    out_path = os.path.join(target_dir, f"{mode}_dataset", "dataset.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dataset_dict, f, indent=2)

    print(f"dataset.json created successfully for {mode} in prepared/")
    for split in ["training", "validation", "testing"]:
        print(f"Number of {split} pairs: {len(dataset_dict[split])}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python dataset_preparation.py <data_dir> <dwi|fmri>")
        sys.exit(1)

    base_dir = sys.argv[1]
    mode = sys.argv[2].lower()
    main(base_dir, mode)