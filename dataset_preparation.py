# -*- coding: utf-8 -*-
"""
Preparation script used to build dataset.json for dMRI and fMRI datasets to be used in
training/validation of the DL motion correction model.

This script converts NIfTI files into PyTorch tensors (.pt), organizes them into
train/val/test splits, and generates a JSON index compatible with the training pipeline.

Overview:
---------
1. For each subject/session:
    - Identify moving, fixed, and mask NIfTI files (according to mode: dmri or fmri).
    - Convert them into PyTorch tensors with an added channel dimension.
    - Save as paired .pt files including affine matrix.
    - Copy original NIfTI files into the prepared structure.
2. Split subjects into:
    - Training (80%)
    - Validation (10%)
    - Testing (10%)
3. Save `dataset.json` with relative paths to .pt files.

Usage:
------
In terminal/command line:

    python dataset_preparation.py /path/to/data mode

Arguments:
    /path/to/data    : data directory for training dataset preparation
    mode             : 'dmri' or 'fmri' depending on dataset type
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
from config_loader import config

# -----------------------------
# Helper: load NIfTI → torch
# -----------------------------
def load_as_tensor(nii_path, add_channel=True):
    """
    Load NIfTI file and return as PyTorch tensor.
    """
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
    """
    Build dataset entries for given subjects and save paired .pt files.
    """
    entries = []
    patterns = config[mode]
    mode_dir = os.path.join(target_dir, f"{mode}_dataset")  # added for relative path calculation

    for sub_ses in subject_list:
        if "_ses-" in sub_ses:
            sub, ses = sub_ses.split("_", 1)
            data_folder = os.path.join(base_dir, sub, ses, patterns["subdir"])
            out_sub = sub_ses
        else:
            sub = sub_ses
            data_folder = os.path.join(base_dir, sub, patterns["subdir"])
            out_sub = sub

        moving_files = glob(os.path.join(data_folder, patterns["moving"]))
        if not moving_files:
            continue

        for moving_path in moving_files:
            fixed_path = glob(os.path.join(data_folder, patterns["fixed"]))[0]
            mask_path = glob(os.path.join(data_folder, patterns["mask"]))[0]
            out_folder = os.path.join(mode_dir, split_name, out_sub, patterns["subdir"])
            os.makedirs(out_folder, exist_ok=True)

            if split_name in ["training", "validation"]:
                # --- Convert and save .pt ---
                moving, affine = load_as_tensor(moving_path, add_channel=True)
                fixed, _ = load_as_tensor(fixed_path, add_channel=True)
                mask, _ = load_as_tensor(mask_path, add_channel=True)

                pt_filename = os.path.basename(moving_path).replace("aug_", "paired_").replace(".nii.gz", ".pt")
                pt_path = os.path.join(out_folder, pt_filename)
                torch.save({"moving": moving, "fixed": fixed, "mask": mask, "affine": affine}, pt_path)

                rel_path = os.path.relpath(pt_path, mode_dir)
                entries.append({"data": rel_path})

            elif split_name == "testing":
                # --- Only copy NIfTIs, no .pt ---
                for src in [moving_path, fixed_path, mask_path]:
                    if src and os.path.exists(src):
                        shutil.copy2(src, out_folder)

                extra_files = glob(os.path.join(data_folder, patterns["raw"]))
                if extra_files:
                    shutil.copy2(extra_files[0], out_folder)
                    print(f"Copied extra test file: {extra_files[0]}")

                # save relative path of NIfTIs into JSON (instead of .pt)
                entries.append({
                    "moving": os.path.join(split_name, out_sub, patterns["subdir"], os.path.basename(moving_path)),
                    "fixed": os.path.join(split_name, out_sub, patterns["subdir"], os.path.basename(fixed_path)),
                    "mask": os.path.join(split_name, out_sub, patterns["subdir"], os.path.basename(mask_path)),
                })

    entries = sorted(entries, key=lambda e: list(e.values())[0])
    return entries

# -----------------------------
# Collect subjects & save JSON file
# -----------------------------
def main(base_dir, mode):
    target_dir = os.path.join(base_dir, "prepared")
    os.makedirs(target_dir, exist_ok=True)

    # Collect subjects
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

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python dataset_preparation.py <data_dir> <dmri|fmri>")
        sys.exit(1)

    base_dir = sys.argv[1]
    mode = sys.argv[2].lower()
    main(base_dir, mode)