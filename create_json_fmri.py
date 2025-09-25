import os
import json
import random
import torch
import shutil
import numpy as np
import nibabel as nib

from glob import glob

base_dir = "/Users/kenggkkeng/Desktop/fmri_dataset/"
target_dir = os.path.join(base_dir, "prepared")
os.makedirs(target_dir, exist_ok=True)

# -----------------------------
# Parameters for splitting
# -----------------------------
train_ratio = 0.80
val_ratio = 0.10
test_ratio = 0.10

# -----------------------------
# List all subject/session folders
# -----------------------------
all_subjects = []
for sub in os.listdir(base_dir):
    sub_path = os.path.join(base_dir, sub)
    if not os.path.isdir(sub_path):
        continue

    # Check if subject has session folders
    ses_folders = [d for d in os.listdir(sub_path) if d.startswith("ses-")]
    if ses_folders:
        # Expand sessions, e.g. sub-01_ses-01
        for ses in ses_folders:
            all_subjects.append(f"{sub}_{ses}")
    else:
        # No session, just sub-XX
        all_subjects.append(sub)

all_subjects.sort()
random.shuffle(all_subjects)

# -----------------------------
# Split subjects (now at sub_ses level)
# -----------------------------
n = len(all_subjects)
n_train = round(train_ratio * n)
n_val   = round(val_ratio * n)
n_test  = n - n_train - n_val
train_subjects = all_subjects[:n_train]
val_subjects   = all_subjects[n_train:n_train + n_val]
test_subjects  = all_subjects[n_train + n_val:]

# -----------------------------
# Helper: load NIfTI → torch
# -----------------------------
def load_as_tensor(nii_path, add_channel=True):
    img = nib.load(nii_path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata().astype(np.float32)
    tensor = torch.from_numpy(data)
    if add_channel:  # add leading channel dimension
        tensor = tensor.unsqueeze(0)
    return tensor, img.affine.astype(np.float32)

# -----------------------------
# Build dataset entries
# -----------------------------
def build_entries(subject_list, split_name):
    entries = []
    for sub_ses in subject_list:
        # Parse subject + optional session
        if "_ses-" in sub_ses:
            sub, ses = sub_ses.split("_", 1)
            fmri_folder = os.path.join(base_dir, sub, ses, "func")
        else:
            sub = sub_ses
            fmri_folder = os.path.join(base_dir, sub, "func")

        moving_files = glob(os.path.join(fmri_folder, "aug_*.nii.gz"))
        for moving_path in moving_files:
            fixed_path = moving_path.replace("aug_", "dup_").replace("_bold.nii.gz", "_mean_fixed.nii.gz")
            mask_files = glob(os.path.join(fmri_folder, "mask_*.nii.gz"))
            mask_path = mask_files[0]

            # Load & convert → save as .pt
            moving, affine = load_as_tensor(moving_path, add_channel=True)  # (1,H,W,D,T)
            fixed, _ = load_as_tensor(fixed_path, add_channel=True)    # (1,H,W,D,T)
            mask, _ = load_as_tensor(mask_path, add_channel=True)      # (1,H,W,D)

            # Save into prepared directory
            out_sub = sub_ses  # e.g. "sub-01_ses-01"
            out_folder = os.path.join(target_dir, "fmri_dataset", split_name, out_sub, "func")
            os.makedirs(out_folder, exist_ok=True)
            # Copy the original NIfTI files
            for src in [moving_path, fixed_path, mask_path]:
                if os.path.exists(src):
                    shutil.copy2(src, out_folder)
                else:
                    print(f"Warning: missing {src}")

            # ----------------------------------------------------
            # Copy extra fMRI
            # ----------------------------------------------------
            patterns = [
                os.path.join(fmri_folder, f"{sub}_*_bold.nii.gz"),  # fallback: subject-only
            ]
            for pattern in patterns:
                for extra_file in glob(pattern):
                    try:
                        shutil.copy2(extra_file, out_folder)
                        print(f"Copied extra file (not in JSON): {extra_file}")
                    except Exception as e:
                        print(f"Could not copy {extra_file}: {e}")

            pt_filename = os.path.basename(moving_path).replace("aug_", "paired_").replace(".nii.gz", ".pt")
            pt_path = os.path.join(out_folder, pt_filename)

            torch.save({"moving": moving, "fixed": fixed, "mask": mask, "affine": affine}, pt_path)

            # JSON entry points to .pt file
            entry = {"data": os.path.relpath(pt_path, target_dir)}
            entries.append(entry)
    return entries


dataset_dict = {
    "training": build_entries(train_subjects, "training"),
    "validation": build_entries(val_subjects, "validation"),
    "testing": build_entries(test_subjects, "testing"),
}

# -----------------------------
# Save JSON inside prepared/
# -----------------------------
out_path = os.path.join(target_dir, "fmri_dataset", "dataset.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(dataset_dict, f, indent=2)

print("dataset.json created successfully in prepared/")
print(f"Number of training pairs: {len(dataset_dict['training'])}")
print(f"Number of validation pairs: {len(dataset_dict['validation'])}")
print(f"Number of testing pairs: {len(dataset_dict['testing'])}")