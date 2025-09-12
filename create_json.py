import os
import json
import random
import shutil
from glob import glob

base_dir = "/Users/kenggkkeng/Desktop/moco_dmri/sourcedata/"

# Parameters for splitting
train_ratio = 0.8
val_ratio = 0.11
test_ratio = 0.09

# List all subject folders automatically
all_subjects = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
all_subjects.sort()
random.shuffle(all_subjects)

# Split subjects
n = len(all_subjects)
n_train = int(train_ratio * n)
n_val = int(val_ratio * n)
train_subjects = all_subjects[:n_train]
val_subjects = all_subjects[n_train:n_train + n_val]
test_subjects = all_subjects[n_train + n_val:]

# Create target folders if not exist
for split in ["training", "validation", "testing"]:
    split_dir = os.path.join(base_dir, split)
    os.makedirs(split_dir, exist_ok=True)

# Move subject folders to split directories
def move_subjects(subject_list, split_name):
    for sub in subject_list:
        src = os.path.join(base_dir, sub)
        dst = os.path.join(base_dir, split_name, sub)
        if os.path.exists(src):
            shutil.move(src, dst)

move_subjects(train_subjects, "training")
move_subjects(val_subjects, "validation")
move_subjects(test_subjects, "testing")

# Build dataset entries
def build_entries(subject_list, split_name):
    entries = []
    for sub in subject_list:
        dwi_folder = os.path.join(base_dir, split_name, sub, "dwi")
        moving_files = glob(os.path.join(dwi_folder, "aug_*.nii.gz"))
        for moving_path in moving_files:
            fixed_path = moving_path.replace("aug_", "").replace(".nii.gz", "_dwi_mean.nii.gz")
            entries.append({
                "moving": os.path.join("sourcedata", os.path.relpath(moving_path, base_dir)),
                "fixed": os.path.join("sourcedata", os.path.relpath(fixed_path, base_dir))
            })
    return entries

dataset_dict = {
    "training": build_entries(train_subjects, "training"),
    "validation": build_entries(val_subjects, "validation"),
    "testing": build_entries(test_subjects, "testing")
}

# Save JSON
out_path = os.path.join(base_dir, "dataset.json")
with open(out_path, "w") as f:
    json.dump(dataset_dict, f, indent=2)

print("dataset.json created successfully!")
print(f"Number of training pairs: {len(dataset_dict['training'])}")
print(f"Training subjects: {train_subjects}")
print(f"Number of validation pairs: {len(dataset_dict['validation'])}")
print(f"Validation subjects: {val_subjects}")
print(f"Number of testing pairs: {len(dataset_dict['testing'])}")
print(f"Testing subjects: {test_subjects}")
