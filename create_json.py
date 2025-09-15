import os
import json
import random
import shutil
from glob import glob

base_dir = "/Users/kenggkkeng/Desktop/sourcedata/"      # if you change directory, please make sure the rest is the same.
target_dir = os.path.join(base_dir, "prepared")
os.makedirs(target_dir, exist_ok=True)

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

# Build dataset entries
def build_entries(subject_list, split_name):
    entries = []
    for sub in subject_list:
        dwi_folder = os.path.join(base_dir, sub, "dwi")
        moving_files = glob(os.path.join(dwi_folder, "aug_*.nii.gz"))
        for moving_path in moving_files:
            fixed_path = moving_path.replace("aug_", "dup_").replace(".nii.gz", "_mean.nii.gz")
            mask_path = moving_path.replace("aug_", "mask_").replace(".nii.gz", "_dwi_mean.nii.gz")

            entry = {
                "moving": os.path.join("sourcedata", split_name, sub, "dwi", os.path.basename(moving_path)),
                "fixed": os.path.join("sourcedata", split_name, sub, "dwi", os.path.basename(fixed_path)),
                "mask": os.path.join("sourcedata", split_name, sub, "dwi", os.path.basename(mask_path)),
            }
            entries.append(entry)
    return entries

dataset_dict = {
    "training": build_entries(train_subjects, "training"),
    "validation": build_entries(val_subjects, "validation"),
    "testing": build_entries(test_subjects, "testing"),
}

# Copy only files referenced in JSON into prepared/
for split, entries in dataset_dict.items():
    for entry in entries:
        for key in ["moving", "fixed", "mask"]:
            src = os.path.join(base_dir, entry[key].replace(f"sourcedata/{split}/", ""))
            dst = os.path.join(target_dir, entry[key])
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"Missing file: {src}")

# Save JSON inside prepared/
out_path = os.path.join(target_dir, "sourcedata", "dataset.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(dataset_dict, f, indent=2)

print("dataset.json created successfully in prepared/")
print(f"Number of training pairs: {len(dataset_dict['training'])}")
print(f"Number of validation pairs: {len(dataset_dict['validation'])}")
print(f"Number of testing pairs: {len(dataset_dict['testing'])}")
