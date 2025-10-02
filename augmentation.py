# -*- coding: utf-8 -*-
"""
Augmentation script to simulate motion artifacts in dMRI and fMRI data via slice-wise affine transforms (rotation + translation).

- Automatically detects dMRI (*dwi.nii.gz) and fMRI (*bold.nii.gz) data following BIDS format.
- Performs slice-wise transformations (x–y translation, in-plane rotation).
- Can handle both dMRI and fMRI; user specifies which mode.

Usage:
------
In terminal/command line:

    python augmentation.py /path/to/data mode

Arguments:
    /path/to/data : data directory for training dataset preparation
    mode          : 'dmri' or 'fmri' depending on dataset type
"""

import os
import sys
import glob
import nibabel as nib
import numpy as np

from config_loader import config
from scipy.ndimage import affine_transform

# ---------------------------
# Custom slice-wise transform
# ---------------------------
class RandSliceWiseAffine:
    """
    Apply random slice-wise affine transform (rotation + shift) to a 3D volume.
    """
    def __init__(self, max_rot=5, max_shift=2, prob=1.0, axis=2):
        self.max_rot = max_rot
        self.max_shift = max_shift
        self.prob = prob
        self.axis = axis

    def __call__(self, vol: np.ndarray) -> np.ndarray:
        out = vol.copy()
        if np.random.rand() > self.prob:
            return out

        H, W, D = out.shape
        if self.axis != 2:
            out = np.moveaxis(out, self.axis, -1)
            H, W, D = out.shape

        for idx in range(D):
            angle = np.random.uniform(-self.max_rot, self.max_rot)
            tx = np.random.uniform(-self.max_shift, self.max_shift)
            ty = np.random.uniform(-self.max_shift, self.max_shift)

            theta = np.deg2rad(angle)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            center = np.array([H / 2, W / 2])
            inv_rot = np.linalg.inv(rot)
            offset = center - inv_rot @ center + np.array([tx, ty])

            out[:, :, idx] = affine_transform(
                out[:, :, idx], inv_rot, offset=offset,
                order=0, mode="nearest"
            )

        if self.axis != 2:
            out = np.moveaxis(out, -1, self.axis)

        return out

# ---------------------------
# Augmentation functions
# ---------------------------
def augment_data(input_img, output_img, slicewise_tf):
    """
    Load a 4D dataset and apply slice-wise augmentation volume-by-volume.
    """
    print(f"Augmenting: {input_img}")
    img = nib.load(input_img)
    data = img.get_fdata().astype(np.float32)
    H, W, D, T = data.shape
    print("Input shape:", data.shape)

    aug_data = np.zeros_like(data)
    for t in range(T):
        vol = data[:, :, :, t]
        aug_vol = slicewise_tf(vol)
        aug_data[:, :, :, t] = aug_vol
        print(f"Augmented volume {t + 1}/{T}")

    aug_img = nib.Nifti1Image(aug_data, img.affine, img.header)
    nib.save(aug_img, output_img)
    print(f"Saved augmented dataset: {output_img}")

# ---------------------------
# Main
# ---------------------------
def main(data_dir, mode):
    """
    Perform augmentation for all subjects in dataset directory (dmri or fmri).
    """
    slicewise_tf = RandSliceWiseAffine(max_rot=5, max_shift=2, prob=1.0, axis=2)
    subfolders = sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])

    patterns = config[mode]

    for sub in subfolders:
        # handle sessions if they exist
        sessions = sorted([f.path for f in os.scandir(sub) if f.is_dir() and f.name.startswith("ses-")])
        targets = sessions if sessions else [sub]

        for target in targets:
            print(f"\nProcessing {target} ...")

            # use the config-defined subdir + raw pattern
            nii_files = glob.glob(os.path.join(target, patterns["subdir"], patterns["raw"]))
            if not nii_files:
                print(f"No {mode} NIfTI found in {target}")
                continue
            input_img = nii_files[0]

            # construct prefix
            root = os.path.basename(input_img)
            parts = root.split("_")
            if len(parts) > 1 and parts[1].startswith("ses-"):
                prefix = "_".join(parts[:2])
            else:
                prefix = parts[0]

            # build output path
            output_img = os.path.join(
                target, patterns["subdir"], f"aug_{prefix}_{patterns['suffix']}.nii.gz"
            )

            # run augmentation
            augment_data(input_img, output_img, slicewise_tf)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python augmentation.py <data_dir> <dmri|fmri>")
        sys.exit(1)

    data_dir = sys.argv[1]
    mode = sys.argv[2].lower()
    main(data_dir, mode)
