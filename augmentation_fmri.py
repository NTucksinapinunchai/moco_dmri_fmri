import os
import glob
import nibabel as nib
import numpy as np
import torchio as tio
from flask import session

from scipy.ndimage import affine_transform

# ---------------------------
# Custom slice-wise transform
# ---------------------------
class RandSliceWiseAffine:
    def __init__(self, max_rot=5, max_shift=2, prob=0.9, axis=2):
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

# -----------------------
# Define transforms
# -----------------------
spine_axis = 2  # slice axis = axial slices through C-spine
slicewise_tf = RandSliceWiseAffine(max_rot=5, max_shift=2, prob=0.9, axis=spine_axis)

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

        nii_files = glob.glob(os.path.join(target, "func", "sub-*bold.nii.gz"))
        if len(nii_files) == 0:
            print(f"No NIfTI file found in {target}")
        input_img = nii_files[0]
        print("Found NIfTI:", input_img)

        basename = os.path.basename(input_img)
        output_img = os.path.join(target, "func", "aug_" + basename)

        img = nib.load(input_img)
        data = img.get_fdata().astype(np.float32)
        H, W, D, T = data.shape
        print("Input shape:", data.shape)

        aug_data = np.zeros_like(data)

        # Apply slice-wise affine augmentation
        for t in range(T):
            vol = data[:, :, :, t]
            aug_vol = slicewise_tf(vol)
            aug_data[:, :, :, t] = aug_vol
            print(f"Augmented volume {t + 1}/{T}")

        # Save augmented 4D data
        aug_img = nib.Nifti1Image(aug_data, img.affine, img.header)
        nib.save(aug_img, output_img)
        print(f"Saved augmented dataset: {output_img}")
