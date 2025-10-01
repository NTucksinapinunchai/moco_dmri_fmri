"""
Augmentation script used to generate motion in x-y axis (translation,rotation) of dMRI and fMRI data in each volume
- the script will automatically find the specific data which is *dwi.nii.gz for dmri and *bold.nii.gz for fmri (due to BIDS format)
- the script is able to handle both dMRI and fMRI but you need to identify what data you need to perform augmentation

Usage: in terminal/command line
    python augmentation.py /path/to/data mode
    - /path/to/data --> directory of dataset
    - mode --> dmri or fmri :depending on dataset
"""

import os
import sys
import glob
import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

# ---------------------------
# Custom slice-wise transform
# ---------------------------
class RandSliceWiseAffine:
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
    slicewise_tf = RandSliceWiseAffine(max_rot=5, max_shift=2, prob=1.0, axis=2)
    subfolders = sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])

    for sub in subfolders:
        if mode == "dmri":
            sessions = sorted([f.path for f in os.scandir(sub) if f.is_dir() and f.name.startswith("ses-")])
            targets = sessions if sessions else [sub]

            for target in targets:
                print(f"\nProcessing {target} ...")
                nii_files = glob.glob(os.path.join(target, "dwi", "*dwi.nii.gz"))
                if not nii_files:
                    print(f"No dwi NIfTI found in {target}")
                    continue
                input_img = nii_files[0]

                root = os.path.basename(input_img)
                parts = root.split("_")
                if len(parts) > 1 and parts[1].startswith("ses-"):
                    prefix = "_".join(parts[:2])
                else:
                    prefix = parts[0]
                output_img = os.path.join(target, "dwi", f"aug_{prefix}_dwi.nii.gz")

                augment_data(input_img, output_img, slicewise_tf)

        elif mode == "fmri":
            sessions = sorted([f.path for f in os.scandir(sub) if f.is_dir() and f.name.startswith("ses-")])
            targets = sessions if sessions else [sub]

            for target in targets:
                print(f"\nProcessing {target} ...")
                nii_files = glob.glob(os.path.join(target, "func", "*bold.nii.gz"))
                if not nii_files:
                    print(f"No bold NIfTI found in {target}")
                    continue
                input_img = nii_files[0]

                root = os.path.basename(input_img)
                parts = root.split("_")
                if len(parts) > 1 and parts[1].startswith("ses-"):
                    prefix = "_".join(parts[:2])
                else:
                    prefix = parts[0]
                output_img = os.path.join(target, "func", f"aug_{prefix}_bold.nii.gz")

                augment_data(input_img, output_img, slicewise_tf)

        else:
            raise ValueError("Mode must be 'dmri' or 'fmri'")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python augmentation.py <data_dir> <dwi|fmri>")
        sys.exit(1)

    data_dir = sys.argv[1]
    mode = sys.argv[2].lower()
    main(data_dir, mode)
