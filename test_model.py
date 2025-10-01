"""
Inference script for motion correction model (VoxelMorphReg).
Loads raw NIfTI (moving, fixed, mask) directly instead of .pt/.json.
"""

import os
import sys
import glob
import time
import torch
torch.set_float32_matmul_precision("medium")

import numpy as np
import nibabel as nib

from moco_main import VoxelMorphReg
from monai.networks.blocks import Warp
from skimage.exposure import match_histograms

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Main inference
# -----------------------------
def main(data_dir, run_name):
    ckpt_path = os.path.join(os.path.dirname(data_dir), "trained_weights", run_name)

    # -----------------------------
    # Load model
    # -----------------------------
    model = VoxelMorphReg.load_from_checkpoint(ckpt_path, map_location=device)
    model = model.to(device)
    model.eval()

    # Nearest-neighbor warper for sharper inference
    warp = Warp(mode="nearest", padding_mode="border").to(device)

    # -----------------------------
    # Detect subjects and files automatically
    # -----------------------------
    subjects = sorted(glob.glob(os.path.join(data_dir, "sub-*")))
    timings = []

    for sub in subjects:
        sessions = sorted(glob.glob(os.path.join(sub, "ses-*")))
        targets = sessions if sessions else [sub]

        for target in targets:
            print(f"\nProcessing {target} ...")

            if "dmri_dataset" in data_dir:
                moving_files = glob.glob(os.path.join(target, "dwi", "aug_*dwi.nii.gz"))
                fixed_files  = glob.glob(os.path.join(target, "dwi", "dup_*fixed.nii.gz"))
                mask_files   = glob.glob(os.path.join(target, "dwi", "mask_*.nii.gz"))
                suffix = "dmri"
                subdir = "dwi"
            else:
                moving_files = glob.glob(os.path.join(target, "func", "aug_*bold.nii.gz"))
                fixed_files  = glob.glob(os.path.join(target, "func", "*_fixed.nii.gz"))
                mask_files   = glob.glob(os.path.join(target, "func", "mask_*.nii.gz"))
                suffix = "fmri"
                subdir = "func"

            if not (moving_files and fixed_files and mask_files):
                print(f"Missing files in {target}, skipping.")
                continue

            moving_img = nib.load(moving_files[0])
            fixed_img  = nib.load(fixed_files[0])
            mask_img   = nib.load(mask_files[0])

            moving = moving_img.get_fdata().astype(np.float32)
            fixed  = fixed_img.get_fdata().astype(np.float32)
            mask   = mask_img.get_fdata().astype(np.float32)

            affine = fixed_img.affine
            header = fixed_img.header
            ref_data = fixed  # for histogram matching

            # add batch/channel dimensions
            moving = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W,D,T)
            if fixed.ndim == 4:
                fixed = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0).to(device) # (1,1,H,W,D,T)
            else:
                fixed = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0).to(device) # (1,1,H,W,D)
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)

            # -----------------------------
            # Run inference
            # -----------------------------
            start_time = time.time()
            with torch.no_grad():
                warped_all, flows_4d = model(moving, fixed, mask)

                # re-warp with nearest interpolation for sharper results
                warped_list = []
                for t in range(moving.shape[-1]):
                    mov_t = moving[..., t]  # (1,1,H,W,D)
                    flow_t = flows_4d[..., t]  # (1,3,H,W,D)
                    warped_t = warp(mov_t, flow_t)
                    warped_list.append(warped_t)

                warped_all = torch.stack(warped_list, dim=-1)  # (1,1,H,W,D,T)

            elapsed_time = time.time() - start_time
            timings.append(elapsed_time)
            print(f"Inference time: {elapsed_time:.2f} sec")

            # convert to numpy
            warped = warped_all.squeeze(0).squeeze(0).cpu().numpy()  # (H,W,D,T)
            flow = flows_4d.squeeze(0).cpu().numpy()  # (3,H,W,D,T)
            dz, dx, dy = flow[0], flow[1], flow[2]

            # histogram matching
            matched = np.zeros_like(warped)
            for t in range(warped.shape[-1]):
                matched[..., t] = match_histograms(warped[..., t], ref_data[..., t])

            # -----------------------------
            # Save the output
            # -----------------------------
            out_dir = os.path.join(target, subdir)
            prefix = os.path.basename(target)
            nib.save(nib.Nifti1Image(matched, affine, header=header), os.path.join(out_dir, f"moco_{prefix}_{suffix}.nii.gz"))
            nib.save(nib.Nifti1Image(dx, affine, header=header), os.path.join(out_dir, f"{prefix}_dx.nii.gz"))
            nib.save(nib.Nifti1Image(dy, affine, header=header), os.path.join(out_dir, f"{prefix}_dy.nii.gz"))
            nib.save(nib.Nifti1Image(dz, affine, header=header), os.path.join(out_dir, f"{prefix}_dz.nii.gz"))

            print(f"Saved outputs to: {out_dir}")

    # -----------------------------
    # Summary time
    # -----------------------------
    if timings:
        print("\n=== Inference Timing Summary ===")
        print(f"Samples: {len(timings)}")
        print(f"Mean:    {np.mean(timings):.2f} sec")
        print(f"Std:     {np.std(timings):.2f} sec")
        print(f"Min:     {np.min(timings):.2f} sec")
        print(f"Max:     {np.max(timings):.2f} sec")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_model.py <data_dir> <run_name>")
        sys.exit(1)

    data_dir = sys.argv[1]
    run_name = sys.argv[2]
    main(data_dir, run_name)