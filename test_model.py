import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import glob
import time
import torch
torch.set_float32_matmul_precision("medium")

import numpy as np
import nibabel as nib
from main import DataModule, VoxelMorphReg
from skimage.exposure import match_histograms
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform, apply_orientation

# -----------------------------
# Setup paths
# -----------------------------
base_path = "/home/ge.polymtl.ca/p122983/nontharat/moco_dmri/"
sys.path.insert(0, base_path)
json_path = os.path.join(base_path, "dmri_dataset", "dataset.json")
# json_path = os.path.join(base_path, "fmri_dataset", "dataset.json")

order_execution = sys.argv[1]
ckpt_path = os.path.join(base_path, "trained_weights", f"{order_execution}_voxelmorph_best-weighted.ckpt")

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda")
# device = torch.device("cpu")

# -----------------------------
# Prepare datamodule and model
# -----------------------------
dm = DataModule(json_path=json_path, base_dir=base_path, batch_size=1, num_workers=8)
dm.setup("test")

model = VoxelMorphReg.load_from_checkpoint(ckpt_path, map_location=device)
model = model.to(device)
model.eval()

test_loader = dm.test_dataloader()

# -----------------------------
# Run inference & save outputs
# -----------------------------
timings = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        start_time = time.time()

        moving = batch["moving"].to(device)  # (B,1,H,W,D,T)
        fixed = batch["fixed"].to(device)    # (B,1,H,W,D,T)
        mask = batch["mask"].to(device)      # (B,1,H,W,D)
        affine = batch["affine"]

        if torch.is_tensor(affine):
            affine = affine.cpu().numpy()

        warped_all, flows_4d = model(moving, fixed, mask)

        elapsed_time = time.time() - start_time
        timings.append(elapsed_time)
        print(f"[{i}] Inference time: {elapsed_time:.2f} sec")

        # -----------------------------
        # Save warped images & flows (RAS -> native)
        # -----------------------------
        pt_path = batch["data_path"]
        if isinstance(pt_path, list):
            pt_path = pt_path[0]
        save_dir = os.path.dirname(pt_path)
        prefix = os.path.basename(os.path.dirname(save_dir))

        # ---- Load raw reference in native orientation ----
        raws = glob.glob(os.path.join(save_dir, f"{prefix}_dwi.nii.gz"))
        # raws = glob.glob(os.path.join(save_dir, f"{prefix}_*_bold.nii.gz"))
        if len(raws) == 0:
            raise FileNotFoundError(f"No fixed file found in {save_dir}")
        ref_path = raws[0]

        ref_img = nib.load(ref_path)  # scanner-native
        ref_affine = ref_img.affine
        ref_header = ref_img.header
        ref_data = ref_img.get_fdata()

        # --- orientation transform RAS -> native ---
        ras_ornt = axcodes2ornt(("R", "A", "S"))  # model always RAS
        native_ornt = io_orientation(ref_affine)  # scanner orientation
        ras2nat = ornt_transform(ras_ornt, native_ornt)

        # -----------------------------
        # Re-orientation helpers
        # -----------------------------
        def reorient_ras_to_native(arr_ras, ras2nat):
            if arr_ras.ndim == 4:  # (H,W,D,T)
                arr_nat = np.stack(
                    [apply_orientation(arr_ras[..., t], ras2nat) for t in range(arr_ras.shape[-1])],
                    axis=-1
                )
            else:  # 3D
                arr_nat = apply_orientation(arr_ras, ras2nat)
            return arr_nat

        def reorient_flow_ras_to_native(flow_ras_t, ras2nat):
            comps = [apply_orientation(flow_ras_t[c], ras2nat) for c in range(3)]
            comps = np.stack(comps, axis=0)

            idx = ras2nat[:, 0].astype(int)
            sgn = ras2nat[:, 1].astype(int)
            comps = comps[idx, ...]
            for k in range(3):
                if sgn[k] == -1:
                    comps[k] *= -1
            return comps  # (3,Hn,Wn,Dn)

        warped_ras = warped_all.squeeze(0).squeeze(0).cpu().numpy()  # (H,W,D,T)
        flow_ras = flows_4d.squeeze(0).cpu().numpy()  # (3,H,W,D,T)

        warped_nat = reorient_ras_to_native(warped_ras, ras2nat)

        # -----------------------------
        # Histogram Matching
        # -----------------------------
        matched = np.zeros_like(warped_nat)
        for t in range(warped_nat.shape[-1]):
            matched[..., t] = match_histograms(warped_nat[..., t], ref_data[..., t])

        flow_nat = np.stack(
            [reorient_flow_ras_to_native(flow_ras[..., t], ras2nat) for t in range(flow_ras.shape[-1])],
            axis=-1
        )  # (3,Hn,Wn,Dn,T)

        dz_nat, dx_nat, dy_nat = flow_nat[0], flow_nat[1], flow_nat[2]      # coordinate convention

        # --- save outputs with native affine/header ---
        nib.save(nib.Nifti1Image(matched, ref_affine, header=ref_header), os.path.join(save_dir, f"moco_{prefix}_dmri.nii.gz"))
        # nib.save(nib.Nifti1Image(matched, ref_affine, header=ref_header), os.path.join(save_dir, f"moco_{prefix}_fmri.nii.gz"))
        nib.save(nib.Nifti1Image(dx_nat, ref_affine, header=ref_header), os.path.join(save_dir, f"{prefix}_dx.nii.gz"))
        nib.save(nib.Nifti1Image(dy_nat, ref_affine, header=ref_header), os.path.join(save_dir, f"{prefix}_dy.nii.gz"))
        nib.save(nib.Nifti1Image(dz_nat, ref_affine, header=ref_header), os.path.join(save_dir, f"{prefix}_dz.nii.gz"))

        print(f"[{i}] Saved outputs to: {save_dir}")

# -----------------------------
# Final Timing Summary
# -----------------------------
if timings:
    timings = np.array(timings)
    print("\n=== Inference Timing Summary ===")
    print(f"Samples: {len(timings)}")
    print(f"Mean:    {timings.mean():.2f} sec")
    print(f"Std:     {timings.std():.2f} sec")
    print(f"Min:     {timings.min():.2f} sec")
    print(f"Max:     {timings.max():.2f} sec")
