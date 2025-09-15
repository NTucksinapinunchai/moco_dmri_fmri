import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import torch
torch.set_float32_matmul_precision("medium")

import torch.nn.functional as F
import nibabel as nib
import numpy as np
import pandas as pd

from monai.metrics import DiceMetric
from monai.networks.blocks import Warp
from main import DataModule, VoxelMorphReg

# -----------------------------
# Setup paths
# -----------------------------
base_path = "/home/ge.polymtl.ca/p122983/nontharat/moco_dmri/"
sys.path.insert(0, base_path)
json_path = os.path.join(base_path, "sourcedata", "dataset.json")

order_execution = sys.argv[1]
ckpt_path = os.path.join(base_path, "trained_weights", f"{order_execution}_voxelmorph_best-weighted.ckpt")

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Prepare datamodule and model
# -----------------------------
dm = DataModule(json_path=json_path, batch_size=1, num_workers=15)
dm.setup("test")

model = VoxelMorphReg.load_from_checkpoint(ckpt_path, map_location=device)
model = model.to(device)
model.eval()

test_loader = dm.test_dataloader()
dice_metric = DiceMetric(include_background=True, reduction="mean")

# -----------------------------
# Run inference & compute/save
# -----------------------------
all_sample_means = []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        moving = batch["moving"].to(device)  # (B,1,H,W,D,T)
        fixed = batch["fixed"].to(device)    # (B,1,H,W,D)

        warped_all, flows_4d = model(moving, fixed)
        B, C, H, W, D, T = warped_all.shape

        sample_dice_before, sample_dice_after = [], []

        for t in range(T):
            warped_t = warped_all[..., t]  # (B,1,H,W,D)

            # --- BEFORE warp ---
            moving_bin = (moving[..., t] > 0.5).float()  # (B,1,H,W,D)
            fixed_bin = (fixed > 0.5).float()
            dice_before = dice_metric(moving_bin, fixed_bin).item()

            # --- AFTER warp ---
            warped_bin = (warped_t > 0.5).float()
            dice_after = dice_metric(warped_bin, fixed_bin).item()

            sample_dice_before.append(dice_before)
            sample_dice_after.append(dice_after)

            print(f"[{i}] Volume {t}: DICE BEFORE = {dice_before:.4f}, AFTER = {dice_after:.4f}")

        mean_dice_before = np.mean(sample_dice_before)
        mean_dice_after = np.mean(sample_dice_after)
        all_sample_means.append(mean_dice_after)

        print(f"[{i}] Mean DICE (sample): BEFORE = {mean_dice_before:.4f}, AFTER = {mean_dice_after:.4f}")

        # -----------------------------
        # Save warped images & flows
        # -----------------------------
        moving_path = batch["moving_path"][0]
        save_dir = os.path.dirname(moving_path)
        prefix = os.path.basename(os.path.dirname(save_dir))

        # Ensure affine is RAS (consistent with DataGenerator)
        src_img = nib.load(moving_path)
        src_img = nib.as_closest_canonical(src_img)
        affine = src_img.affine.astype(np.float64)

        warped_np = warped_all.squeeze(0).squeeze(0).cpu().numpy()  # (H,W,D,T)
        dx_np = flows_4d[:, 0].squeeze(0).cpu().numpy()
        dy_np = flows_4d[:, 1].squeeze(0).cpu().numpy()
        dz_np = flows_4d[:, 2].squeeze(0).cpu().numpy()

        nib.save(nib.Nifti1Image(warped_np, affine), os.path.join(save_dir, f"{prefix}_moco.nii.gz"))
        nib.save(nib.Nifti1Image(dx_np, affine), os.path.join(save_dir, f"{prefix}_dx.nii.gz"))
        nib.save(nib.Nifti1Image(dy_np, affine), os.path.join(save_dir, f"{prefix}_dy.nii.gz"))
        nib.save(nib.Nifti1Image(dz_np, affine), os.path.join(save_dir, f"{prefix}_dz.nii.gz"))

        print(f"[{i}] Saved outputs to: {save_dir}")

        # -----------------------------
        # Save per-case DICE scores
        # -----------------------------
        dice_csv_path = os.path.join(save_dir, f"{prefix}_dice_scores.csv")
        with open(dice_csv_path, "w") as f:
            f.write("volume,dice_before,dice_after\n")
            for t, (db, da) in enumerate(zip(sample_dice_before, sample_dice_after)):
                f.write(f"{t},{db:.6f},{da:.6f}\n")
            f.write(f"mean,{mean_dice_before:.6f},{mean_dice_after:.6f}\n")

        print(f"[{i}] Saved DICE to: {dice_csv_path}")

# -----------------------------
# Final Excel Summary (3 sheets)
# -----------------------------
dice_data_before, dice_data_after = {}, {}
max_vols = 0

for i in range(len(test_loader.dataset)):
    moving_path = test_loader.dataset.data[i]["moving"]
    save_dir = os.path.dirname(moving_path)
    prefix = os.path.basename(os.path.dirname(save_dir))

    dice_csv_path = os.path.join(save_dir, f"{prefix}_dice_scores.csv")
    df = pd.read_csv(dice_csv_path)

    volume_dice_before = df[df["volume"] != "mean"]["dice_before"].astype(float).tolist()
    volume_dice_before.append(df[df["volume"] == "mean"]["dice_before"].values[0])

    volume_dice_after = df[df["volume"] != "mean"]["dice_after"].astype(float).tolist()
    volume_dice_after.append(df[df["volume"] == "mean"]["dice_after"].values[0])

    dice_data_before[prefix] = volume_dice_before
    dice_data_after[prefix] = volume_dice_after
    max_vols = max(max_vols, len(volume_dice_before))

row_labels = [str(v) for v in range(max_vols - 1)] + ["mean"]
df_before, df_after, df_improve = pd.DataFrame(index=row_labels), pd.DataFrame(index=row_labels), pd.DataFrame(index=row_labels)

eps = 1e-8
for case in dice_data_before:
    padded_before = dice_data_before[case] + [np.nan] * (max_vols - len(dice_data_before[case]))
    padded_after = dice_data_after[case] + [np.nan] * (max_vols - len(dice_data_after[case]))

    df_before[case] = padded_before
    df_after[case] = padded_after

    arr_before = np.array(padded_before, dtype=float)
    arr_after = np.array(padded_after, dtype=float)
    improvement = (arr_after - arr_before) / (arr_before + eps) * 100
    df_improve[case] = improvement

testing_dir = os.path.join(base_path, "sourcedata")
excel_out = os.path.join(testing_dir, "testing", f"{order_execution}_dice_summary.xlsx")

with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
    df_before.index.name = "volume"
    df_before.to_excel(writer, sheet_name="before")

    df_after.index.name = "volume"
    df_after.to_excel(writer, sheet_name="after")

    df_improve.index.name = "volume"
    df_improve.to_excel(writer, sheet_name="%improvement")

print(f"Saved final summary Excel to: {excel_out} (sheets: before, after, improvement_%)")
