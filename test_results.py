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

label_warper = Warp(mode="nearest", padding_mode="border").to(device)

# -----------------------------
# Run inference & compute/save
# -----------------------------
all_sample_means = []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        moving = batch["moving"].to(device)
        fixed = batch["fixed"].to(device)

        warped_all, flows_4d, (dx, dy, dz) = model(moving, fixed)

        B, C, D, H, W, T = warped_all.shape
        sample_dice_scores = []

        for t in range(T):
            warped_t = warped_all[..., t]  # (B,1,D,H,W)

            if "moving_label" in batch and "fixed_label" in batch:
                moving_lbl = batch["moving_label"].to(device).float()  # (B,1,D,H,W)
                fixed_lbl = batch["fixed_label"].to(device).float()

                flow_t = flows_4d[..., t]  # (B,3,D,H,W)
                warped_lbl_t = label_warper(moving_lbl, flow_t)

                # ensure exact match before Dice
                if warped_lbl_t.shape != fixed_lbl.shape:
                    warped_lbl_t = F.interpolate(
                        warped_lbl_t, size=fixed_lbl.shape[2:], mode="nearest"
                    )
                dice = dice_metric(warped_lbl_t, fixed_lbl).item()


            else:
                warped_bin = (warped_t > 0.5).float()  # (B,1,D,H,W)
                fixed_bin = (fixed > 0.5).float()  # (B,1,D,H,W)
                if warped_bin.shape != fixed_bin.shape:
                    warped_bin = F.interpolate(
                        warped_bin, size=fixed_bin.shape[2:], mode="nearest"
                    )
                dice = dice_metric(warped_bin, fixed_bin).item()

            sample_dice_scores.append(dice)
            print(f"[{i}] Volume {t}: DICE = {dice:.4f}")

        # Mean DICE for this sample
        mean_dice_sample = np.mean(sample_dice_scores)
        all_sample_means.append(mean_dice_sample)
        print(f"[{i}] Mean DICE (sample): {mean_dice_sample:.4f}")

        # -----------------------------
        # Save results to original subfolder
        # -----------------------------
        if "moving_path" not in batch:
            raise KeyError("Your DataGenerator must return 'moving_path' in each batch.")

        moving_path = batch["moving_path"][0]
        save_dir = os.path.dirname(moving_path)
        prefix = os.path.basename(os.path.dirname(save_dir))

        affine = nib.load(moving_path).affine.astype(np.float64)

        # Explicit squeezes (avoid dropping spatial dims by accident)
        warped_np = warped_all.squeeze(0).squeeze(0).cpu().numpy()  # (D,H,W,T)
        dx_np = dx.squeeze(0).squeeze(0).cpu().numpy()
        dy_np = dy.squeeze(0).squeeze(0).cpu().numpy()
        dz_np = dz.squeeze(0).squeeze(0).cpu().numpy()

        nib.save(nib.Nifti1Image(warped_np, affine), os.path.join(save_dir, f"{prefix}_moco.nii.gz"))
        nib.save(nib.Nifti1Image(dx_np, affine), os.path.join(save_dir, f"{prefix}_dx.nii.gz"))
        nib.save(nib.Nifti1Image(dy_np, affine), os.path.join(save_dir, f"{prefix}_dy.nii.gz"))
        nib.save(nib.Nifti1Image(dz_np, affine), os.path.join(save_dir, f"{prefix}_dz.nii.gz"))

        print(f"[{i}] Saved outputs to: {save_dir}")

        # -----------------------------
        # Save DICE scores to CSV
        # -----------------------------
        dice_csv_path = os.path.join(save_dir, f"{prefix}_dice_scores.csv")
        with open(dice_csv_path, "w") as f:
            f.write("volume,dice\n")
            for t, d in enumerate(sample_dice_scores):
                f.write(f"{t},{d:.6f}\n")
            f.write(f"mean,{mean_dice_sample:.6f}\n")

        print(f"[{i}] Saved DICE to: {dice_csv_path}")

# -----------------------------
# Print final mean DICE
# -----------------------------
final_mean_dice = np.mean(all_sample_means)
print(f"Mean DICE over {len(all_sample_means)} samples: {final_mean_dice:.4f}")

# -----------------------------
# Accumulate DICE tables in wide format (volume x case)
# -----------------------------
dice_data = {}
max_vols = 0

for i in range(len(test_loader.dataset)):
    moving_path = test_loader.dataset.data[i]["moving"]
    save_dir = os.path.dirname(moving_path)
    prefix = os.path.basename(os.path.dirname(save_dir))

    dice_csv_path = os.path.join(save_dir, f"{prefix}_dice_scores.csv")
    df = pd.read_csv(dice_csv_path)

    volume_dice = df[df["volume"] != "mean"]["dice"].astype(float).tolist()
    volume_dice.append(df[df["volume"] == "mean"]["dice"].values[0])
    dice_data[prefix] = volume_dice

    max_vols = max(max_vols, len(volume_dice))

# Create DataFrame: rows = volumes + mean, columns = cases
row_labels = [str(v) for v in range(max_vols - 1)] + ["mean"]
df_result = pd.DataFrame(index=row_labels)

for case, scores in dice_data.items():
    padded_scores = scores + [np.nan] * (max_vols - len(scores))  # pad if needed
    df_result[case] = padded_scores

# Save final table
testing_dir = os.path.join(base_path, "sourcedata")
csv_out = os.path.join(testing_dir, "testing", f"{order_execution}_dice_summary_volume_wise.csv")
df_result.index.name = "volume"
df_result.to_csv(csv_out)
print(f"Saved volume-wise wide DICE table to: {csv_out}")
