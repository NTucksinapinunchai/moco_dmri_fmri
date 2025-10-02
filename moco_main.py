# -*- coding: utf-8 -*-
"""
Created on Mon Sep 1 2025
@author: Nontharat Tucksinapinunchai

This script used to train a deep learning model for motion correction in dMRI or fMRI.
It uses a VoxelMorphUNet to register 4D moving volumes to a fixed reference guided by a mask.
The model combines L2 (MSE) and GNCC losses across time, with smoothness regularization and flow scaling.

Key features:
- Handles both 4D moving + 4D fixed (dMRI) or 4D moving + 3D fixed (fMRI) with mask inputs
- Mixed similarity loss (L2 + GNCC) computed across timepoints
- Flow scaling and resampling to make input compatible with UNet architecture
- Normalization inside mask using 1%–99% percentiles
- Saves best weights, logs training/validation losses, and integrates with W&B

Usage:
------
In terminal/command line:

    python moco_main.py /path/to/base path/to/data run1 [run2]

Arguments:
    /path/to/base       : base directory containing the script and trained_weights
    /path/to/data    : data directory depending on training dataset
    run1                : identifier for this training run (checkpoint filename)
    run2 (opt)          : if provided, fine-tune or continue from run1 but save under run2 name

"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import sys
import time
import torch
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

import json
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import optim
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from monai.data import DataLoader,Dataset
from monai.networks.nets import VoxelMorphUNet
from monai.networks.blocks import Warp

start_time = time.ctime()
print("Start at:", start_time)

# -----------------------------
# Parse CLI args
# -----------------------------
if len(sys.argv) < 3:
    print("Usage: python main.py <base_path> <data_subfolder> <run_name1> [<run_name2>]")
    print("Example: python main.py /path/to/project fmri_dataset run1 [run2]")
    sys.exit(1)

base_path = sys.argv[1]                          # e.g. /home/.../moco_dmri/
data_path = sys.argv[2]                          # e.g. /home/.../prepared/dmri_dataset
order_execution_1 = sys.argv[3]                  # run name e.g. yyyymmdd_order_voxelmorph_best-weighted
order_execution_2 = sys.argv[4] if len(sys.argv) > 4 else None      # format like order_execution_1

json_path = os.path.join(data_path, "dataset.json")

print("Base path   :", base_path)
print("Data folder :", data_path)
print("JSON path   :", json_path)

# -----------------------------
# DataGenerator
# -----------------------------
class DataGenerator(Dataset):
    def __init__(self, file_list, data_dir):
        self.file_list = file_list
        self.data_dir = data_dir

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        pt_path = sample["data"]

        # If absolute path in JSON
        if os.path.isabs(pt_path):
            final_path = pt_path
        else:
            final_path = os.path.join(self.data_dir, pt_path)

        if not os.path.exists(final_path):
            raise FileNotFoundError(f"Could not find file: {final_path}")

        data = torch.load(final_path, weights_only=False)

        moving = data["moving"]  # (B,1,H,W,D,T)
        fixed = data["fixed"]  # (B,1,H,W,D) [fMRI] or (B,1,H,W,D,T) [dMRI]
        mask = data["mask"]
        affine = data["affine"]

        # --- Only apply subsampling if fMRI ---
        if fixed.ndim == 5:  # fMRI case
            T = moving.shape[-1]
            max_T = max(100, T//2)
            if T > max_T:
                step = T // max_T
                idxs = torch.arange(0, T, step)[:max_T]

                # subsample only moving (fixed stays as static reference)
                moving = moving[..., idxs]

        return {"moving": moving, "fixed": fixed, "mask": mask,
                "affine": affine, "data_path": pt_path}

# -----------------------------
# DataModule
# -----------------------------
class DataModule(pl.LightningDataModule):
    def __init__(self, json_path, data_dir, batch_size, num_workers):
        super().__init__()
        self.json_path = json_path
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load JSON
        with open(self.json_path, "r") as f:
            dataset_dict = json.load(f)

        self.train_ds = DataGenerator(dataset_dict["training"], self.data_dir)
        self.val_ds = DataGenerator(dataset_dict["validation"], self.data_dir)
        self.test_ds = DataGenerator(dataset_dict["testing"], self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

# -----------------------------
# Some helper function
# -----------------------------
def resample_to_multiple(x, multiple=32, mode="trilinear"):
    """
    Resample tensor to nearest multiple of given size using interpolation.
    """
    B, C, H, W, D = x.shape
    new_H = ((H + multiple - 1) // multiple) * multiple
    new_W = ((W + multiple - 1) // multiple) * multiple
    new_D = ((D + multiple - 1) // multiple) * multiple

    if mode in ["trilinear", "bilinear", "linear"]:
        x_resampled = F.interpolate(x, size=(new_H, new_W, new_D),
                                    mode=mode, align_corners=False)
    else:
        x_resampled = F.interpolate(x, size=(new_H, new_W, new_D),
                                    mode=mode)
    return x_resampled, (H, W, D)

def resample_back(x, orig_size, mode="trilinear"):
    """
    Resample tensor back to original size after UNet processing.
    """
    H, W, D = orig_size
    if mode in ["trilinear", "bilinear", "linear"]:
        return F.interpolate(x, size=(H, W, D), mode=mode, align_corners=False)
    else:
        return F.interpolate(x, size=(H, W, D), mode=mode)

def resample_flow_back(flow, orig_size, resampled_size, mode="trilinear"):
    """
    Resample and scale flow fields back to original resolution.
    """
    H, W, D = orig_size
    Hres, Wres, Dres = resampled_size

    # Resample flow as if it was an image
    if mode in ["trilinear", "bilinear", "linear"]:
        flow_back = F.interpolate(flow, size=(H, W, D), mode=mode, align_corners=False)
    else:
        flow_back = F.interpolate(flow, size=(H, W, D), mode=mode)

    scale_factors = (
        H / Hres,   # scale for axis 0 (x)
        W / Wres,   # scale for axis 1 (y)
        D / Dres    # scale for axis 2 (z)
    )
    scale = torch.tensor(scale_factors, device=flow.device, dtype=flow.dtype).view(1, 3, 1, 1, 1)

    return flow_back * scale

def normalize_volume(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize x within mask using 1%–99% percentiles per (B,C).
    Outside the mask, values are set to 0.
    """
    # Flatten spatial dims → (B, C, N)
    flat = x.flatten(start_dim=2)
    flat_mask = mask.flatten(start_dim=2)

    # Masked values only
    masked_vals = torch.where(flat_mask > 0, flat, torch.nan)

    # Compute per-(B,C) percentiles inside mask
    vmin = torch.nanquantile(masked_vals, 0.01, dim=-1, keepdim=True)  # (B,C,1)
    vmax = torch.nanquantile(masked_vals, 0.99, dim=-1, keepdim=True)  # (B,C,1)

    # Reshape to broadcast
    spatial_nd = x.ndim - 2
    shape = list(vmin.shape[:2]) + [1] * spatial_nd
    vmin = vmin.view(*shape)
    vmax = vmax.view(*shape)

    # Normalize inside mask
    x_norm = torch.clamp(x, min=vmin, max=vmax)
    x_norm = (x_norm - vmin) / (vmax - vmin + eps)

    # Apply mask: keep normalized inside, zero outside
    out = torch.where(mask > 0, x_norm, torch.zeros_like(x))

    return out

def _get_fixed_t(fixed: torch.Tensor, t: int) -> torch.Tensor:
    """
    Return fixed volume (B,1,H,W,D) at time t,
    supporting both 3D fixed (fMRI) and 4D fixed (dMRI).
    """
    if fixed.ndim == 5:             # (B,1,H,W,D)
        return fixed.detach()       # same ref for all timepoints, no gradient
    elif fixed.ndim == 6:           # (B,1,H,W,D,T)
        return fixed[..., t]
    else:
        raise ValueError(f"Unexpected fixed ndim={fixed.ndim}")

# -----------------------------
# Smoothness regularization
# -----------------------------
def gradient_loss(flow_5d: torch.Tensor) -> torch.Tensor:
    """
    Compute smoothness regularization loss on flow fields (dx,dy,dz).
    """
    dx = flow_5d[:, :, 1:, :, :] - flow_5d[:, :, :-1, :, :]
    dy = flow_5d[:, :, :, 1:, :] - flow_5d[:, :, :, :-1, :]
    dz = flow_5d[:, :, :, :, 1:] - flow_5d[:, :, :, :, :-1]

    loss = (dx.pow(2).mean() +
            dy.pow(2).mean() +
            dz.pow(2).mean()) / 3.0
    return loss

# -----------------------------
# L2 loss (MSE)
# -----------------------------
def l2_loss(warped_all: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute L2 loss across timepoints, inside the mask.
        warped_all: (B,1,H,W,D,T)
        fixed:      (B,1,H,W,D,T)
        mask:       (B,1,H,W,D)
    """
    B, C, H, W, D, T = warped_all.shape
    losses = []
    for t in range(T):
        warped_t = warped_all[..., t]        # (B,1,H,W,D)
        fixed_t  = _get_fixed_t(fixed, t)    # (B,1,H,W,D)
        mask_t   = mask                      # (B,1,H,W,D)

        warped_n = normalize_volume(warped_t, mask_t) * mask_t
        fixed_n  = normalize_volume(fixed_t,  mask_t) * mask_t
        losses.append(F.mse_loss(warped_n, fixed_n))

    return torch.stack(losses).mean()

# -----------------------------
# Global Normalized Cross-Correlation (GNCC)
# -----------------------------
def global_ncc(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute Global normalized cross-correlation across timepoints.
        a, b:   (B,1,H,W,D,T)
        mask:   (B,1,H,W,D)
    """
    B, C, H, W, D, T = a.shape
    vals = []
    for t in range(T):
        a_t = a[..., t]
        b_t = _get_fixed_t(b, t)
        m_t = mask

        a_t = normalize_volume(a_t, m_t) * m_t
        b_t = normalize_volume(b_t, m_t) * m_t

        a_mean = a_t.mean(dim=(2,3,4), keepdim=True)
        b_mean = b_t.mean(dim=(2,3,4), keepdim=True)
        a0 = a_t - a_mean
        b0 = b_t - b_mean

        num = (a0 * b0).mean(dim=(2,3,4), keepdim=True)
        den = torch.sqrt((a0**2).mean(dim=(2,3,4), keepdim=True) * (b0**2).mean(dim=(2,3,4), keepdim=True)) + eps
        vals.append(-(num / den).mean())

    return torch.stack(vals).mean()
# -----------------------------
# LightningModule
# -----------------------------
class VoxelMorphReg(pl.LightningModule):
    def __init__(self, lr=1e-4, lambda_smooth=0.01, flow_scale=3.0):
        super().__init__()
        self.lr = lr
        self.lambda_smooth = lambda_smooth
        self.flow_scale = flow_scale

        self.unet = VoxelMorphUNet(
            spatial_dims=3,
            in_channels=2,
            unet_out_channels=3,                # flow (dx, dy, dz)
            channels=(16, 32, 32, 32, ),          # keep pairs; this is OK
            final_conv_channels=(16, ),
            kernel_size=3,
            # dropout=0.5
        )
        # Use border padding to reduce black edge artifacts
        self.transformer = Warp(mode="bilinear", padding_mode="border")

    def forward(self, moving, fixed, mask):
        """
        moving: (B, 1, H, W, D, T)
        fixed:  (B, 1, H, W, D) or (B, 1, H, W, D, T) --> the model will handle both 3D and 4D
                - dMRI: (B,1,H,W,D,T) -> mean b0 and mean dwi over timepoints
                - fMRI: (B,1,H,W,D)   -> static mean fMRI
        mask:  (B, 1, H, W, D)
        returns:
          warped_all_raw: (B, 1, H, W, D, T)
          flows_4d:       (B, 3, H, W, D, T)
        """
        B, C, H, W, D, T = moving.shape
        warped_list, flow_list = [], []

        # Precompute mask once
        mask_res, _ = resample_to_multiple(mask, multiple=32, mode="nearest")
        mask_bin = (mask_res > 0.5).float()

        for t in range(T):
            mov_t_raw = moving[..., t]          # (B,1,H,W,D)
            fix_t_raw = _get_fixed_t(fixed, t)  # fMRI → static or dMRI → time-varying

            # Resample to make a compatible shape
            mov_t_res, orig_size = resample_to_multiple(mov_t_raw, multiple=32, mode="trilinear")
            fix_t_res, _ = resample_to_multiple(fix_t_raw, multiple=32, mode="trilinear")

            # Downsample before UNet
            mov_t_ds = F.interpolate(mov_t_res, scale_factor=0.5, mode="trilinear", align_corners=False)
            fix_t_ds = F.interpolate(fix_t_res, scale_factor=0.5, mode="trilinear", align_corners=False)
            mask_ds = F.interpolate(mask_res, scale_factor=0.5, mode="nearest")

            mask_bin_ds = (mask_ds > 0.5).float()

            # Normalize inside mask
            mov_t_norm = normalize_volume(mov_t_ds, mask=mask_bin_ds) * mask_bin_ds
            fix_t_norm = normalize_volume(fix_t_ds, mask=mask_bin_ds) * mask_bin_ds

            # UNet input
            x = torch.cat([mov_t_norm, fix_t_norm], dim=1)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                flow_ds = self.unet(x)
            flow_ds = torch.tanh(flow_ds) * self.flow_scale

            # Warp moving
            warped_ds = self.transformer(mov_t_ds, flow_ds)

            # Upsample back to original resolution
            warped_res = F.interpolate(warped_ds, size=mov_t_res.shape[2:], mode="trilinear", align_corners=False)
            flow_res = F.interpolate(flow_ds, size=mov_t_res.shape[2:], mode="trilinear", align_corners=False)

            # Back to exact original size
            warped_back = resample_back(warped_res, orig_size, mode="trilinear")
            flow_back = resample_flow_back(flow_res, orig_size, mov_t_res.shape[2:], mode="trilinear")

            warped_list.append(warped_back)
            flow_list.append(flow_back)

            # free intermediates early
            del mov_t_raw, fix_t_raw, mov_t_res, fix_t_res
            del mov_t_ds, fix_t_ds, warped_ds, flow_ds, warped_res, flow_res
            torch.cuda.empty_cache()

        warped_all = torch.stack(warped_list, dim=-1)   # (B,1,H,W,D,T)
        flows_4d = torch.stack(flow_list, dim=-1)       # (B,3,H,W,D,T)

        return warped_all, flows_4d

    def training_step(self, batch, batch_idx):
        moving, fixed, mask = batch["moving"], batch["fixed"], batch["mask"]
        warped_all, flows_4d = self(moving, fixed, mask)

        loss_l2 = l2_loss(warped_all, fixed, mask)
        loss_ncc = global_ncc(warped_all, fixed, mask)
        loss_sim = 1 * loss_l2 + 1 * loss_ncc

        B, _, H, W, D, T = warped_all.shape
        loss_smooth = 0.0
        for t in range(T):
            flow_t = flows_4d[..., t]
            loss_smooth = loss_smooth + gradient_loss(flow_t)
        loss_smooth = loss_smooth / T

        loss = loss_ncc + self.lambda_smooth * loss_smooth

        self.log("train_loss", loss, prog_bar=True, batch_size=B)
        self.log("train_l2_loss", loss_l2, batch_size=B)
        self.log("train_ncc_loss", loss_ncc, batch_size=B)
        self.log("train_sim_loss", loss_sim, batch_size=B)
        self.log("train_smooth_loss", loss_smooth, batch_size=B)

        del warped_all, flows_4d
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            moving, fixed, mask = batch["moving"], batch["fixed"], batch["mask"]
            warped_all, flows_4d = self(moving, fixed, mask)

            loss_l2 = l2_loss(warped_all, fixed, mask)
            loss_ncc = global_ncc(warped_all, fixed, mask)
            loss_sim = 1 * loss_l2 + 1 * loss_ncc

            B, _, H, W, D, T = warped_all.shape
            loss_smooth = 0.0
            for t in range(T):
                flow_t = flows_4d[..., t]
                loss_smooth = loss_smooth + gradient_loss(flow_t)
            loss_smooth = loss_smooth / T

            loss = loss_ncc + self.lambda_smooth * loss_smooth

            self.log("val_loss", loss, prog_bar=True, batch_size=B)
            self.log("val_l2_loss", loss_l2, batch_size=B)
            self.log("val_ncc_loss", loss_ncc, batch_size=B)
            self.log("val_sim_loss", loss_sim, batch_size=B)
            self.log("val_smooth_loss", loss_smooth, batch_size=B)

            del warped_all, flows_4d
            torch.cuda.empty_cache()
            return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

# -----------------------------
# Training setup
# -----------------------------
num_epochs = 50
batch_size = 1
lr = 1e-4
lambda_smooth = 0.01
num_workers = 8

# -----------------------------
# callbacks
# -----------------------------
ckpt_dir = os.path.join(base_path, 'trained_weights')
os.makedirs(ckpt_dir, exist_ok=True)

if order_execution_2:
    pretrained_ckpt = os.path.join(ckpt_dir, f"{order_execution_1}.ckpt")
    ckpt_name = f"{order_execution_2}"
else:
    pretrained_ckpt = None
    ckpt_name = f"{order_execution_1}"

checkpoint_cb = ModelCheckpoint(
    dirpath=ckpt_dir,
    filename=ckpt_name,
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_weights_only=False
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=25, verbose=True)

# Load only model weights (use checkpoint as initialization for a new run)
if pretrained_ckpt is not None and os.path.exists(pretrained_ckpt):
    print(f"Starting NEW training, initialized from: {pretrained_ckpt}")
    model = VoxelMorphReg.load_from_checkpoint(pretrained_ckpt, lr=lr, lambda_smooth=lambda_smooth)
else:
    print("No pretrained checkpoint found, training from scratch.")
    model = VoxelMorphReg(lr=lr, lambda_smooth=lambda_smooth)

# -----------------------------
# trainer
# -----------------------------
if __name__ == "__main__":
    import wandb
    from pytorch_lightning.loggers import WandbLogger

    wandb.login(key="ab48d9c3a5dee9883bcb676015f2487c1bc51f74")
    wandb_logger = WandbLogger(project="moco-dmri", name=f"{order_execution_1}")
    # If continue with the pretrained_ckpt, resume logs to the same wandb run
    # wandb_logger = WandbLogger(project="moco-dmri", name=f"{order_execution_2}", id="x0u9ms56", resume="must")
    wandb_config = wandb_logger.experiment.config
    wandb_config.update({
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lambda_smooth": lambda_smooth,
        "data_path": data_path
    }, allow_val_change=True)

    dm = DataModule(json_path=json_path, data_dir=data_path, batch_size=batch_size, num_workers=num_workers)
    gc.collect()
    torch.cuda.empty_cache()
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor, early_stop],
        accelerator="gpu",
        devices=1,
        precision=16,
        log_every_n_steps=10,
        deterministic=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True
    )

    # -----------------------------
    # Fit model
    # -----------------------------
    # Load full checkpoint (resume training exactly where it left off)
    # if os.path.exists(pretrained_ckpt):
    #     print(f"Resuming training from: {pretrained_ckpt}")
    #     trainer.fit(model, datamodule=dm, ckpt_path=pretrained_ckpt)
    # else:
    #     print("No checkpoint found, training from scratch.")
    trainer.fit(model, datamodule=dm)

    # -----------------------------
    # Best checkpoint info
    # -----------------------------
    print("Best checkpoint path:", checkpoint_cb.best_model_path)
    if checkpoint_cb.best_model_score is not None:
        best_val_loss = checkpoint_cb.best_model_score.item()
        print("Best val_loss:", best_val_loss)
    else:
        best_val_loss = None
        print("No checkpoint was saved yet.")

    wandb_logger.log_metrics({
        "best_val_loss": best_val_loss,
        "best_checkpoint_path": checkpoint_cb.best_model_path
    })

end_time = time.ctime()
print("End at:", end_time)