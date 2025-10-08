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
from monai.data import DataLoader, Dataset
from monai.networks.nets import VoxelMorphUNet
from monai.networks.blocks import Warp

start_time = time.ctime()
print("Start at:", start_time)

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
    new_size = (new_H, new_W, new_D)

    # Align corners only if interpolation mode supports it
    if mode in ["linear", "bilinear", "bicubic", "trilinear"]:
        x_resampled = F.interpolate(x, size=new_size, mode=mode, align_corners=False)
    else:
        x_resampled = F.interpolate(x, size=new_size, mode=mode)
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

def normalize_meanstd(x, mask, eps=1e-6):
    """
    Normalize within mask using mean/std instead of quantiles for stability.
    """
    m = mask > 0
    mean = (x[m]).mean() if m.any() else 0
    std = (x[m]).std() + eps
    x = (x - mean) / std
    return x * mask

def _get_fixed_t(fixed: torch.Tensor, t: int) -> torch.Tensor:
    """
    Return fixed volume (B,1,H,W,D) at time t,
    supporting both 3D fixed (fMRI) and 4D fixed (dMRI).
    """
    if fixed.ndim == 5:  # (B,1,H,W,D)
        return fixed.contiguous()  # same ref for all timepoints, no gradient
    if fixed.ndim == 6:
        if isinstance(t, torch.Tensor):
            t = int(t.item())
        return fixed[..., t].contiguous()
    else:
        raise ValueError(f"Unexpected fixed ndim={fixed.ndim}")

# -----------------------------
# Loss function
# -----------------------------
# L2 loss (MSE)
def l2_loss(a, b, mask):
    """
    Compute Mean squared error loss inside the mask
    """
    m = (mask > 0).float()
    diff = (a - b) * m
    l2 = (diff.pow(2).sum() / (m.sum() + 1e-6))
    return l2

# Global Normalized Cross-Correlation (GNCC)
def gncc_loss(a, b, mask, eps=1e-6):
    """
    Compute Global NCC loss inside mask
    """
    m = (mask > 0).float()
    a = normalize_meanstd(a, m)
    b = normalize_meanstd(b, m)
    num = ((a * b) * m).sum()
    den = torch.sqrt(((a ** 2) * m).sum() * ((b ** 2) * m).sum() + eps)
    return -(num / den)

# Similarlity loss (L2+GNCC)
def similarity_loss(warped_all, fixed, mask):
    """
    Average masked L2+GNCC loss across timepoints
        a, b:   (B,1,H,W,D,T)
        mask:   (B,1,H,W,D)
    """
    B, _, _, _, _, T = warped_all.shape
    total = 0
    for t in range(T):
        w_t, f_t = warped_all[..., t], _get_fixed_t(fixed, t)
        w_t, f_t = normalize_meanstd(w_t, mask), normalize_meanstd(f_t, mask)
        total += 0.5 * l2_loss(w_t, f_t, mask) + 1.0 * gncc_loss(w_t, f_t, mask)
    return total / T

# -----------------------------
# LightningModule
# -----------------------------
class VoxelMorphReg(pl.LightningModule):
    def __init__(self, lr=1e-4, lambda_smooth=0.01, flow_scale=3.0):
        super().__init__()
        self.lr = lr
        self.lambda_smooth = lambda_smooth
        self.flow_scale = flow_scale

        # UNet now outputs 2 channels = Tx, Ty
        self.unet = VoxelMorphUNet(
            spatial_dims=3,
            in_channels=2,
            unet_out_channels=2,
            channels=(16, 32, 32, 32,),
            final_conv_channels=(16,),
            kernel_size=3,
        )
        self.transformer = Warp(mode="bilinear", padding_mode="border")

    def forward(self, moving, fixed, mask):
        """
        moving: (B, 1, H, W, D, T)
        fixed:  (B, 1, H, W, D) or (B, 1, H, W, D, T) --> the model will handle both 3D and 4D
                - dMRI: (B,1,H,W,D,T) -> mean b0 and mean dwi over timepoints
                - fMRI: (B,1,H,W,D)   -> static mean fMRI
        mask:  (B, 1, H, W, D)
        returns:
          warped: (B, 1, H, W, D, T)
          Tx_all, Ty_all, Tz_all : (B, 1, D, T)
        """
        B, C, H, W, D, T = moving.shape
        warped_list, Tx_list, Ty_list, Tz_list = [], [], [], []

        # Precompute mask once
        mask_res, _ = resample_to_multiple(mask, multiple=32, mode="nearest")

        for t in range(T):
            mov_t = moving[..., t]  # (B,1,H,W,D)
            fix_t = _get_fixed_t(fixed, t)  # fMRI → static or dMRI → time-varying

            # Resample to UNet-compatible shape
            mov_t_res, orig_size = resample_to_multiple(mov_t, multiple=32, mode="trilinear")
            fix_t_res, _ = resample_to_multiple(fix_t, multiple=32, mode="trilinear")

            # Downsample before UNet (OOM control)
            mov_ds = F.interpolate(mov_t_res, scale_factor=0.5, mode="trilinear", align_corners=False)
            fix_ds = F.interpolate(fix_t_res, scale_factor=0.5, mode="trilinear", align_corners=False)
            mask_ds = F.interpolate(mask_res, scale_factor=0.5, mode="nearest")

            # Normalize at the same resolution as the network input
            mov_t_norm_ds = normalize_meanstd(mov_ds, mask_ds)
            fix_t_norm_ds = normalize_meanstd(fix_ds, mask_ds)

            # UNet input
            x = torch.cat([mov_t_norm_ds, fix_t_norm_ds], dim=1)

            # Predict Tx, Ty
            flow_ds = torch.tanh(self.unet(x)) * self.flow_scale  # UNet

            # Upsample Tx/Ty back to original resolution
            flow = F.interpolate(flow_ds, size=mov_t_res.shape[2:], mode="trilinear", align_corners=False)

            # Mean over H,W → per-slice translation
            flow_mean = flow.mean(dim=(2, 3))  # (B, 2, D)
            Tx = flow_mean[:, 0, :]
            Ty = flow_mean[:, 1, :]

            orig_D = mov_t.shape[-1]
            if Tx.shape[-1] != orig_D:
                Tx = F.interpolate(Tx.unsqueeze(1), size=orig_D, mode="linear", align_corners=False).squeeze(1)
                Ty = F.interpolate(Ty.unsqueeze(1), size=orig_D, mode="linear", align_corners=False).squeeze(1)

            # Build rigid-like 3D field (Z displacement = 0)
            B_, _, H, W, D = mov_t.shape
            flow_field = torch.zeros((B_, 3, H, W, D), device=mov_t.device, dtype=mov_t.dtype)
            flow_field[:, 0] = Tx[:, None, None, :].expand(B_, H, W, D)
            flow_field[:, 1] = Ty[:, None, None, :].expand(B_, H, W, D)
            # flow_field[:, 2] = 0 (no Z motion)

            warped = self.transformer(mov_t, flow_field)
            warped_list.append(warped)

            Tx_list.append(Tx)
            Ty_list.append(Ty)

            del mov_t, fix_t, mov_t_res, fix_t_res, mov_ds, fix_ds, mask_ds, flow_ds, flow, flow_field

        warped_all = torch.stack(warped_list, dim=-1)
        Tx_all = torch.stack(Tx_list, dim=-1).unsqueeze(1)
        Ty_all = torch.stack(Ty_list, dim=-1).unsqueeze(1)

        return warped_all, Tx_all, Ty_all

    def training_step(self, batch, batch_idx):
        moving, fixed, mask = batch["moving"], batch["fixed"], batch["mask"]
        warped, Tx, Ty = self(moving, fixed, mask)

        loss_sim = similarity_loss(warped, fixed, mask)

        # Smoothness only across Z and T for Tx, Ty
        dTx_z = Tx.diff(dim=2).pow(2).mean()
        dTy_z = Ty.diff(dim=2).pow(2).mean()
        dTx_t = Tx.diff(dim=3).pow(2).mean()
        dTy_t = Ty.diff(dim=3).pow(2).mean()
        loss_smooth = (dTx_z + dTy_z + dTx_t + dTy_t) / 4.0

        loss = loss_sim + self.lambda_smooth * loss_smooth

        B = moving.size(0)
        self.log("train_loss", loss, prog_bar=True, batch_size=B)
        self.log("train_sim_loss", loss_sim, batch_size=B)
        self.log("train_smooth_loss", loss_smooth, batch_size=B)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            moving, fixed, mask = batch["moving"], batch["fixed"], batch["mask"]
            warped, Tx, Ty = self(moving, fixed, mask)

            loss_sim = similarity_loss(warped, fixed, mask)
            dTx_z = Tx.diff(dim=2).pow(2).mean()
            dTy_z = Ty.diff(dim=2).pow(2).mean()
            dTx_t = Tx.diff(dim=3).pow(2).mean()
            dTy_t = Ty.diff(dim=3).pow(2).mean()
            loss_smooth = (dTx_z + dTy_z + dTx_t + dTy_t) / 4.0

            loss = loss_sim + self.lambda_smooth * loss_smooth
            B = moving.size(0)
            self.log("val_loss", loss, prog_bar=True, batch_size=B)
            self.log("val_sim_loss", loss_sim, batch_size=B)
            self.log("val_smooth_loss", loss_smooth, batch_size=B)
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
# Parse CLI args
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python moco_main.py <base_path> <data_path> <run_name1> [<run_name2>]")
        sys.exit(1)

    base_path = sys.argv[1]  # e.g. /home/.../moco_dmri/
    data_path = sys.argv[2]  # e.g. /home/.../prepared/dmri_dataset
    order_execution_1 = sys.argv[3]  # run name e.g. yyyymmdd_order_voxelmorph_best-weighted
    order_execution_2 = sys.argv[4] if len(sys.argv) > 4 else None  # format like order_execution_1

    json_path = os.path.join(data_path, "dataset.json")

    print("Base path   :", base_path)
    print("Data folder :", data_path)
    print("JSON path   :", json_path)

    # -----------------------------
    # Training setup
    # -----------------------------
    num_epochs = 50
    batch_size = 1
    lr = 1e-4
    lambda_smooth = 0.01
    flow_scale = 3.0
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
        model = VoxelMorphReg.load_from_checkpoint(pretrained_ckpt, lr=lr, lambda_smooth=lambda_smooth,
                                                   flow_scale=flow_scale)
    else:
        print("No pretrained checkpoint found, training from scratch.")
        model = VoxelMorphReg(lr=lr, lambda_smooth=lambda_smooth, flow_scale=flow_scale)

    # -----------------------------
    # trainer
    # -----------------------------
    import wandb
    from pytorch_lightning.loggers import WandbLogger

    wandb.login(key="ab48d9c3a5dee9883bcb676015f2487c1bc51f74")
    wandb_logger = WandbLogger(project="moco-dmri", name=f"{order_execution_1}")
    # If continue with the pretrained_ckpt, resume logs to the same wandb run
    # wandb_logger = WandbLogger(project="moco-dmri", name=f"{order_execution_1}", id="9on2ss9i", resume="must")
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
        precision="16-mixed",
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