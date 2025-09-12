# -*- coding: utf-8 -*-
"""
Created on Mon Sep 1 2025
@author: Nontharat Tucksinapinunchai
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
import torch
torch.set_float32_matmul_precision("high")

import numpy as np
import nibabel as nib
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import optim
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from monai.data import load_decathlon_datalist
from monai.data import DataLoader,Dataset
from monai.networks.nets import VoxelMorphUNet
from monai.networks.blocks import Warp

start_time = time.ctime()
print("Start at:", start_time)

path = "/home/ge.polymtl.ca/p122983/nontharat/moco_dmri/"
data_path = "/home/ge.polymtl.ca/p122983/nontharat/moco_dmri/sourcedata/"
sys.path.insert(0,data_path)
json_path = os.path.join(data_path, 'dataset.json')

# -----------------------------
# DataGenerator
# -----------------------------
class DataGenerator(Dataset):
    def __init__(self, file_list):
        super().__init__(data=file_list)
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def normalize_volume(self, vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        vmin = np.percentile(vol, 1)
        vmax = np.percentile(vol, 99)
        vol = np.clip(vol, vmin, vmax)
        return (vol - vmin) / (vmax - vmin + eps)

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        moving_path = sample["moving"]
        fixed_path = sample["fixed"]

        # Moving: nib -> (H, W, D, T)
        moving_img = nib.load(moving_path)
        moving_img = nib.as_closest_canonical(moving_img)
        moving_np = moving_img.get_fdata()  # (H,W,D,T)
        moving_np = self.normalize_volume(moving_np)
        moving = torch.from_numpy(moving_np.astype(np.float32)).unsqueeze(0)  # (1,H,W,D,T)

        # Fixed: nib -> (H, W, D)
        fixed_img = nib.load(fixed_path)
        fixed_img = nib.as_closest_canonical(fixed_img)
        fixed_np = fixed_img.get_fdata()  # (H,W,D)
        fixed_np = self.normalize_volume(fixed_np)
        fixed = torch.from_numpy(fixed_np.astype(np.float32)).unsqueeze(0)  # (1,H,W,D)

        torch.cuda.empty_cache()
        return {"moving": moving, "fixed": fixed, "moving_path": moving_path, "fixed_path": fixed_path}

# -----------------------------
# DataModule
# -----------------------------
class DataModule(pl.LightningDataModule):
    def __init__(self, json_path, batch_size, num_workers):
        super().__init__()
        self.json_path = json_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load file lists from JSON
        self.train_files = load_decathlon_datalist(self.json_path, True, "training")
        self.val_files = load_decathlon_datalist(self.json_path, True, "validation")
        self.test_files = load_decathlon_datalist(self.json_path, True, "testing")

        # Build datasets
        self.train_ds = DataGenerator(self.train_files)
        self.val_ds = DataGenerator(self.val_files)
        self.test_ds = DataGenerator(self.test_files)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

# -----------------------------
# Smoothness regularization
# -----------------------------
def gradient_loss(flow_5d: torch.Tensor) -> torch.Tensor:
    # flow_5d: (B, 3, H, W, D)
    dz = torch.abs(flow_5d[:, :, :, :, 1:] - flow_5d[:, :, :, :, :-1])  # along D
    dy = torch.abs(flow_5d[:, :, :, 1:, :] - flow_5d[:, :, :, :-1, :])  # along W
    dx = torch.abs(flow_5d[:, :, 1:, :, :] - flow_5d[:, :, :-1, :, :])  # along H

    dz = F.pad(dz, (0, 1, 0, 0, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
    dx = F.pad(dx, (0, 0, 0, 0, 0, 1))
    return torch.mean(dx**2 + dy**2 + dz**2)

# -----------------------------
# L2 loss (MSE)
# -----------------------------
def l2_loss(warped_all: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
    # warped_all: (B,1,H,W,D,T)
    B, C, H, W, D, T = warped_all.shape
    if fixed.shape[2:] != (H, W, D):
        fixed = F.interpolate(fixed, size=(H, W, D), mode="trilinear", align_corners=False)

    def norm01(x, eps=1e-6):
        x_cpu = x.detach().cpu().numpy()
        vmin = np.percentile(x_cpu, 1)
        vmax = np.percentile(x_cpu, 99)
        x = (x - vmin) / (vmax - vmin + eps)
        return x

    fixed_n = norm01(fixed)  # (B,1,H,W,D)
    warped_n = norm01(warped_all)  # (B,1,H,W,D,T)
    fixed_T = fixed_n[..., None].expand(-1, -1, H, W, D, T)

    return F.mse_loss(torch.from_numpy(warped_n).float(), torch.from_numpy(fixed_T).float())

# -----------------------------
# Global Normalized Cross-Correlation
# -----------------------------
def global_ncc(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if a.shape != b.shape:
        b = b.expand_as(a)

    a_mean = a.mean(dim=(2, 3, 4), keepdim=True)
    b_mean = b.mean(dim=(2, 3, 4), keepdim=True)
    a = a - a_mean
    b = b - b_mean

    num = (a * b).mean(dim=(2, 3, 4), keepdim=True)
    den = torch.sqrt(
        (a**2).mean(dim=(2, 3, 4), keepdim=True) * (b**2).mean(dim=(2, 3, 4), keepdim=True)
    ) + eps
    return -(num / den).mean()

# -----------------------------
# Some helper function
# -----------------------------
def power(x, n=32):
    return ((int(x) + n - 1) // n) * n

# -----------------------------
# LightningModule
# -----------------------------
class VoxelMorphReg(pl.LightningModule):
    def __init__(self, lr=1e-4, lambda_smooth=50):
        super().__init__()
        self.lr = lr
        self.lambda_smooth = lambda_smooth

        self.unet = VoxelMorphUNet(
            spatial_dims=3,
            in_channels=2,
            unet_out_channels=3,                # flow (dx, dy, dz)
            channels=(16, 32, 32, 32, 32, 32),  # keep pairs; this is OK
            final_conv_channels=(16, 16),
            kernel_size=3,
            dropout=0.5
        )
        # Use border padding to reduce black edge artifacts
        self.transformer = Warp(mode="bilinear", padding_mode="border")

    def forward(self, moving, fixed):
        """
        moving: (B, 1, H, W, D, T)
        fixed:  (B, 1, H, W, D)
        returns:
          warped_all:     (B, 1, H*, W*, D*, T)
          flows_4d:       (B, 3, H*, W*, D*, T)
          (dx, dy, dz):   each (B, 1, H*, W*, D*, T)
        """
        B, C, H, W, D, T = moving.shape

        # Make sizes UNet-friendly
        target_H = power(H, 32)
        target_W = power(W, 32)
        target_D = power(D, 32)

        # Pre-resize fixed once
        fixed_res = F.interpolate(fixed, size=(target_H, target_W, target_D),
                                  mode="trilinear", align_corners=False)

        warped_list = []
        flow_list = []

        for t in range(T):
            moving_t = moving[..., t]  # (B,1,H,W,D)
            moving_t_res = F.interpolate(moving_t, size=(target_H, target_W, target_D),
                                         mode="trilinear", align_corners=False)

            x = torch.cat([moving_t_res, fixed_res], dim=1)  # (B,2,H,W,D)
            flow = self.unet(x)  # (B,3,H,W,D)
            warped = self.transformer(moving_t_res, flow)  # (B,1,H,W,D)

            warped_list.append(warped)
            flow_list.append(flow)

        warped_all = torch.stack(warped_list, dim=-1)  # (B,1,H*,W*,D*,T)
        flows_4d = torch.stack(flow_list, dim=-1)  # (B,3,H*,W*,D*,T)

        return warped_all, flows_4d

    def training_step(self, batch, batch_idx):
        moving, fixed = batch["moving"], batch["fixed"]
        warped_all, flows_4d = self(moving, fixed)

        B, _, H, W, D, T = warped_all.shape
        if fixed.shape[2:] != (H, W, D):
            fixed = F.interpolate(fixed, size=(H, W, D), mode="trilinear", align_corners=False)
        # Expand fixed across time
        fixed_T = fixed.unsqueeze(-1).expand_as(warped_all)

        loss_l2 = l2_loss(warped_all, fixed)
        loss_ncc = global_ncc(warped_all, fixed_T)
        loss_sim = 0.25 * loss_l2 + 0.75 * loss_ncc

        flows_bt = flows_4d.permute(0, 5, 1, 2, 3, 4).reshape(B * T, 3, H, W, D)
        loss_smooth = gradient_loss(flows_bt)

        loss = loss_sim + self.lambda_smooth * loss_smooth

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_l2_loss", loss_l2)
        self.log("train_ncc_loss", loss_ncc)
        self.log("train_sim_loss", loss_sim)
        self.log("train_smooth_loss", loss_smooth)

        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        moving, fixed = batch["moving"], batch["fixed"]
        warped_all, flows_4d = self(moving, fixed)

        B, _, H, W, D, T = warped_all.shape
        if fixed.shape[2:] != (H, W, D):
            fixed = F.interpolate(fixed, size=(H, W, D), mode="trilinear", align_corners=False)
        # Expand fixed across time
        fixed_T = fixed.unsqueeze(-1).expand_as(warped_all)

        loss_l2 = l2_loss(warped_all, fixed)
        loss_ncc = global_ncc(warped_all, fixed_T)
        loss_sim = 0.25 * loss_l2 + 0.75 * loss_ncc

        flows_bt = flows_4d.permute(0, 5, 1, 2, 3, 4).reshape(B * T, 3, H, W, D)
        loss_smooth = gradient_loss(flows_bt)

        loss = loss_sim + self.lambda_smooth * loss_smooth

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_l2_loss", loss_l2)
        self.log("val_ncc_loss", loss_ncc)
        self.log("val_sim_loss", loss_sim)
        self.log("val_smooth_loss", loss_smooth)

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
# logger (Weights & Biases)
# -----------------------------
order_execution = sys.argv[1]
num_epochs = 25
batch_size = 1
lr = 1e-4
lambda_smooth = 50
num_workers = 15

# -----------------------------
# callbacks
# -----------------------------
ckpt_dir = os.path.join(path, 'trained_weights')
os.makedirs(ckpt_dir, exist_ok=True)
pretrained_ckpt = os.path.join(ckpt_dir, f"{order_execution}_voxelmorph_best-weighted.ckpt")

checkpoint_cb = ModelCheckpoint(
    dirpath=ckpt_dir,
    filename=f"{order_execution}_voxelmorph_best-weighted",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_weights_only=True
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")
early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=25,
    verbose=True
)

if os.path.exists(pretrained_ckpt):
    print(f"Starting NEW training, initialized from: {pretrained_ckpt}")
    model = VoxelMorphReg.load_from_checkpoint(
        pretrained_ckpt,
        lr=lr,
        lambda_smooth=lambda_smooth
    )
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
    wandb_logger = WandbLogger(project="moco-dmri", name=f"{order_execution}_VoxelMorphUNet")
    wandb_config = wandb_logger.experiment.config
    wandb_config.update({
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lambda_smooth": lambda_smooth,
        "data_path": data_path
    })
    dm = DataModule(json_path=json_path, batch_size=batch_size, num_workers=num_workers)
    model = VoxelMorphReg(lr=lr, lambda_smooth=lambda_smooth)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor, early_stop],
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=5,
        deterministic=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True
    )

    # -----------------------------
    # fit
    # -----------------------------
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