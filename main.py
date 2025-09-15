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

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        moving_path = sample["moving"]
        fixed_path = sample["fixed"]
        mask_path = sample["mask"]

        # Moving: (H, W, D, T)
        moving_img = nib.load(moving_path)
        moving_img = nib.as_closest_canonical(moving_img)
        moving_np = moving_img.get_fdata()
        moving = torch.from_numpy(moving_np.astype(np.float32)).unsqueeze(0)  # (1,H,W,D,T)

        # Fixed: (H, W, D, T)
        fixed_img = nib.load(fixed_path)
        fixed_img = nib.as_closest_canonical(fixed_img)
        fixed_np = fixed_img.get_fdata()
        fixed = torch.from_numpy(fixed_np.astype(np.float32)).unsqueeze(0)  # (1,H,W,D,T)

        # Mask: (H, W, D)
        mask_img = nib.load(mask_path)
        mask_img = nib.as_closest_canonical(mask_img)
        mask_np = mask_img.get_fdata()
        mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)  # (1,H,W,D)

        torch.cuda.empty_cache()
        return {"moving": moving, "fixed": fixed, "mask": mask,
                "moving_path": moving_path, "fixed_path": fixed_path, "mask_path": mask_path}

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
def l2_loss(warped_all: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, C, H, W, D, T = warped_all.shape
    losses = []
    for t in range(T):
        warped_t = warped_all[..., t]   # (B,1,H,W,D)
        fixed_t = fixed[..., t]         # (B,1,H,W,D)
        mask_t = mask                   # (B,1,H,W,D)

        warped_n = normalize_volume(warped_t)
        fixed_n = normalize_volume(fixed_t)

        warped_masked = warped_n * mask_t
        fixed_masked = fixed_n * mask_t

        losses.append(F.mse_loss(warped_masked, fixed_masked))

    return torch.stack(losses).mean()

# # -----------------------------
# # Global Normalized Cross-Correlation (GNCC)
# # -----------------------------
# def global_ncc(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     B, C, H, W, D, T = a.shape
#     ncc_vals = []
#
#     for t in range(T):
#         a_t = normalize_volume(a[..., t])
#         b_t = normalize_volume(b[..., t])
#         m_t = mask
#
#         a_t = a_t * m_t
#         b_t = b_t * m_t
#
#         a_mean = a_t.mean(dim=(2, 3, 4), keepdim=True)
#         b_mean = b_t.mean(dim=(2, 3, 4), keepdim=True)
#         a_t = a_t - a_mean
#         b_t = b_t - b_mean
#
#         num = (a_t * b_t).mean(dim=(2, 3, 4), keepdim=True)
#         den = torch.sqrt(
#             (a_t ** 2).mean(dim=(2, 3, 4), keepdim=True) *
#             (b_t ** 2).mean(dim=(2, 3, 4), keepdim=True)
#         ) + eps
#         ncc_vals.append(-(num / den).mean())
#
#     return torch.stack(ncc_vals).mean()

# -----------------------------
# Local Normalized Cross-Correlation (LNCC) over timepoints.
# -----------------------------
def local_ncc(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, win_size: int = 9, eps: float = 1e-6) -> torch.Tensor:
    B, C, H, W, D, T = a.shape
    ncc_vals = []

    # convolution filter for local sums
    sum_filt = torch.ones(1, 1, win_size, win_size, win_size, device=a.device, dtype=a.dtype)
    win_vol = win_size ** 3
    pad = win_size // 2

    for t in range(T):
        a_t = a[..., t] * mask
        b_t = b[..., t] * mask

        # reshape to (B,1,H,W,D) for conv3d
        a_t = a_t.view(B, 1, H, W, D)
        b_t = b_t.view(B, 1, H, W, D)

        # local sums
        I_sum = F.conv3d(a_t, sum_filt, padding=pad)
        J_sum = F.conv3d(b_t, sum_filt, padding=pad)

        I2_sum = F.conv3d(a_t * a_t, sum_filt, padding=pad)
        J2_sum = F.conv3d(b_t * b_t, sum_filt, padding=pad)
        IJ_sum = F.conv3d(a_t * b_t, sum_filt, padding=pad)

        # means
        u_I = I_sum / win_vol
        u_J = J_sum / win_vol

        # cross terms
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_vol
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_vol
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_vol

        ncc = (cross * cross) / (I_var * J_var + eps)
        ncc_vals.append(-ncc.mean())

    return torch.stack(ncc_vals).mean()

# -----------------------------
# Some helper function
# -----------------------------
def next_multiple(x, n=32):
    return ((int(x) + n - 1) // n) * n

def compute_padding(H, W, D, n=32):
    Hn, Wn, Dn = next_multiple(H, n), next_multiple(W, n), next_multiple(D, n)
    ph, pw, pd = Hn - H, Wn - W, Dn - D
    # For (B,C,H,W,D): pad order is (D_left,D_right, W_left,W_right, H_left,H_right)
    pad = (0, pd, 0, pw, 0, ph)
    return pad, (ph, pw, pd)

def pad_moving(moving, pad):
    # moving: (B,1,H,W,D,T)
    B,C,H,W,D,T = moving.shape
    padded_list = []
    for t in range(T):
        mov_t = moving[..., t]                # (B,1,H,W,D)
        mov_t_pad = F.pad(mov_t, pad)         # (B,1,Hp,Wp,Dp)
        padded_list.append(mov_t_pad)
    return torch.stack(padded_list, dim=-1)   # (B,1,Hp,Wp,Dp,T)


def unpad_5d(x, phpwpd):
    ph, pw, pd = phpwpd
    H, W, D = x.shape[2], x.shape[3], x.shape[4]
    return x[:, :, 0:H-ph if ph else H,
                   0:W-pw if pw else W,
                   0:D-pd if pd else D]

# def normalize_volume(x, eps: float = 1e-6):
#     x_cpu = x.detach().cpu().numpy()   # NumPy-based normalization, safe conversion
#     vmin = np.percentile(x_cpu, 1)
#     vmax = np.percentile(x_cpu, 99)
#     x_cpu = np.clip(x_cpu, vmin, vmax)
#     x_cpu = (x_cpu - vmin) / (vmax - vmin + eps)
#     return torch.from_numpy(x_cpu).to(x.device, dtype=x.dtype)

def normalize_volume(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Flatten spatial dims → (B, C, N)
    flat = x.flatten(start_dim=2)

    # Compute per-sample percentiles (1% and 99%)
    vmin = torch.quantile(flat, 0.01, dim=-1, keepdim=True)
    vmax = torch.quantile(flat, 0.99, dim=-1, keepdim=True)

    vmin = vmin.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    vmax = vmax.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # Clip and scale
    x = torch.clamp(x, vmin, vmax)
    return (x - vmin) / (vmax - vmin + eps)

# -----------------------------
# LightningModule
# -----------------------------
class VoxelMorphReg(pl.LightningModule):
    def __init__(self, lr=1e-4, lambda_smooth=1):
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

    def forward(self, moving, fixed, mask):
        """
        moving: (B, 1, H, W, D, T)              # RAW (H,W,D ordering)
        fixed:  (B, 1, H, W, D, T)              # RAW
        mask:  (B, 1, H, W, D)
        returns:
          warped_all_raw: (B, 1, H, W, D, T)    # raw intensities preserved
          flows_4d:       (B, 3, H, W, D, T)
        """
        B, C, H, W, D, T = moving.shape

        # ---- Padding to multiples of 32 ----
        pad, phpwpd = compute_padding(H, W, D, n=32)
        fixed_pad = F.pad(fixed[..., 0], pad).unsqueeze(-1)  # pick one T since fixed is identical across T
        moving_pad = pad_moving(moving, pad)
        mask_pad = F.pad(mask, pad)
        mask_pad_bin = (mask_pad > 0.5).float()

        warped_list, flow_list = [], []

        for t in range(T):
            mov_t_raw = moving_pad[..., t]  # (B,1,Hp,Wp,Dp)
            fix_t_raw = fixed_pad[..., 0]  # (B,1,Hp,Wp,Dp)

            # ---- Normalize per timepoint ----
            mov_t_norm = normalize_volume(mov_t_raw)
            fix_t_norm = normalize_volume(fix_t_raw)

            # Masked inputs
            mov_t_norm = mov_t_norm * mask_pad_bin
            fix_t_norm = fix_t_norm * mask_pad_bin

            # UNet input
            x = torch.cat([mov_t_norm, fix_t_norm], dim=1)  # (B,2,Hp,Wp,Dp)
            flow = self.unet(x)

            # Warp raw moving (not normalized!)
            warped_raw = self.transformer(mov_t_raw, flow)

            # ---- Unpad ----
            warped_raw = unpad_5d(warped_raw, phpwpd)
            flow = unpad_5d(flow, phpwpd)

            warped_list.append(warped_raw)
            flow_list.append(flow)

        warped_all_raw = torch.stack(warped_list, dim=-1)   # (B,1,H,W,D,T)
        flows_4d = torch.stack(flow_list, dim=-1)           # (B,3,H,W,D,T)
        return warped_all_raw, flows_4d

    def training_step(self, batch, batch_idx):
        moving, fixed, mask = batch["moving"], batch["fixed"], batch["mask"]
        warped_all, flows_4d = self(moving, fixed, mask)

        loss_l2 = l2_loss(warped_all, fixed, mask)
        loss_ncc = local_ncc(warped_all, fixed, mask, win_size=9)
        loss_sim = 0.25 * loss_l2 + 0.75 * loss_ncc

        B, _, H, W, D, T = warped_all.shape
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
        moving, fixed, mask = batch["moving"], batch["fixed"], batch["mask"]
        warped_all, flows_4d = self(moving, fixed, mask)

        loss_l2 = l2_loss(warped_all, fixed, mask)
        loss_ncc = local_ncc(warped_all, fixed, mask, win_size=9)
        loss_sim = 0.25 * loss_l2 + 0.75 * loss_ncc

        B, _, H, W, D, T = warped_all.shape
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
num_epochs = 30
batch_size = 1
lr = 1e-4
lambda_smooth = 0.5
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