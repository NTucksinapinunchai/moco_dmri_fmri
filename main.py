# -*- coding: utf-8 -*-
"""
Created on Mon Sep 1 2025
@author: Nontharat Tucksinapinunchai
"""
"""
original VoxelmorphUNet no dropout, 4D moving +4D new fixed +mask, mix loss (1,1) across time, add flow_scale, normalized_volume with mask, lambda smooth=0.05
no permutation, original size and intensity, resampling instead padding, save stage and weight
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
import torch
torch.set_float32_matmul_precision("high")

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

base_path = "/home/ge.polymtl.ca/p122983/nontharat/moco_dmri/"
data_path = "/home/ge.polymtl.ca/p122983/nontharat/moco_dmri/dmri_dataset/"
sys.path.insert(0,data_path)
json_path = os.path.join(data_path, 'dataset.json')

# -----------------------------
# DataGenerator
# -----------------------------
class DataGenerator(Dataset):
    def __init__(self, file_list, base_dir):
        # super().__init__(data=file_list)
        self.file_list = file_list
        self.base_dir = base_dir

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        pt_path = os.path.join(self.base_dir, sample["data"])
        data = torch.load(pt_path, weights_only=False)

        torch.cuda.empty_cache()
        return {"moving": data["moving"], "fixed": data["fixed"], "mask": data["mask"], "affine": data["affine"],
            "data_path": pt_path}

# -----------------------------
# DataModule
# -----------------------------
class DataModule(pl.LightningDataModule):
    def __init__(self, json_path, base_dir, batch_size, num_workers):
        super().__init__()
        self.json_path = json_path
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load JSON
        with open(self.json_path, "r") as f:
            dataset_dict = json.load(f)

        self.train_ds = DataGenerator(dataset_dict["training"], self.base_dir)
        self.val_ds = DataGenerator(dataset_dict["validation"], self.base_dir)
        self.test_ds = DataGenerator(dataset_dict["testing"], self.base_dir)

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
        mask = mask                   # (B,1,H,W,D)

        warped_n = normalize_volume(warped_t, mask)
        fixed_n = normalize_volume(fixed_t, mask)

        warped_masked = warped_n * mask
        fixed_masked = fixed_n * mask

        losses.append(F.mse_loss(warped_masked, fixed_masked))

    return torch.stack(losses).mean()

# -----------------------------
# Global Normalized Cross-Correlation (GNCC)
# -----------------------------
def global_ncc(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, C, H, W, D, T = a.shape
    ncc_vals = []

    for t in range(T):
        a_t = normalize_volume(a[..., t], mask)
        b_t = normalize_volume(b[..., t], mask)
        m_t = mask

        a_t = a_t * m_t
        b_t = b_t * m_t

        a_mean = a_t.mean(dim=(2, 3, 4), keepdim=True)
        b_mean = b_t.mean(dim=(2, 3, 4), keepdim=True)
        a_t = a_t - a_mean
        b_t = b_t - b_mean

        num = (a_t * b_t).mean(dim=(2, 3, 4), keepdim=True)
        den = torch.sqrt(
            (a_t ** 2).mean(dim=(2, 3, 4), keepdim=True) *
            (b_t ** 2).mean(dim=(2, 3, 4), keepdim=True)
        ) + eps
        ncc_vals.append(-(num / den).mean())

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


def resample_to_multiple(x, multiple=32, mode="trilinear"):
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
    H, W, D = orig_size
    if mode in ["trilinear", "bilinear", "linear"]:
        return F.interpolate(x, size=(H, W, D), mode=mode, align_corners=False)
    else:
        return F.interpolate(x, size=(H, W, D), mode=mode)

def resample_flow_back(flow, orig_size, resampled_size, mode="trilinear"):
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

# -----------------------------
# LightningModule
# -----------------------------
class VoxelMorphReg(pl.LightningModule):
    def __init__(self, lr=1e-4, lambda_smooth=1, flow_scale=5.0):
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
        moving: (B, 1, H, W, D, T)              # RAW (H,W,D ordering)
        fixed:  (B, 1, H, W, D, T)              # RAW
        mask:  (B, 1, H, W, D)
        returns:
          warped_all_raw: (B, 1, H, W, D, T)    # raw intensities preserved
          flows_4d:       (B, 3, H, W, D, T)
        """
        B, C, H, W, D, T = moving.shape

        warped_list, flow_list = [], []

        for t in range(T):
            mov_t_raw = moving[..., t]  # (B,1,H,W,D)
            fix_t_raw = fixed[..., t]  # reference (same for all t)
            mask_raw = mask  # (B,1,H,W,D)

            # --- Resample all to multiples of 32 ---
            mov_t_res, orig_size = resample_to_multiple(mov_t_raw, multiple=32, mode="trilinear")
            fix_t_res, _ = resample_to_multiple(fix_t_raw, multiple=32, mode="trilinear")
            mask_res, _ = resample_to_multiple(mask_raw, multiple=32, mode="nearest")

            resampled_size = mov_t_res.shape[2:]  # (Hres, Wres, Dres)
            mask_bin = (mask_res > 0.5).float()

            # ---- Normalize per timepoint ----
            mov_t_norm = normalize_volume(mov_t_res, mask=mask_bin) * mask_bin
            fix_t_norm = normalize_volume(fix_t_res, mask=mask_bin) * mask_bin

            # UNet input
            x = torch.cat([mov_t_norm, fix_t_norm], dim=1)  # (B,2,H',W',D')

            flow_res = self.unet(x)
            flow_res = torch.tanh(flow_res) * self.flow_scale

            # Warp raw moving (not normalized!)
            warped_res = self.transformer(mov_t_res, flow_res)

            # --- Resample back to original size ---
            warped_back = resample_back(warped_res, orig_size, mode="trilinear")
            flow_back   = resample_flow_back(flow_res, orig_size, resampled_size, mode="trilinear")

            warped_list.append(warped_back)
            flow_list.append(flow_back)

        warped_all = torch.stack(warped_list, dim=-1)       # (B,1,H,W,D,T)
        flows_4d = torch.stack(flow_list, dim=-1)           # (B,3,H,W,D,T)
        return warped_all, flows_4d

    def training_step(self, batch, batch_idx):
        moving, fixed, mask = batch["moving"], batch["fixed"], batch["mask"]
        warped_all, flows_4d = self(moving, fixed, mask)

        loss_l2 = l2_loss(warped_all, fixed, mask)
        loss_ncc = global_ncc(warped_all, fixed, mask)
        loss_sim = 1 * loss_l2 + 1 * loss_ncc

        B, _, H, W, D, T = warped_all.shape
        flows_bt = flows_4d.permute(0, 5, 1, 2, 3, 4).reshape(B * T, 3, H, W, D)
        loss_smooth = gradient_loss(flows_bt)

        loss = loss_ncc + self.lambda_smooth * loss_smooth

        self.log("train_loss", loss, prog_bar=True, batch_size=B)
        self.log("train_l2_loss", loss_l2, batch_size=B)
        self.log("train_ncc_loss", loss_ncc, batch_size=B)
        self.log("train_sim_loss", loss_sim, batch_size=B)
        self.log("train_smooth_loss", loss_smooth, batch_size=B)

        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        moving, fixed, mask = batch["moving"], batch["fixed"], batch["mask"]
        warped_all, flows_4d = self(moving, fixed, mask)

        loss_l2 = l2_loss(warped_all, fixed, mask)
        loss_ncc = global_ncc(warped_all, fixed, mask)
        loss_sim = 1 * loss_l2 + 1 * loss_ncc

        B, _, H, W, D, T = warped_all.shape
        flows_bt = flows_4d.permute(0, 5, 1, 2, 3, 4).reshape(B * T, 3, H, W, D)
        loss_smooth = gradient_loss(flows_bt)

        loss = loss_ncc + self.lambda_smooth * loss_smooth

        self.log("val_loss", loss, prog_bar=True, batch_size=B)
        self.log("val_l2_loss", loss_l2, batch_size=B)
        self.log("val_ncc_loss", loss_ncc, batch_size=B)
        self.log("val_sim_loss", loss_sim, batch_size=B)
        self.log("val_smooth_loss", loss_smooth, batch_size=B)

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
num_epochs = 150
batch_size = 1
lr = 1e-4
lambda_smooth = 0.05
num_workers = 8

# -----------------------------
# callbacks
# -----------------------------
ckpt_dir = os.path.join(base_path, 'trained_weights')
os.makedirs(ckpt_dir, exist_ok=True)
pretrained_ckpt = os.path.join(ckpt_dir, f"{order_execution}_voxelmorph_best-weighted.ckpt")

checkpoint_cb = ModelCheckpoint(
    dirpath=ckpt_dir,
    filename=f"{order_execution}_voxelmorph_best-weighted",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_weights_only=False
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")
early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=25,
    verbose=True
)

# # Load only model weights (use checkpoint as initialization for a new run)
# if os.path.exists(pretrained_ckpt):
#     print(f"Starting NEW training, initialized from: {pretrained_ckpt}")
#     model = VoxelMorphReg.load_from_checkpoint(
#         pretrained_ckpt,
#         lr=lr,
#         lambda_smooth=lambda_smooth
#     )
# else:
#     print("No pretrained checkpoint found, training from scratch.")
model = VoxelMorphReg(lr=lr, lambda_smooth=lambda_smooth)

# -----------------------------
# trainer
# -----------------------------
if __name__ == "__main__":
    import wandb
    from pytorch_lightning.loggers import WandbLogger

    wandb.login(key="ab48d9c3a5dee9883bcb676015f2487c1bc51f74")
    wandb_logger = WandbLogger(project="moco-dmri", name=f"{order_execution}_VoxelMorphUNet")
    # If continue with the pretrained_ckpt, resume logs to the same wandb run
    # wandb_logger = WandbLogger(project="moco-dmri", name=f"{order_execution}_VoxelMorphUNet", id="kzv9u4mi", resume="must")
    wandb_config = wandb_logger.experiment.config
    wandb_config.update({
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lambda_smooth": lambda_smooth,
        "data_path": data_path
    }, allow_val_change=True)

    dm = DataModule(json_path=json_path, base_dir=base_path, batch_size=batch_size, num_workers=num_workers)
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
    # fit
    # -----------------------------
    # Load full checkpoint (resume training exactly where it left off)
    if os.path.exists(pretrained_ckpt):
        print(f"Resuming training from: {pretrained_ckpt}")
        trainer.fit(model, datamodule=dm, ckpt_path=pretrained_ckpt)
    else:
        print("No checkpoint found, training from scratch.")
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