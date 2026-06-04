"""
Training script for RGB → Hyperspectral reconstruction (MST++).

Usage:
    python train.py
    python train.py --epochs 300 --batch_size 8 --lr 4e-4
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from architecture.MSTpp import MSTpp
from dataset import get_dataloaders

# ── Constants ──────────────────────────────────────────────────────────────────
PATCH_DIR   = Path(r'E:\2026\05\r2h\dataset\patches')
CKPT_DIR    = Path(r'E:\2026\05\r2h\checkpoints\v6')
LOG_PATH    = Path(r'E:\2026\05\r2h\checkpoints\v6\train_log.csv')
# Full range: 176 bands, 402-1005 nm.
# Default: trim UV tail (bands 0-4, 402-416nm, SRF≈0) and
#          NIR tail (bands 160-175, 953-1005nm, sensor falloff).
BAND_START_DEFAULT = 10    #~419 nm
BAND_END_DEFAULT   = 160  # ~950 nm  (exclusive), 155 bands
EPS         = 1e-3   # MRAE epsilon


# ── Loss ───────────────────────────────────────────────────────────────────────

def make_mask(gt: torch.Tensor, thresh: float) -> torch.Tensor:
    """Soft pixel weight based on mean GT reflectance across bands.
    Pixels where mean reflectance < thresh are down-weighted linearly to 0.
    Shape: (B, 1, H, W), broadcast-ready.
    """
    gt_mean = gt.mean(dim=1, keepdim=True)          # (B,1,H,W)
    return torch.clamp(gt_mean / thresh, 0.0, 1.0)  # soft ramp [0,1]


def mrae_loss(pred: torch.Tensor, gt: torch.Tensor,
              mask: torch.Tensor = None) -> torch.Tensor:
    per_pixel = torch.abs(pred - gt) / (gt + EPS)   # (B,C,H,W)
    if mask is not None:
        return (per_pixel * mask).sum() / (mask.sum() * pred.shape[1] + 1e-8)
    return per_pixel.mean()


def combined_loss(pred: torch.Tensor, gt: torch.Tensor,
                  mask: torch.Tensor = None) -> torch.Tensor:
    per_pixel_mse = (pred - gt) ** 2
    if mask is not None:
        mse = (per_pixel_mse * mask).sum() / (mask.sum() * pred.shape[1] + 1e-8)
    else:
        mse = per_pixel_mse.mean()
    return mrae_loss(pred, gt, mask) + 0.5 * mse


# ── Metrics ────────────────────────────────────────────────────────────────────

def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean SSIM across all bands. pred/gt: (B, C, H, W)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = 11
    B, C, H, W = pred.shape
    p = pred.reshape(B * C, 1, H, W)
    g = gt.reshape(B * C, 1, H, W)
    w = torch.ones(1, 1, k, k, device=pred.device, dtype=pred.dtype) / (k * k)
    mu_p  = F.conv2d(p, w, padding=k // 2)
    mu_g  = F.conv2d(g, w, padding=k // 2)
    sig_pp = F.conv2d(p * p, w, padding=k // 2) - mu_p ** 2
    sig_gg = F.conv2d(g * g, w, padding=k // 2) - mu_g ** 2
    sig_pg = F.conv2d(p * g, w, padding=k // 2) - mu_p * mu_g
    num = (2 * mu_p * mu_g + C1) * (2 * sig_pg + C2)
    den = (mu_p ** 2 + mu_g ** 2 + C1) * (sig_pp + sig_gg + C2)
    return (num / (den + 1e-8)).mean().item()


def _sam(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean SAM in degrees. pred/gt: (B, C, H, W)."""
    dot    = (pred * gt).sum(dim=1)
    norm_p = pred.norm(dim=1).clamp(min=1e-8)
    norm_g = gt.norm(dim=1).clamp(min=1e-8)
    cos    = torch.clamp(dot / (norm_p * norm_g), -1.0, 1.0)
    return torch.acos(cos).mean().item() * 180.0 / np.pi


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, gt: torch.Tensor):
    """Returns (mrae, rmse, psnr, ssim, sam) as Python floats."""
    mrae = torch.mean(torch.abs(pred - gt) / (gt + EPS)).item()
    mse  = torch.mean((pred - gt) ** 2).item()
    rmse = mse ** 0.5
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    ssim = _ssim(pred, gt)
    sam  = _sam(pred, gt)
    return mrae, rmse, psnr, ssim, sam


# ── Training loop ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs, mask_thresh=0.05):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f'Epoch [{epoch:3d}/{total_epochs}] train',
                leave=False, dynamic_ncols=True)
    for rgb, hsi in pbar:
        rgb, hsi = rgb.to(device), hsi.to(device)
        pred = model(rgb)
        mask = make_mask(hsi, thresh=mask_thresh)
        loss = combined_loss(pred, hsi, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, desc='val'):
    model.eval()
    mrae_list, rmse_list, psnr_list, ssim_list, sam_list = [], [], [], [], []
    pbar = tqdm(loader, desc=f'  {desc:>5}', leave=False, dynamic_ncols=True)
    for rgb, hsi in pbar:
        rgb, hsi = rgb.to(device), hsi.to(device)
        pred = torch.clamp(model(rgb), 0.0, 1.0)
        m, r, p, s, a = compute_metrics(pred, hsi)
        mrae_list.append(m)
        rmse_list.append(r)
        psnr_list.append(p)
        ssim_list.append(s)
        sam_list.append(a)
        pbar.set_postfix(MRAE=f'{m:.4f}')
    return (float(np.mean(mrae_list)),
            float(np.mean(rmse_list)),
            float(np.mean(psnr_list)),
            float(np.mean(ssim_list)),
            float(np.mean(sam_list)))


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',     type=int,   default=300)
    p.add_argument('--batch_size', type=int,   default=2)
    p.add_argument('--lr',         type=float, default=4e-4)
    p.add_argument('--n_feat',     type=int,   default=31)
    p.add_argument('--stage',      type=int,   default=3)
    p.add_argument('--num_workers',type=int,   default=4)
    p.add_argument('--resume',     type=str,   default=None,
                   help='path to checkpoint to resume from')
    p.add_argument('--pretrain',   type=str,   default=None,
                   help='path to pretrained weights for backbone init (strict=False)')
    p.add_argument('--mask_thresh', type=float, default=0.05,
                   help='GT mean reflectance below this is down-weighted in loss')
    p.add_argument('--band_start', type=int, default=BAND_START_DEFAULT,
                   help='first band index to use (inclusive). default=5 (~419nm)')
    p.add_argument('--band_end',   type=int, default=BAND_END_DEFAULT,
                   help='last band index to use (exclusive). default=160 (~950nm)')
    return p.parse_args()


def main():
    args = parse_args()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    band_indices = list(range(args.band_start, args.band_end))
    hsi_bands    = len(band_indices)
    wl_all       = np.linspace(402, 1005, 176)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Band range: band {args.band_start}–{args.band_end-1} '
          f'({wl_all[args.band_start]:.1f}–{wl_all[args.band_end-1]:.1f} nm), '
          f'{hsi_bands} bands')

    train_loader, val_loader, _ = get_dataloaders(
        PATCH_DIR, batch_size=args.batch_size, num_workers=args.num_workers,
        band_indices=band_indices)
    print(f'Train patches: {len(train_loader.dataset)}  '
          f'Val patches: {len(val_loader.dataset)}')

    model = MSTpp(in_channels=3, out_channels=hsi_bands,
                  n_feat=args.n_feat, stage=args.stage).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model params: {total_params/1e6:.2f}M')

    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_mrae   = float('inf')

    if args.pretrain:
        raw = torch.load(args.pretrain, map_location=device)
        state = raw['model'] if isinstance(raw, dict) and 'model' in raw else raw
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in model_state and v.shape == model_state[k].shape}
        skipped = [k for k in state if k not in filtered]
        model_state.update(filtered)
        model.load_state_dict(model_state)
        print(f'Pretrain loaded: {len(filtered)} keys transferred, '
              f'{len(skipped)} skipped (shape mismatch): {skipped}')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_mrae   = ckpt.get('best_mrae', float('inf'))
        print(f'Resumed from epoch {ckpt["epoch"]}')

    # write CSV header
    if not LOG_PATH.exists():
        LOG_PATH.write_text('epoch,train_loss,val_mrae,val_rmse,val_psnr,val_ssim,val_sam,lr\n')

    print(f'\nTraining for {args.epochs} epochs ...\n')
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs, args.mask_thresh)
        val_mrae, val_rmse, val_psnr, val_ssim, val_sam = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        print(f'[{epoch:3d}/{args.epochs}] '
              f'loss={train_loss:.4f}  '
              f'val_MRAE={val_mrae:.4f}  '
              f'val_RMSE={val_rmse:.4f}  '
              f'val_PSNR={val_psnr:.2f}dB  '
              f'val_SSIM={val_ssim:.4f}  '
              f'val_SAM={val_sam:.2f}°  '
              f'lr={lr_now:.2e}  '
              f't={elapsed:.0f}s')

        with open(LOG_PATH, 'a') as f:
            f.write(f'{epoch},{train_loss:.6f},{val_mrae:.6f},'
                    f'{val_rmse:.6f},{val_psnr:.4f},{val_ssim:.4f},{val_sam:.4f},{lr_now:.2e}\n')

        ckpt = {
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_mrae': best_mrae,
            'band_start': args.band_start,
            'band_end':   args.band_end,
        }
        # save latest
        torch.save(ckpt, CKPT_DIR / 'latest.pth')

        # save every 50 epochs
        if epoch % 50 == 0:
            torch.save(ckpt, CKPT_DIR / f'epoch_{epoch:03d}.pth')
            print(f'  → checkpoint saved: epoch_{epoch:03d}.pth')

        # save best
        if val_mrae < best_mrae:
            best_mrae = val_mrae
            torch.save(ckpt, CKPT_DIR / 'best.pth')
            print(f'  → new best MRAE: {best_mrae:.4f}')

    print(f'\nTraining complete. Best val MRAE: {best_mrae:.4f}')
    print(f'Best checkpoint: {CKPT_DIR / "best.pth"}')


if __name__ == '__main__':
    main()
