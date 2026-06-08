"""
Training script — valid mask edition (Route A).

与 train.py 的唯一区别：
  训练时从 RGB 检测 warp 黑边（培养皿外区域），loss 只在皿内像素上计算。
  warp 黑边的 RGB 精确为 0（双线性插值边缘），阈值 0.02 可安全区分。

Usage:
    python train_masked.py
    python train_masked.py --epochs 300 --batch_size 2 --lr 4e-4 --n_feat 64
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

BASE_DIR  = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from architecture import build_model, list_models
from dataset import get_dataloaders

# ── Constants ──────────────────────────────────────────────────────────────────
PATCH_DIR = BASE_DIR / 'dataset' / 'patches'
CKPT_DIR  = BASE_DIR / 'checkpoints' / 'v8_masked'

BAND_START_DEFAULT = 10   # ~419 nm
BAND_END_DEFAULT   = 160  # ~950 nm (exclusive), 150 bands
EPS          = 1e-3
VALID_THRESH = 0.02   # RGB 阈值：低于此视为 warp 黑边


# ── Valid mask ──────────────────────────────────────────────────────────────────

def get_valid_mask(rgb: torch.Tensor) -> torch.Tensor:
    """
    从 RGB 检测皿内有效像素。
    warp 黑边所有通道精确为 0，皿内像素至少有一个通道 > VALID_THRESH。
    返回 (B, 1, H, W) float，1 = 皿内，0 = 黑边。
    """
    return (rgb.max(dim=1, keepdim=True).values > VALID_THRESH).float()


# ── Loss ───────────────────────────────────────────────────────────────────────

def mrae_loss(pred: torch.Tensor, gt: torch.Tensor,
              mask: torch.Tensor = None) -> torch.Tensor:
    per_pixel = torch.abs(pred - gt) / (gt + EPS)   # (B,C,H,W)
    if mask is not None:
        return (per_pixel * mask).sum() / (mask.sum() * pred.shape[1] + 1e-8)
    return per_pixel.mean()


def sam_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    dot    = (pred * gt).sum(dim=1)
    norm_p = pred.norm(dim=1).clamp(min=1e-8)
    norm_g = gt.norm(dim=1).clamp(min=1e-8)
    cos    = torch.clamp(dot / (norm_p * norm_g), -1.0, 1.0)
    return torch.acos(cos).mean()


def build_loss_fn(loss_name: str, sam_weight: float):
    def combined(pred, gt, mask=None):
        per_pixel_mse = (pred - gt) ** 2
        mse = (per_pixel_mse * mask).sum() / (mask.sum() * pred.shape[1] + 1e-8) \
              if mask is not None else per_pixel_mse.mean()
        return mrae_loss(pred, gt, mask) + 0.5 * mse

    if loss_name == 'mrae':
        return lambda pred, gt, mask=None: mrae_loss(pred, gt, mask)
    elif loss_name == 'combined':
        return combined
    elif loss_name == 'combined_sam':
        def combined_sam(pred, gt, mask=None):
            return combined(pred, gt, mask) + sam_weight * sam_loss(pred, gt)
        return combined_sam
    else:
        raise ValueError(f'Unknown loss: {loss_name}')


# ── Metrics ────────────────────────────────────────────────────────────────────

def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = 11
    B, C, H, W = pred.shape
    p = pred.reshape(B * C, 1, H, W)
    g = gt.reshape(B * C, 1, H, W)
    w = torch.ones(1, 1, k, k, device=pred.device, dtype=pred.dtype) / (k * k)
    mu_p   = F.conv2d(p, w, padding=k // 2)
    mu_g   = F.conv2d(g, w, padding=k // 2)
    sig_pp = F.conv2d(p * p, w, padding=k // 2) - mu_p ** 2
    sig_gg = F.conv2d(g * g, w, padding=k // 2) - mu_g ** 2
    sig_pg = F.conv2d(p * g, w, padding=k // 2) - mu_p * mu_g
    num = (2 * mu_p * mu_g + C1) * (2 * sig_pg + C2)
    den = (mu_p ** 2 + mu_g ** 2 + C1) * (sig_pp + sig_gg + C2)
    return (num / (den + 1e-8)).mean().item()


def _sam(pred: torch.Tensor, gt: torch.Tensor) -> float:
    dot    = (pred * gt).sum(dim=1)
    norm_p = pred.norm(dim=1).clamp(min=1e-8)
    norm_g = gt.norm(dim=1).clamp(min=1e-8)
    cos    = torch.clamp(dot / (norm_p * norm_g), -1.0, 1.0)
    return torch.acos(cos).mean().item() * 180.0 / np.pi


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, gt: torch.Tensor):
    mrae = torch.mean(torch.abs(pred - gt) / (gt + EPS)).item()
    mse  = torch.mean((pred - gt) ** 2).item()
    rmse = mse ** 0.5
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    ssim = _ssim(pred, gt)
    sam  = _sam(pred, gt)
    return mrae, rmse, psnr, ssim, sam


# ── Training loop ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs, loss_fn):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f'Epoch [{epoch:3d}/{total_epochs}] train',
                leave=False, dynamic_ncols=True)
    for rgb, hsi in pbar:
        rgb, hsi = rgb.to(device), hsi.to(device)
        pred  = model(rgb)
        valid = get_valid_mask(rgb)          # (B,1,H,W): 皿内=1, 黑边=0
        loss  = loss_fn(pred, hsi, valid)
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
        mrae_list.append(m); rmse_list.append(r); psnr_list.append(p)
        ssim_list.append(s); sam_list.append(a)
        pbar.set_postfix(MRAE=f'{m:.4f}')
    return (float(np.mean(mrae_list)), float(np.mean(rmse_list)),
            float(np.mean(psnr_list)), float(np.mean(ssim_list)),
            float(np.mean(sam_list)))


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',      type=str,   default='mstpp',
                   help=f'Model architecture. Available: {list_models()}')
    p.add_argument('--epochs',     type=int,   default=300)
    p.add_argument('--batch_size', type=int,   default=2)
    p.add_argument('--lr',         type=float, default=4e-4)
    p.add_argument('--n_feat',     type=int,   default=31)
    p.add_argument('--stage',      type=int,   default=3)
    p.add_argument('--num_workers',type=int,   default=4)
    p.add_argument('--loss',       type=str,   default='mrae',
                   choices=['mrae', 'combined', 'combined_sam'])
    p.add_argument('--sam_weight', type=float, default=0.1)
    p.add_argument('--resume',     type=str,   default=None)
    p.add_argument('--pretrain',   type=str,   default=None)
    p.add_argument('--ckpt_dir',   type=str,   default=None)
    p.add_argument('--band_start', type=int,   default=BAND_START_DEFAULT)
    p.add_argument('--band_end',   type=int,   default=BAND_END_DEFAULT)
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else CKPT_DIR
    log_path = ckpt_dir / 'train_log.csv'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    band_indices = list(range(args.band_start, args.band_end))
    hsi_bands    = len(band_indices)
    wl_all       = np.linspace(402, 1005, 176)
    loss_fn      = build_loss_fn(args.loss, args.sam_weight)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Loss: {args.loss}  |  valid mask: RGB > {VALID_THRESH} (皿内像素)')
    print(f'Band range: band {args.band_start}–{args.band_end-1} '
          f'({wl_all[args.band_start]:.1f}–{wl_all[args.band_end-1]:.1f} nm), '
          f'{hsi_bands} bands')

    train_loader, val_loader, _ = get_dataloaders(
        PATCH_DIR, batch_size=args.batch_size, num_workers=args.num_workers,
        band_indices=band_indices)
    print(f'Train patches: {len(train_loader.dataset)}  '
          f'Val patches: {len(val_loader.dataset)}')

    model = build_model(args.model, in_channels=3, out_channels=hsi_bands,
                        n_feat=args.n_feat, stage=args.stage).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {args.model}  n_feat={args.n_feat}  stage={args.stage}  '
          f'Params: {total_params/1e6:.2f}M')

    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    start_epoch, best_mrae = 1, float('inf')

    if args.pretrain:
        raw   = torch.load(args.pretrain, map_location=device)
        state = raw['model'] if isinstance(raw, dict) and 'model' in raw else raw
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in model_state and v.shape == model_state[k].shape}
        skipped  = [k for k in state if k not in filtered]
        model_state.update(filtered)
        model.load_state_dict(model_state)
        print(f'Pretrain: {len(filtered)} keys loaded, {len(skipped)} skipped')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_mrae   = ckpt.get('best_mrae', float('inf'))
        print(f'Resumed from epoch {ckpt["epoch"]}')

    if not log_path.exists():
        log_path.write_text('epoch,train_loss,val_mrae,val_rmse,val_psnr,val_ssim,val_sam,lr\n')

    print(f'\nTraining for {args.epochs} epochs ...\n')
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs, loss_fn)
        val_mrae, val_rmse, val_psnr, val_ssim, val_sam = evaluate(
            model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        print(f'[{epoch:3d}/{args.epochs}] '
              f'loss={train_loss:.4f}  '
              f'val_MRAE={val_mrae:.4f}  '
              f'val_RMSE={val_rmse:.4f}  '
              f'val_PSNR={val_psnr:.2f}dB  '
              f'val_SSIM={val_ssim:.4f}  '
              f'val_SAM={val_sam:.2f}°  '
              f'lr={lr_now:.2e}  t={elapsed:.0f}s')

        with open(log_path, 'a') as f:
            f.write(f'{epoch},{train_loss:.6f},{val_mrae:.6f},'
                    f'{val_rmse:.6f},{val_psnr:.4f},{val_ssim:.4f},'
                    f'{val_sam:.4f},{lr_now:.2e}\n')

        ckpt_data = {
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_mrae': best_mrae,
            'band_start': args.band_start,
            'band_end':   args.band_end,
            'model_name': args.model,
            'n_feat':     args.n_feat,
            'stage':      args.stage,
        }
        torch.save(ckpt_data, ckpt_dir / 'latest.pth')

        if epoch % 50 == 0:
            torch.save(ckpt_data, ckpt_dir / f'epoch_{epoch:03d}.pth')
            print(f'  → checkpoint saved: epoch_{epoch:03d}.pth')

        if val_mrae < best_mrae:
            best_mrae = val_mrae
            torch.save(ckpt_data, ckpt_dir / 'best.pth')
            print(f'  → new best MRAE: {best_mrae:.4f}')

    print(f'\nDone. Best val MRAE: {best_mrae:.4f}')
    print(f'Best checkpoint: {ckpt_dir / "best.pth"}')


if __name__ == '__main__':
    main()
