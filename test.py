"""
Test script: evaluate MST++ on the held-out test set.

Usage:
    python test.py
    python test.py --ckpt checkpoints/best.pth --save_vis
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from architecture.MSTpp import MSTpp
from dataset import BlueberryHSIDataset

PATCH_DIR  = Path(r'E:\2026\05\r2h\dataset\patches')
CKPT_PATH  = Path(r'E:\2026\05\r2h\checkpoints\v2\best.pth')
VIS_DIR    = Path(r'E:\2026\05\r2h\test_vis')
RESULTS_PATH = Path(r'E:\2026\05\r2h\test_results.npz')
HSI_BANDS  = 176
EPS        = 1e-3


# ── Metrics ────────────────────────────────────────────────────────────────────

def mrae(pred, gt):
    return float(np.mean(np.abs(pred - gt) / (gt + EPS)))

def rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))

def psnr(pred, gt):
    mse_val = np.mean((pred - gt) ** 2)
    return float(10 * np.log10(1.0 / (mse_val + 1e-10)))

def ssim_band(pred_b, gt_b, win=11):
    """Single-band SSIM (luminance + contrast + structure)."""
    from scipy.ndimage import uniform_filter
    mu1  = uniform_filter(pred_b.astype(np.float64), win)
    mu2  = uniform_filter(gt_b.astype(np.float64),   win)
    mu1_sq = mu1 ** 2; mu2_sq = mu2 ** 2; mu12 = mu1 * mu2
    s1   = uniform_filter(pred_b ** 2, win) - mu1_sq
    s2   = uniform_filter(gt_b   ** 2, win) - mu2_sq
    s12  = uniform_filter(pred_b * gt_b, win) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num  = (2 * mu12 + C1) * (2 * s12 + C2)
    den  = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return float(np.mean(num / (den + 1e-10)))

def mean_ssim(pred, gt):
    """Average SSIM across all bands. pred/gt: (H,W,C)."""
    return float(np.mean([ssim_band(pred[:,:,i], gt[:,:,i])
                          for i in range(pred.shape[2])]))


# ── Visualization ──────────────────────────────────────────────────────────────

def false_color(hsi: np.ndarray, bands=None):
    """Quick false-color image from 3 HSI bands. hsi: (H,W,C)."""
    c = hsi.shape[2]
    if bands is None:
        bands = (int(c * 0.75), int(c * 0.50), int(c * 0.10))
    bands = tuple(min(b, c - 1) for b in bands)
    img = hsi[:, :, list(bands)]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def save_comparison(rgb, pred, gt, idx, out_dir: Path):
    wl = np.linspace(402, 1005, pred.shape[0])  # pred: (C,H,W)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb.transpose(1, 2, 0))
    axes[0].set_title('Input RGB')
    axes[0].axis('off')

    axes[1].imshow(false_color(pred.transpose(1, 2, 0)))
    axes[1].set_title('Predicted HSI (false-color)')
    axes[1].axis('off')

    axes[2].imshow(false_color(gt.transpose(1, 2, 0)))
    axes[2].set_title('Ground Truth HSI (false-color)')
    axes[2].axis('off')

    # mean reflectance spectrum
    pred_mean = pred.reshape(pred.shape[0], -1).mean(axis=1)
    gt_mean   = gt.reshape(gt.shape[0],   -1).mean(axis=1)
    axes[3].plot(wl, pred_mean, color='tomato',    label='Pred')
    axes[3].plot(wl, gt_mean,   color='steelblue', label='GT')
    axes[3].set_xlabel('Wavelength (nm)')
    axes[3].set_ylabel('Reflectance')
    axes[3].set_title('Mean Spectral Reflectance')
    axes[3].legend()
    axes[3].set_ylim(0, 1)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_dir / f'sample_{idx:04d}.png'), dpi=120)
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',     type=str, default=str(CKPT_PATH))
    p.add_argument('--out_dir',  type=str, default=None,
                   help='Directory to save results and visualizations (default: project root)')
    p.add_argument('--n_feat',   type=int, default=31)
    p.add_argument('--stage',    type=int, default=3)
    p.add_argument('--save_vis', action='store_true',
                   help='Save false-color comparison images')
    p.add_argument('--vis_n',    type=int, default=10,
                   help='Number of samples to visualize')
    return p.parse_args()


def main():
    args  = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    out_dir  = Path(args.out_dir) if args.out_dir else None
    vis_dir  = (out_dir / 'vis') if out_dir else VIS_DIR
    res_path = (out_dir / 'test_results.npz') if out_dir else RESULTS_PATH
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = MSTpp(in_channels=3, out_channels=HSI_BANDS,
                  n_feat=args.n_feat, stage=args.stage).to(device)
    ckpt  = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
        print(f'Loaded checkpoint from epoch {ckpt.get("epoch", "?")}')
    else:
        model.load_state_dict(ckpt)
        print('Loaded checkpoint (raw state_dict)')
    model.eval()

    test_ds = BlueberryHSIDataset(PATCH_DIR / 'test', augment=False)
    print(f'Test patches: {len(test_ds)}')

    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with torch.no_grad():
        for i, (rgb, hsi) in enumerate(test_ds):
            rgb_t = rgb.unsqueeze(0).to(device)
            pred  = torch.clamp(model(rgb_t), 0.0, 1.0).squeeze(0)

            pred_np = pred.cpu().numpy().transpose(1, 2, 0)   # (H,W,C)
            gt_np   = hsi.numpy().transpose(1, 2, 0)

            m = mrae(pred_np, gt_np)
            r = rmse(pred_np, gt_np)
            p = psnr(pred_np, gt_np)
            s = mean_ssim(pred_np, gt_np)
            results.append((m, r, p, s))

            if args.save_vis and i < args.vis_n:
                save_comparison(rgb.numpy(), pred.cpu().numpy(),
                                hsi.numpy(), i, vis_dir)

    mrae_arr = np.array([x[0] for x in results])
    rmse_arr = np.array([x[1] for x in results])
    psnr_arr = np.array([x[2] for x in results])
    ssim_arr = np.array([x[3] for x in results])

    print('\n' + '='*55)
    print(f'{"Metric":<12} {"Mean":>10} {"Std":>10} {"Best":>10}')
    print('-'*55)
    print(f'{"MRAE":<12} {mrae_arr.mean():>10.4f} {mrae_arr.std():>10.4f} {mrae_arr.min():>10.4f}')
    print(f'{"RMSE":<12} {rmse_arr.mean():>10.4f} {rmse_arr.std():>10.4f} {rmse_arr.min():>10.4f}')
    print(f'{"PSNR (dB)":<12} {psnr_arr.mean():>10.2f} {psnr_arr.std():>10.2f} {psnr_arr.max():>10.2f}')
    print(f'{"SSIM":<12} {ssim_arr.mean():>10.4f} {ssim_arr.std():>10.4f} {ssim_arr.max():>10.4f}')
    print('='*55)

    # save results
    np.savez(str(res_path), mrae=mrae_arr, rmse=rmse_arr,
             psnr=psnr_arr, ssim=ssim_arr)
    print(f'\nDetailed results saved to {res_path}')
    if args.save_vis:
        print(f'Visualizations saved to {vis_dir}')


if __name__ == '__main__':
    main()
