"""
Test NTIRE pretrained MST++ (31 bands) zero-shot on blueberry test set.
Reads 256×256 patches from dataset/patches/test/, evaluates against
31-band GT (BAND_INDICES). Results saved to checkpoints/v4/.

Usage:
    python test_v4.py
    python test_v4.py --save_vis
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

CKPT_PATH  = Path(r'E:\2026\05\r2h\save_model\mstpp\best_model_v16_n.pth')
PATCH_DIR  = Path(r'E:\2026\05\r2h\dataset\patches\test')
OUT_DIR    = Path(r'E:\2026\05\r2h\checkpoints\v4')
EPS        = 1e-3

# 31 bands aligned to NTIRE 400–700 nm (10 nm step)
BAND_INDICES = [0, 2, 5, 9, 12, 15, 18, 21, 24, 27, 30,
                34, 37, 40, 43, 46, 49, 52, 55, 58, 61,
                64, 67, 70, 73, 76, 79, 81, 84, 87, 90]
NTIRE_WL = np.arange(400, 701, 10)


# ── Metrics ────────────────────────────────────────────────────────────────────

def mrae(pred, gt):
    return float(np.mean(np.abs(pred - gt) / (gt + EPS)))

def rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))

def psnr(pred, gt):
    return float(10 * np.log10(1.0 / (np.mean((pred - gt) ** 2) + 1e-10)))

def ssim_band(a, b, win=11):
    from scipy.ndimage import uniform_filter
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu1, mu2 = uniform_filter(a, win), uniform_filter(b, win)
    s1  = uniform_filter(a**2, win) - mu1**2
    s2  = uniform_filter(b**2, win) - mu2**2
    s12 = uniform_filter(a*b,  win) - mu1*mu2
    C1, C2 = 0.01**2, 0.03**2
    return float(np.mean(((2*mu1*mu2+C1)*(2*s12+C2)) /
                         ((mu1**2+mu2**2+C1)*(s1+s2+C2) + 1e-10)))

def mean_ssim(pred, gt):   # (H,W,C)
    return float(np.mean([ssim_band(pred[:,:,i], gt[:,:,i])
                          for i in range(pred.shape[2])]))


# ── Visualization ──────────────────────────────────────────────────────────────

def false_color(hsi, idx=(20, 10, 2)):   # ~620, 500, 420 nm
    img = hsi[:, :, list(idx)].copy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def save_vis(fname, rgb, pred_hwc, gt_hwc, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(fname, fontsize=10)

    axes[0].imshow(rgb.transpose(1, 2, 0))
    axes[0].set_title('Input RGB'); axes[0].axis('off')

    axes[1].imshow(false_color(pred_hwc))
    axes[1].set_title('Pred (false-color)'); axes[1].axis('off')

    axes[2].imshow(false_color(gt_hwc))
    axes[2].set_title('GT (false-color)'); axes[2].axis('off')

    axes[3].plot(NTIRE_WL, pred_hwc.reshape(-1,31).mean(0), color='tomato',    label='Pred')
    axes[3].plot(NTIRE_WL, gt_hwc.reshape(-1,31).mean(0),   color='steelblue', label='GT')
    axes[3].set_xlabel('Wavelength (nm)'); axes[3].set_ylabel('Reflectance')
    axes[3].set_title('Mean Spectrum')
    axes[3].legend(); axes[3].grid(True, alpha=0.3); axes[3].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(str(out_dir / f'{fname}.png'), dpi=120)
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--save_vis', action='store_true')
    p.add_argument('--vis_n', type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vis_dir = OUT_DIR / 'vis'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = MSTpp(in_channels=3, out_channels=31, n_feat=31, stage=3).to(device)
    state = torch.load(str(CKPT_PATH), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f'Loaded: {CKPT_PATH.name}')

    test_ds = BlueberryHSIDataset(PATCH_DIR, augment=False, band_indices=BAND_INDICES)
    print(f'Test patches: {len(test_ds)}\n')

    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    mrae_list, rmse_list, psnr_list, ssim_list = [], [], [], []

    with torch.no_grad():
        for i, (rgb, hsi) in enumerate(test_ds):
            rgb_t = rgb.unsqueeze(0).to(device)
            pred  = torch.clamp(model(rgb_t), 0.0, 1.0).squeeze(0)

            pred_np = pred.cpu().numpy().transpose(1, 2, 0)   # (H,W,31)
            gt_np   = hsi.numpy().transpose(1, 2, 0)          # (H,W,31)

            m = mrae(pred_np, gt_np)
            r = rmse(pred_np, gt_np)
            p = psnr(pred_np, gt_np)
            s = mean_ssim(pred_np, gt_np)
            mrae_list.append(m); rmse_list.append(r)
            psnr_list.append(p); ssim_list.append(s)

            if args.save_vis and i < args.vis_n:
                fname = test_ds.files[i].stem
                save_vis(fname, rgb.numpy(), pred_np, gt_np, vis_dir)

    mrae_a = np.array(mrae_list)
    rmse_a = np.array(rmse_list)
    psnr_a = np.array(psnr_list)
    ssim_a = np.array(ssim_list)

    print('='*55)
    print(f'{"Metric":<12} {"Mean":>10} {"Std":>10}')
    print('-'*55)
    print(f'{"MRAE":<12} {mrae_a.mean():>10.4f} {mrae_a.std():>10.4f}')
    print(f'{"RMSE":<12} {rmse_a.mean():>10.4f} {rmse_a.std():>10.4f}')
    print(f'{"PSNR (dB)":<12} {psnr_a.mean():>10.2f} {psnr_a.std():>10.2f}')
    print(f'{"SSIM":<12} {ssim_a.mean():>10.4f} {ssim_a.std():>10.4f}')
    print('='*55)

    np.savez(str(OUT_DIR / 'test_results.npz'),
             mrae=mrae_a, rmse=rmse_a, psnr=psnr_a, ssim=ssim_a)
    print(f'\nResults saved to {OUT_DIR / "test_results.npz"}')
    if args.save_vis:
        print(f'Visualizations saved to {vis_dir}')


if __name__ == '__main__':
    main()
