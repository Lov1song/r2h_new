"""
全图推理 + 可视化（测试集，保存到 checkpoints/v3/）

对 dataset/split.json 中的 test 样本：
  - 从 data/{name}.npz 读取完整配准 RGB + HSI + valid_mask
  - 用最佳 v6 checkpoint 推理全图
  - 裁剪至培养皿有效区域（valid_mask bounding box）
  - 在培养皿内用 Otsu 分割菌落（colony_mask）
  - 生成 5 列对比图：RGB菌落overlay / 预测假彩色 / GT假彩色 / 菌落掩码 / 光谱曲线
  - 保存定量指标

Usage:
    python test_fullimg.py
    python test_fullimg.py --ckpt checkpoints/v6/best.pth
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
from architecture import build_model

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT       = Path(r'E:\2026\05\r2h')
DATA_DIR   = ROOT / 'data'
SPLIT_JSON = ROOT / 'dataset' / 'split.json'
CKPT_PATH  = ROOT / 'checkpoints' / 'v6' / 'best.pth'
OUT_DIR    = ROOT / 'checkpoints' / 'v6'

EPS        = 1e-3
WL_ALL     = np.linspace(402, 1005, 176)
WL_NM      = WL_ALL  # updated in main() after reading band_start/end from ckpt

# ── 评价指标 ──────────────────────────────────────────────────────────────────

def mrae(pred, gt, mask=None):
    v = np.abs(pred - gt) / (gt + EPS)
    return float(v[mask].mean()) if mask is not None else float(v.mean())

def rmse(pred, gt, mask=None):
    v = (pred - gt) ** 2
    return float(np.sqrt(v[mask].mean() if mask is not None else v.mean()))

def psnr(pred, gt, mask=None):
    mse_v = ((pred - gt) ** 2)
    mse_v = float(mse_v[mask].mean()) if mask is not None else float(mse_v.mean())
    return float(10 * np.log10(1.0 / (mse_v + 1e-10)))

def ssim_band(a, b, win=11):
    from scipy.ndimage import uniform_filter
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu1, mu2 = uniform_filter(a, win), uniform_filter(b, win)
    s1  = uniform_filter(a*a, win) - mu1**2
    s2  = uniform_filter(b*b, win) - mu2**2
    s12 = uniform_filter(a*b, win) - mu1*mu2
    C1, C2 = 0.01**2, 0.03**2
    num = (2*mu1*mu2+C1) * (2*s12+C2)
    den = (mu1**2+mu2**2+C1) * (s1+s2+C2)
    return float(np.mean(num / (den + 1e-10)))

def mean_ssim(pred, gt):
    return float(np.mean([ssim_band(pred[:,:,i], gt[:,:,i])
                          for i in range(pred.shape[2])]))

def sam(pred, gt, mask=None):
    """Spectral Angle Mapper, averaged over pixels. Returns degrees."""
    dot      = (pred * gt).sum(axis=-1)                        # (H, W)
    norm_p   = np.linalg.norm(pred, axis=-1)                   # (H, W)
    norm_g   = np.linalg.norm(gt,   axis=-1)
    cos_a    = np.clip(dot / (norm_p * norm_g + 1e-8), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_a))                   # (H, W)
    return float(angle_deg[mask].mean()) if mask is not None else float(angle_deg.mean())

# ── 可视化工具 ────────────────────────────────────────────────────────────────

def band_idx(nm):
    return int(np.argmin(np.abs(WL_NM - nm)))

def false_color(hsi_hwc, mask=None):
    fc = np.stack([
        hsi_hwc[:, :, band_idx(800)],
        hsi_hwc[:, :, band_idx(670)],
        hsi_hwc[:, :, band_idx(550)],
    ], axis=2)
    if mask is not None:
        fc[~mask] = 0.0
    return np.clip(fc, 0.0, 1.0)

def crop_to_mask(arr_hwc, mask):
    rows, cols = np.where(mask)
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    return arr_hwc[r0:r1, c0:c1], mask[r0:r1, c0:c1]

def make_rgb_display(rgb_u8, mask):
    rgb = rgb_u8.astype(np.float32) / 255.0
    rgb[~mask] = 0.0
    return rgb


def make_colony_mask(hsi_hwc, valid_mask):
    """
    在培养皿区域内用 Otsu 阈值分割菌落（高反射率前景）与培养基背景。
    基于各波段均值反射率做直方图 Otsu，不依赖 cv2。
    返回 colony_mask (H, W) bool
    """
    mean_ref = hsi_hwc.mean(axis=2)          # (H, W)
    dish_vals = mean_ref[valid_mask]          # 仅在培养皿内统计

    counts, edges = np.histogram(dish_vals, bins=256, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) * 0.5
    total   = counts.sum()
    s_total = (counts * centers).sum()

    best_thresh, best_var, w0, s0 = 0.0, -1.0, 0, 0.0
    for i in range(256):
        w0 += counts[i]
        s0 += counts[i] * centers[i]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = s0 / w0
        m1 = (s_total - s0) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_thresh = var, centers[i]

    colony_mask = valid_mask & (mean_ref >= best_thresh)
    n_dish   = int(valid_mask.sum())
    n_colony = int(colony_mask.sum())
    print(f'  colony Otsu={best_thresh:.3f}  '
          f'{n_colony:,}/{n_dish:,} ({100*n_colony/max(n_dish,1):.1f}%)')
    return colony_mask


# ── 全图推理 ──────────────────────────────────────────────────────────────────

def infer_full(model, rgb_u8, device):
    rgb_f = rgb_u8.astype(np.float32) / 255.0
    t = torch.from_numpy(rgb_f.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.clamp(model(t), 0.0, 1.0)
    return pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)

# ── 绘图 ──────────────────────────────────────────────────────────────────────

def save_vis(name, rgb_disp, pred_fc, gt_fc, pred_hwc, gt_hwc,
             mask, colony_mask, metrics, out_dir):
    """
    5 列图：
      0 RGB + 菌落绿色overlay
      1 预测 HSI 假彩色（菌落区域）
      2 GT HSI 假彩色（菌落区域）
      3 菌落掩码（绿=菌落，灰=培养基，黑=背景）
      4 光谱曲线（GT全皿虚线 / GT菌落实线 / Pred菌落实线）
    """
    ax_bg = '#1A2B3C'

    fig = plt.figure(figsize=(24, 5))
    fig.patch.set_facecolor('#0D1B2A')
    fig.suptitle(
        f'{name}    MRAE={metrics["mrae"]:.4f}  '
        f'RMSE={metrics["rmse"]:.4f}  '
        f'PSNR={metrics["psnr"]:.2f} dB  '
        f'SSIM={metrics["ssim"]:.4f}  '
        f'SAM={metrics["sam"]:.2f}°',
        color='#E0E0E0', fontsize=11, y=1.01
    )

    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.06)

    # 0: RGB（原图，无 overlay）
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb_disp)
    ax.set_title('Input RGB', color='#4FC3F7', fontsize=9, pad=3)
    ax.axis('off')
    ax.set_facecolor(ax_bg)

    # 1: 预测假彩色
    ax = fig.add_subplot(gs[0, 1])
    pred_fc_col = pred_fc.copy()
    pred_fc_col[~mask] = 0.0
    ax.imshow(pred_fc_col)
    ax.set_title('Predicted HSI (800/670/550 nm)', color='#FF8A65', fontsize=9, pad=3)
    ax.axis('off')
    ax.set_facecolor(ax_bg)

    # 2: GT 假彩色
    ax = fig.add_subplot(gs[0, 2])
    gt_fc_col = gt_fc.copy()
    gt_fc_col[~mask] = 0.0
    ax.imshow(gt_fc_col)
    ax.set_title('Ground Truth HSI (800/670/550 nm)', color='#66BB6A', fontsize=9, pad=3)
    ax.axis('off')
    ax.set_facecolor(ax_bg)

    # 3: 菌落掩码可视化
    ax = fig.add_subplot(gs[0, 3])
    mask_vis = np.zeros((*mask.shape, 3), dtype=np.float32)
    mask_vis[mask]        = [0.25, 0.25, 0.25]   # 培养基 → 灰
    mask_vis[colony_mask] = [0.10, 0.75, 0.20]   # 菌落   → 绿
    ax.imshow(mask_vis)
    n_col_pct = 100 * colony_mask.sum() / max(mask.sum(), 1)
    ax.set_title(f'Colony mask  ({n_col_pct:.1f}%)\n(green=colony  gray=agar)',
                 color='#B0BEC5', fontsize=9, pad=3)
    ax.axis('off')
    ax.set_facecolor(ax_bg)

    # 4: 光谱曲线（GT全皿 / GT菌落 / Pred菌落）
    ax = fig.add_subplot(gs[0, 4])
    ax.set_facecolor(ax_bg)

    gt_dish_spec   = gt_hwc[mask].mean(axis=0)
    gt_col_spec    = gt_hwc[colony_mask].mean(axis=0)   if colony_mask.any() else gt_dish_spec
    pred_col_spec  = pred_hwc[colony_mask].mean(axis=0) if colony_mask.any() else pred_hwc[mask].mean(axis=0)

    ax.plot(WL_NM, gt_dish_spec,  color='#42A5F5', lw=1.0, ls='--', alpha=0.55,
            label='GT (dish)')
    ax.plot(WL_NM, gt_col_spec,   color='#42A5F5', lw=1.8, label='GT (colony)')
    ax.plot(WL_NM, pred_col_spec, color='#FF7043', lw=1.8, label='Pred (colony)')
    ax.fill_between(WL_NM, pred_col_spec, gt_col_spec, alpha=0.12, color='#FFD54F')

    all_vals = np.concatenate([gt_dish_spec, gt_col_spec, pred_col_spec])
    vmin, vmax = all_vals.min(), all_vals.max()
    margin = (vmax - vmin) * 0.1 + 1e-4
    ax.set_ylim(vmin - margin, vmax + margin)
    ax.set_xlabel('Wavelength (nm)', color='#B0BEC5', fontsize=8)
    ax.set_ylabel('Reflectance',     color='#B0BEC5', fontsize=8)
    ax.set_title('Mean Reflectance (colony pixels)', color='#B0BEC5', fontsize=9, pad=3)
    ax.tick_params(colors='#B0BEC5', labelsize=7)
    ax.grid(True, color='#2A3B4C', linestyle='--', alpha=0.6)
    ax.legend(fontsize=7, facecolor='#1A2B3C', edgecolor='#2A3B4C', labelcolor='#E0E0E0')
    for sp in ax.spines.values():
        sp.set_edgecolor('#2A3B4C')

    plt.tight_layout()
    out_path = out_dir / f'{name}.png'
    plt.savefig(str(out_path), dpi=130, bbox_inches='tight', facecolor='#0D1B2A')
    plt.close()
    return out_path

# ── 主函数 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',    default=str(CKPT_PATH))
    p.add_argument('--model',   type=str, default=None,
                   help='Model architecture (overrides checkpoint; default: read from ckpt)')
    p.add_argument('--n_feat',  type=int, default=None,
                   help='Override n_feat (default: read from ckpt)')
    p.add_argument('--stage',   type=int, default=None,
                   help='Override stage (default: read from ckpt)')
    p.add_argument('--split',   default='test',
                   help='Which split to evaluate: test | val | all')
    p.add_argument('--out_dir', type=str, default=None,
                   help='Output directory (default: same as ckpt folder)')
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    out_dir   = Path(args.out_dir) if args.out_dir else ckpt_path.parent

    vis_dir = out_dir / 'vis_fullimg'
    vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # load checkpoint first to read training metadata
    ckpt = torch.load(str(ckpt_path), map_location=device)
    band_start = ckpt.get('band_start', 0)
    band_end   = ckpt.get('band_end',   176)
    model_name = args.model  or ckpt.get('model_name', 'mstpp')
    n_feat     = args.n_feat if args.n_feat is not None else ckpt.get('n_feat', 31)
    stage      = args.stage  or ckpt.get('stage',  3)
    hsi_bands  = band_end - band_start

    global WL_NM
    WL_NM = WL_ALL[band_start:band_end]
    print(f'Model: {model_name}  n_feat={n_feat}  stage={stage}')
    print(f'Band range: band {band_start}–{band_end-1} '
          f'({WL_NM[0]:.1f}–{WL_NM[-1]:.1f} nm), {hsi_bands} bands')

    model = build_model(model_name, in_channels=3, out_channels=hsi_bands,
                        n_feat=n_feat, stage=stage).to(device)
    model.load_state_dict(ckpt['model'])
    ep = ckpt.get('epoch', '?')
    print(f'Loaded checkpoint  epoch={ep}  from {ckpt_path}')
    model.eval()

    split_data = json.loads(SPLIT_JSON.read_text())
    if args.split == 'all':
        names = split_data['train'] + split_data['val'] + split_data['test']
    else:
        names = split_data[args.split]
    print(f'Evaluating {len(names)} samples ({args.split} split)\n')

    all_metrics = []
    all_gt_dish_spectra    = []
    all_gt_colony_spectra  = []
    all_pred_colony_spectra = []
    sample_names = []

    for name in names:
        npz_path = DATA_DIR / f'{name}.npz'
        if not npz_path.exists():
            print(f'  [skip] {name}: npz 不存在')
            continue

        print(f'  {name} ...', end=' ', flush=True)
        d = np.load(str(npz_path))
        rgb_u8    = d['rgb']
        hsi_gt    = d['hsi'][:, :, band_start:band_end]   # trim to trained bands
        mask_full = d['valid_mask']

        pred_full = infer_full(model, rgb_u8, device)

        pred_crop, mask_crop = crop_to_mask(pred_full, mask_full)
        gt_crop,   _         = crop_to_mask(hsi_gt,   mask_full)
        rgb_crop,  _         = crop_to_mask(rgb_u8,   mask_full)

        # 菌落分割（基于 GT HSI 均值，仅在培养皿内）
        colony_crop = make_colony_mask(gt_crop, mask_crop)

        # 用菌落掩码计算指标（若无菌落像素则退回全皿）
        eval_mask = colony_crop if colony_crop.any() else mask_crop
        m = {
            'name': name,
            'mrae': mrae(pred_crop, gt_crop, eval_mask),
            'rmse': rmse(pred_crop, gt_crop, eval_mask),
            'psnr': psnr(pred_crop, gt_crop, eval_mask),
            'ssim': mean_ssim(pred_crop, gt_crop),
            'sam':  sam(pred_crop, gt_crop, eval_mask),
        }
        all_metrics.append(m)

        # 收集光谱
        all_gt_dish_spectra.append(gt_crop[mask_crop].mean(axis=0))
        gt_col_spec   = gt_crop[colony_crop].mean(axis=0)   if colony_crop.any() else gt_crop[mask_crop].mean(axis=0)
        pred_col_spec = pred_crop[colony_crop].mean(axis=0) if colony_crop.any() else pred_crop[mask_crop].mean(axis=0)
        all_gt_colony_spectra.append(gt_col_spec)
        all_pred_colony_spectra.append(pred_col_spec)
        sample_names.append(name)

        rgb_disp = make_rgb_display(rgb_crop, mask_crop)
        pred_fc  = false_color(pred_crop, mask_crop)
        gt_fc    = false_color(gt_crop,   mask_crop)

        out_path = save_vis(name, rgb_disp, pred_fc, gt_fc,
                            pred_crop, gt_crop, mask_crop, colony_crop,
                            m, vis_dir)
        print(f'MRAE={m["mrae"]:.4f}  PSNR={m["psnr"]:.2f}dB  → {out_path.name}')

    # 汇总
    print('\n' + '='*60)
    print(f'{"Name":<12} {"MRAE":>8} {"RMSE":>8} {"PSNR":>9} {"SSIM":>8} {"SAM(°)":>8}')
    print('-'*70)
    for m in all_metrics:
        print(f'{m["name"]:<12} {m["mrae"]:>8.4f} {m["rmse"]:>8.4f} '
              f'{m["psnr"]:>9.2f} {m["ssim"]:>8.4f} {m["sam"]:>8.2f}')
    if all_metrics:
        print('-'*70)
        print(f'{"Mean":<12} '
              f'{np.mean([m["mrae"] for m in all_metrics]):>8.4f} '
              f'{np.mean([m["rmse"] for m in all_metrics]):>8.4f} '
              f'{np.mean([m["psnr"] for m in all_metrics]):>9.2f} '
              f'{np.mean([m["ssim"] for m in all_metrics]):>8.4f} '
              f'{np.mean([m["sam"]  for m in all_metrics]):>8.2f}')
    print('='*60)

    csv_path = out_dir / 'test_fullimg_metrics.csv'
    with open(csv_path, 'w') as f:
        f.write('name,mrae,rmse,psnr,ssim,sam\n')
        for m in all_metrics:
            f.write(f'{m["name"]},{m["mrae"]:.6f},{m["rmse"]:.6f},'
                    f'{m["psnr"]:.4f},{m["ssim"]:.6f},{m["sam"]:.4f}\n')
    print(f'\n指标已保存：{csv_path}')
    print(f'可视化图像：{vis_dir}')

    # ── 聚合光谱曲线（菌落区域 均值 ± std） ─────────────────────────────────
    if all_gt_colony_spectra:
        gt_dish_mat  = np.stack(all_gt_dish_spectra,    axis=0)
        gt_col_mat   = np.stack(all_gt_colony_spectra,  axis=0)
        pred_col_mat = np.stack(all_pred_colony_spectra, axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        fig.patch.set_facecolor('#0D1B2A')

        for ax, (gt_mat, pred_mat, title_suffix) in zip(
            axes,
            [(gt_dish_mat,  pred_col_mat, '(Full Dish GT  vs  Colony Pred)'),
             (gt_col_mat,   pred_col_mat, '(Colony GT  vs  Colony Pred)')]):

            ax.set_facecolor('#1A2B3C')
            for i in range(len(gt_mat)):
                ax.plot(WL_NM, gt_mat[i],   color='#42A5F5', lw=0.7, alpha=0.3)
                ax.plot(WL_NM, pred_mat[i], color='#FF7043', lw=0.7, alpha=0.3)

            gt_mean, gt_std     = gt_mat.mean(0),   gt_mat.std(0)
            pred_mean, pred_std = pred_mat.mean(0), pred_mat.std(0)

            ax.fill_between(WL_NM, gt_mean - gt_std,   gt_mean + gt_std,
                            color='#42A5F5', alpha=0.18)
            ax.fill_between(WL_NM, pred_mean - pred_std, pred_mean + pred_std,
                            color='#FF7043', alpha=0.18)
            ax.plot(WL_NM, gt_mean,   color='#42A5F5', lw=2.2,
                    label=f'GT mean ± std  (n={len(gt_mat)})')
            ax.plot(WL_NM, pred_mean, color='#FF7043', lw=2.2,
                    label=f'Pred mean ± std (n={len(pred_mat)})')

            all_vals = np.concatenate([gt_mat.ravel(), pred_mat.ravel()])
            vmin, vmax = all_vals.min(), all_vals.max()
            margin = (vmax - vmin) * 0.1 + 1e-4
            ax.set_xlim(WL_NM[0], WL_NM[-1])
            ax.set_ylim(vmin - margin, vmax + margin)
            ax.set_xlabel('Wavelength (nm)', color='#B0BEC5', fontsize=11)
            ax.set_ylabel('Reflectance',     color='#B0BEC5', fontsize=11)
            ax.set_title(f'Mean Spectral Reflectance — Test Set\n{title_suffix}',
                         color='#E0E0E0', fontsize=11, pad=8)
            ax.tick_params(colors='#B0BEC5', labelsize=9)
            ax.grid(True, color='#2A3B4C', linestyle='--', alpha=0.5)
            ax.legend(fontsize=10, facecolor='#1A2B3C',
                      edgecolor='#2A3B4C', labelcolor='#E0E0E0')
            for sp in ax.spines.values():
                sp.set_edgecolor('#2A3B4C')

        plt.tight_layout()
        agg_path = out_dir / 'mean_reflectance_spectrum.png'
        plt.savefig(str(agg_path), dpi=150, bbox_inches='tight', facecolor='#0D1B2A')
        plt.close()
        print(f'聚合光谱曲线：{agg_path}')


if __name__ == '__main__':
    main()
