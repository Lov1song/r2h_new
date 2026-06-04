"""
Inference script: reconstruct HSI from a single RGB image.

Usage:
    python infer.py --input path/to/rgb.png --output path/to/output.npy
    python infer.py --input photo.jpg              # saves photo_hsi.npy + photo_vis.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from architecture.MSTpp import MSTpp

CKPT_PATH = Path(r'E:\2026\05\r2h\checkpoints\best.pth')
HSI_BANDS  = 176


def load_rgb(path) -> np.ndarray:
    """Load RGB image and return float32 array in [0,1], shape (H,W,3)."""
    img = Image.open(str(path)).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def save_false_color(hsi: np.ndarray, path, bands=(50, 30, 10)):
    """Save a 3-panel false-color montage of the reconstructed HSI."""
    img = hsi[:, :, list(bands)]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    axes.imshow(img)
    axes.set_title(f'False-color HSI (bands {bands[0]},{bands[1]},{bands[2]})')
    axes.axis('off')
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()


def infer(model: torch.nn.Module, rgb: np.ndarray, device: torch.device) -> np.ndarray:
    """Run model on a full-size RGB image; returns HSI (H,W,176)."""
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)  # (1,3,H,W)
    with torch.no_grad():
        pred = torch.clamp(model(t), 0.0, 1.0)
    return pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H,W,C)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  type=str, required=True,
                   help='Path to input RGB image (PNG/JPG/BMP/TIFF)')
    p.add_argument('--output', type=str, default=None,
                   help='Output .npy path (default: <input>_hsi.npy)')
    p.add_argument('--ckpt',   type=str, default=str(CKPT_PATH))
    p.add_argument('--n_feat', type=int, default=31)
    p.add_argument('--stage',  type=int, default=3)
    p.add_argument('--no_vis', action='store_true',
                   help='Skip saving false-color visualization')
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'Input image not found: {input_path}')

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    out_hsi = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + '_hsi.npy')
    out_vis = input_path.with_name(input_path.stem + '_vis.png')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = MSTpp(in_channels=3, out_channels=HSI_BANDS,
                  n_feat=args.n_feat, stage=args.stage).to(device)
    ckpt  = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f'Input: {input_path}')
    rgb = load_rgb(input_path)
    print(f'  shape: {rgb.shape}  range: [{rgb.min():.3f}, {rgb.max():.3f}]')

    hsi = infer(model, rgb, device)
    print(f'Output HSI: {hsi.shape}  range: [{hsi.min():.3f}, {hsi.max():.3f}]')

    np.save(str(out_hsi), hsi)
    print(f'Saved HSI to: {out_hsi}')

    if not args.no_vis:
        save_false_color(hsi, out_vis)
        print(f'Saved visualization to: {out_vis}')


if __name__ == '__main__':
    main()
