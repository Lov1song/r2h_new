"""
Process 2 raw samples with preprocess.py functions, then run inference.

Selected samples:
  1-花香  / kejian / 1-1
  3-丹东  / kejian / 1-1

All intermediate and final results go to output/.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from preprocess import load_scan, calibrate, build_srf_matrix, synthesize_rgb

RAW_ROOT   = Path(r'F:\蓝莓')
OUTPUT_DIR = Path('output')
INFER      = Path('infer.py')

TARGETS = [
    {
        'name':    '1-花香_1-1',
        'sample':  RAW_ROOT / '1-花香'  / 'kejian' / '1-1',
        'dark':    RAW_ROOT / '1-花香'  / 'kejian' / 'dark',
        'white':   RAW_ROOT / '1-花香'  / 'kejian' / 'white',
    },
    {
        'name':    '3-丹东_1-1',
        'sample':  RAW_ROOT / '3-丹东'  / 'kejian' / '1-1',
        'dark':    RAW_ROOT / '3-丹东'  / 'kejian' / 'dark',
        'white':   RAW_ROOT / '3-丹东'  / 'kejian' / 'white',
    },
]


def save_display_png(rgb: np.ndarray, path: Path):
    """Per-channel 2–75 percentile stretch → 8-bit PNG for human viewing.

    Uses 75th percentile as upper bound so the white reference panel at the
    bottom of the scan frame does not dominate the stretch and crush the
    dark blueberry region.
    """
    out = np.zeros_like(rgb)
    for c in range(rgb.shape[2]):
        ch = rgb[:, :, c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 75)
        out[:, :, c] = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
    Image.fromarray((out * 255).astype(np.uint8)).save(str(path))


def save_infer_png(rgb: np.ndarray, path: Path):
    """Raw reflectance × 255 → 8-bit PNG (preserves training-time value range)."""
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(str(path))


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    srf_matrix = None

    for t in TARGETS:
        name   = t['name']
        print(f'\n=== {name} ===')

        # ── 1. preprocess ──────────────────────────────────────────────────
        print('  loading dark / white references ...')
        dark_data,  _  = load_scan(t['dark'])
        white_data, _  = load_scan(t['white'])

        print('  loading sample raw ...')
        raw, wl = load_scan(t['sample'])

        if srf_matrix is None:
            srf_matrix = build_srf_matrix(wl)
            vis_bands = wl[srf_matrix.sum(0) > 1e-4]
            print(f'  SRF built: {vis_bands[0]:.1f}–{vis_bands[-1]:.1f} nm')

        print('  calibrating ...', end=' ', flush=True)
        hsi = calibrate(raw, dark_data, white_data)
        rgb = synthesize_rgb(hsi, srf_matrix)
        print(f'HSI {hsi.shape} [{hsi.min():.3f},{hsi.max():.3f}]  '
              f'RGB {rgb.shape} [{rgb.min():.4f},{rgb.max():.4f}]')

        # ── 2. save npy ────────────────────────────────────────────────────
        hsi_npy = OUTPUT_DIR / f'{name}_hsi.npy'
        rgb_npy = OUTPUT_DIR / f'{name}_rgb.npy'
        np.save(str(hsi_npy), hsi)
        np.save(str(rgb_npy), rgb)
        print(f'  saved {hsi_npy.name}  {rgb_npy.name}')

        # ── 3. export PNGs ─────────────────────────────────────────────────
        infer_png   = OUTPUT_DIR / f'{name}_rgb.png'
        display_png = OUTPUT_DIR / f'{name}_rgb_display.png'
        save_infer_png(rgb, infer_png)
        save_display_png(rgb, display_png)
        print(f'  saved {infer_png.name}  {display_png.name}')

        # ── 4. run inference ───────────────────────────────────────────────
        out_hsi = OUTPUT_DIR / f'{name}_pred_hsi.npy'
        cmd = [sys.executable, str(INFER),
               '--input',  str(infer_png),
               '--output', str(out_hsi)]
        print(f'  infer: {" ".join(cmd)}')
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            print(f'  ERROR: infer.py returned {result.returncode}')
            continue

        # move vis png into output/
        vis_src = infer_png.with_name(infer_png.stem + '_vis.png')
        if vis_src.exists() and vis_src.parent != OUTPUT_DIR:
            vis_dst = OUTPUT_DIR / vis_src.name
            vis_src.rename(vis_dst)

    print('\n=== Done. Files in output/ ===')
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f'  {f.name:<45} {size_mb:6.1f} MB')


if __name__ == '__main__':
    main()
