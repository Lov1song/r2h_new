"""
Export RGB patches from test .npz files as PNG, then run inference.

Selects 2 representative patches from the test set, exports them as PNG
images into output/, then runs infer.py to produce HSI .npy + false-color
visualisation for each.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT_DIR = Path('output')
PATCH_DIR  = Path('dataset/patches/test')
INFER_SCRIPT = Path('infer.py')

SAMPLES = [
    '1-花香_1-2_p010.npz',
    '云南花香蓝莓_Day1_1-2_p010.npz',
]


def export_rgb_png(npz_path: Path, out_dir: Path) -> Path:
    d   = np.load(str(npz_path))
    rgb = d['rgb']                          # (128,128,3) float32 [0,1]
    stem = npz_path.stem

    # Inference PNG: raw reflectance × 255 — preserves the training-time value range
    infer_path = out_dir / f'{stem}_rgb.png'
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(str(infer_path))

    # Display PNG: 2–98 percentile stretch — makes dark blueberry scenes visible
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb_disp = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    disp_path = out_dir / f'{stem}_rgb_display.png'
    Image.fromarray((rgb_disp * 255).astype(np.uint8)).save(str(disp_path))

    print(f'[export] {npz_path.name}  raw range [{rgb.min():.4f}, {rgb.max():.4f}]')
    print(f'         infer  → {infer_path}')
    print(f'         display→ {disp_path}')
    return infer_path


def run_infer(png_path: Path, out_dir: Path):
    out_hsi = out_dir / f'{png_path.stem}_hsi.npy'
    cmd = [
        sys.executable, str(INFER_SCRIPT),
        '--input',  str(png_path),
        '--output', str(out_hsi),
    ]
    print(f'[infer] {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f'  ERROR: infer.py returned {result.returncode}')
    else:
        # infer.py saves vis next to input; move it into output/ as well
        vis_src = png_path.with_name(png_path.stem + '_vis.png')
        if vis_src.exists() and vis_src.parent != out_dir:
            vis_dst = out_dir / vis_src.name
            vis_src.rename(vis_dst)
            print(f'  moved vis → {vis_dst}')


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for sample_name in SAMPLES:
        npz_path = PATCH_DIR / sample_name
        if not npz_path.exists():
            print(f'[WARN] not found: {npz_path}')
            continue
        png_path = export_rgb_png(npz_path, OUTPUT_DIR)
        run_infer(png_path, OUTPUT_DIR)

    print('\nDone. Files in output/:')
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f'  {f.name}')


if __name__ == '__main__':
    main()
