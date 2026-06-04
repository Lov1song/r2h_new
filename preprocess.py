"""
Blueberry kejian hyperspectral preprocessing pipeline:
  1. Load ENVI BIL .raw files
  2. Radiometric calibration: reflectance = (raw - dark) / (white - dark)
  3. Synthesize RGB with Nikon D700 SRF
  4. Save (hsi, rgb) pairs as .npy

Output structure:
  output_dir/
    1_花香_1-1_hsi.npy      shape (H, W, 176), float32, [0, 1]
    1_花香_1-1_rgb.npy      shape (H, W, 3),   float32, [0, 1]
    ...
"""

import numpy as np
from pathlib import Path
import re

# ── Nikon D700 SRF ─────────────────────────────────────────────────────────────
# Jiang et al. "What is the Space of Spectral Sensitivity Functions for Digital
# Color Cameras?" CVPR 2013.  Wavelengths 400–720 nm, step 10 nm (33 points).
# To use your own SRF: replace these arrays with (3, N) sensitivity + wavelength.
_D700_WL = np.arange(400, 721, 10, dtype=np.float32)  # (33,)
_D700_SRF = np.array([
    # R
    [0.0006, 0.0008, 0.0015, 0.0026, 0.0044, 0.0073, 0.0111, 0.0127, 0.0132,
     0.0130, 0.0124, 0.0135, 0.0186, 0.0352, 0.0832, 0.1935, 0.3740, 0.5790,
     0.7491, 0.8613, 0.9167, 0.9316, 0.9101, 0.8430, 0.7241, 0.5524, 0.3434,
     0.1790, 0.0866, 0.0392, 0.0164, 0.0075, 0.0041],
    # G
    [0.0010, 0.0021, 0.0048, 0.0110, 0.0262, 0.0597, 0.1293, 0.2516, 0.4280,
     0.5910, 0.6878, 0.7191, 0.6880, 0.5871, 0.4530, 0.3320, 0.2405, 0.1789,
     0.1384, 0.1127, 0.0951, 0.0818, 0.0697, 0.0580, 0.0464, 0.0344, 0.0223,
     0.0135, 0.0080, 0.0048, 0.0026, 0.0014, 0.0007],
    # B
    [0.0117, 0.0325, 0.0850, 0.1863, 0.3390, 0.5090, 0.6510, 0.7250, 0.7104,
     0.6158, 0.4765, 0.3320, 0.2178, 0.1365, 0.0840, 0.0505, 0.0292, 0.0163,
     0.0091, 0.0050, 0.0025, 0.0013, 0.0006, 0.0003, 0.0001, 0.0001, 0.0000,
     0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
], dtype=np.float32)  # (3, 33)


# ── ENVI I/O ───────────────────────────────────────────────────────────────────

def parse_hdr(hdr_path: Path) -> dict:
    """Parse ENVI .hdr file into a metadata dict."""
    text = hdr_path.read_text(encoding='utf-8', errors='ignore')
    meta = {}

    # extract wavelength block before line-by-line parsing (it spans multiple lines)
    wl_match = re.search(r'wavelength\s*=\s*\{([^}]+)\}', text, re.DOTALL | re.IGNORECASE)
    if wl_match:
        meta['wavelength_array'] = np.array(
            re.findall(r'[\d.]+', wl_match.group(1)), dtype=np.float32
        )

    # parse single-line key = value pairs
    for line in text.splitlines():
        if '=' not in line or '{' in line:
            continue
        key, _, val = line.partition('=')
        meta[key.strip().lower()] = val.strip()

    return meta


def read_envi_bil(raw_path: Path, lines: int, samples: int, bands: int) -> np.ndarray:
    """Read BIL-interleaved uint16 ENVI raw file.

    BIL layout on disk: for each line, all bands of that line are stored
    contiguously → shape on disk is (lines, bands, samples).
    Returns (lines, samples, bands) i.e. standard (H, W, C).
    """
    data = np.fromfile(str(raw_path), dtype=np.uint16)
    data = data.reshape(lines, bands, samples)
    return data.transpose(0, 2, 1).astype(np.float32)  # (H, W, C)


def load_scan(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load hyperspectral data and wavelengths from a scan folder."""
    hdrs = list(folder.glob('*.hdr'))
    assert hdrs, f"No .hdr found in {folder}"
    hdr = parse_hdr(hdrs[0])
    lines   = int(hdr['lines'])
    samples = int(hdr['samples'])
    bands   = int(hdr['bands'])
    wl      = hdr['wavelength_array']
    raw_path = hdrs[0].with_suffix('.raw')
    data = read_envi_bil(raw_path, lines, samples, bands)
    return data, wl


# ── Calibration ────────────────────────────────────────────────────────────────

def calibrate(raw: np.ndarray, dark: np.ndarray, white: np.ndarray) -> np.ndarray:
    """Convert raw DN to reflectance, clip to [0, 1]."""
    denom = white - dark
    denom = np.where(denom < 1.0, 1.0, denom)  # avoid divide-by-zero
    ref = (raw - dark) / denom
    return np.clip(ref, 0.0, 1.0)


# ── RGB synthesis ──────────────────────────────────────────────────────────────

def build_srf_matrix(hsi_wl: np.ndarray,
                     srf_wl: np.ndarray = _D700_WL,
                     srf: np.ndarray = _D700_SRF) -> np.ndarray:
    """Interpolate SRF to the HSI wavelength grid.

    Returns (3, C) matrix, each row sums to 1 (energy-normalised).
    """
    C = len(hsi_wl)
    mat = np.zeros((3, C), dtype=np.float32)
    for i in range(3):
        mat[i] = np.interp(hsi_wl, srf_wl, srf[i], left=0.0, right=0.0)
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-8, 1.0, row_sum)
    return mat / row_sum


def synthesize_rgb(hsi: np.ndarray, srf_matrix: np.ndarray) -> np.ndarray:
    """Apply SRF matrix to hyperspectral cube.

    Args:
        hsi:        (H, W, C) float32 reflectance
        srf_matrix: (3, C)    float32

    Returns:
        rgb: (H, W, 3) float32 in [0, 1]
    """
    H, W, C = hsi.shape
    rgb = hsi.reshape(-1, C) @ srf_matrix.T   # (H*W, 3)
    return np.clip(rgb.reshape(H, W, 3), 0.0, 1.0)


# ── Dataset traversal ──────────────────────────────────────────────────────────

SAMPLE_PATTERN = re.compile(r'^\d+-\d+$')  # matches 1-1, 2-1, 3-2, etc.

def iter_kejian_groups(root: Path):
    """Yield (group_name, sample_subfolders, dark_folder, white_folder).

    Handles both flat variety folders (1-花香, 2-秘鲁进口, 3-丹东) and
    time-series folders (云南花香蓝莓/DayX).
    """
    for variety in sorted(root.iterdir()):
        if not variety.is_dir():
            continue
        # check if this variety has direct jinhongwai/kejian or Day*/
        day_dirs = [d for d in variety.iterdir() if d.is_dir() and d.name.startswith('Day')]
        if day_dirs:
            for day in sorted(day_dirs):
                kejian = day / 'kejian'
                if kejian.is_dir():
                    samples, dark, white = _collect_subfolders(kejian)
                    yield f"{variety.name}_{day.name}", samples, dark, white
        else:
            kejian = variety / 'kejian'
            if kejian.is_dir():
                samples, dark, white = _collect_subfolders(kejian)
                yield variety.name, samples, dark, white


def _collect_subfolders(kejian: Path):
    samples = sorted(
        d for d in kejian.iterdir()
        if d.is_dir() and SAMPLE_PATTERN.match(d.name)
    )
    dark  = kejian / 'dark'
    white = kejian / 'white'
    return samples, dark, white


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    root       = Path(r'F:\蓝莓')
    output_dir = Path(r'E:\2026\05\r2h\dataset')
    output_dir.mkdir(parents=True, exist_ok=True)

    srf_matrix = None  # built once after first wavelength array is seen

    for group_name, samples, dark_dir, white_dir in iter_kejian_groups(root):
        print(f"\n[{group_name}]")

        dark_data,  _ = load_scan(dark_dir)
        white_data, _ = load_scan(white_dir)

        for sample_dir in samples:
            out_stem = f"{group_name}_{sample_dir.name}".replace(' ', '_')
            hsi_path = output_dir / f"{out_stem}_hsi.npy"
            rgb_path = output_dir / f"{out_stem}_rgb.npy"

            if hsi_path.exists() and rgb_path.exists():
                print(f"  skip {out_stem} (already processed)")
                continue

            print(f"  processing {out_stem} ...", end=' ', flush=True)
            raw, wl = load_scan(sample_dir)

            if srf_matrix is None:
                srf_matrix = build_srf_matrix(wl)
                print(f"\n  SRF matrix built: visible bands "
                      f"{wl[(srf_matrix.sum(0) > 1e-4)][0]:.1f}–"
                      f"{wl[(srf_matrix.sum(0) > 1e-4)][-1]:.1f} nm")

            hsi = calibrate(raw, dark_data, white_data)
            rgb = synthesize_rgb(hsi, srf_matrix)

            np.save(str(hsi_path), hsi)
            np.save(str(rgb_path), rgb)
            print(f"HSI {hsi.shape}  RGB {rgb.shape}  saved.")

    print("\nDone.")


if __name__ == '__main__':
    main()
