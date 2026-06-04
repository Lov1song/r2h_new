"""
从配准后的全图 npz 切 128×128 patch，按样本 8:1:1 划分 train/val/test。

输入:  E:\\2026\\05\\r2h\\data\\{name}.npz
         rgb  (H, W, 3)   uint8  [0, 255]
         hsi  (H, W, 176) float32 [0, 1]
         valid_mask (H, W) bool   — warp 黑边排除

输出:  E:\\2026\\05\\r2h\\dataset\\patches\\{train,val,test}\\{name}_p{i:03d}.npz
         rgb  (128, 128, 3)   float32 [0, 1]
         hsi  (128, 128, 176) float32 [0, 1]

split 记录: E:\\2026\\05\\r2h\\dataset\\split.json
"""

import json
import random
from pathlib import Path

import numpy as np


def make_colony_mask(hsi_hwc, valid_mask):
    """Otsu on HSI mean reflectance within dish area → colony pixels."""
    mean_ref  = hsi_hwc.mean(axis=2)
    dish_vals = mean_ref[valid_mask]
    counts, edges = np.histogram(dish_vals, bins=256, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) * 0.5
    total   = counts.sum()
    s_total = (counts * centers).sum()
    best_thresh, best_var, w0, s0 = 0.0, -1.0, 0, 0.0
    for i in range(256):
        w0 += counts[i]; s0 += counts[i] * centers[i]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = s0 / w0
        m1 = (s_total - s0) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_thresh = var, centers[i]
    return valid_mask & (mean_ref >= best_thresh)

# ─────────────────────────── 配置 ────────────────────────────────
DATA_DIR   = Path(r'E:\2026\05\r2h\data')
OUT_DIR    = Path(r'E:\2026\05\r2h\dataset\patches')
SPLIT_JSON = Path(r'E:\2026\05\r2h\dataset\split.json')

PATCH_SIZE     = 256
SEED           = 42
SPLIT          = (0.8, 0.1, 0.1)
MIN_VALID_FRAC   = 0.7   # patch 内培养皿像素占比低于此则跳过
MIN_COLONY_FRAC  = 0.05  # patch 内菌落像素占比低于此则跳过（过滤纯培养基 patch）


# ─────────────────────────── 划分 ────────────────────────────────
def split_samples(names: list[str],
                  ratio: tuple = (0.8, 0.1, 0.1),
                  seed: int = 42):
    """按品种内 8:1:1 划分，保证每个品种在 val/test 都有样本。"""
    from collections import defaultdict
    rng = random.Random(seed)

    # 按品种分组
    variety_map: dict[str, list[str]] = defaultdict(list)
    for name in names:
        # 品种 = 去掉末尾 "-数字"
        variety = '-'.join(name.split('-')[:-1]) if '-' in name else name
        variety_map[variety].append(name)

    train, val, test = [], [], []
    for variety, samples in sorted(variety_map.items()):
        s = samples[:]
        rng.shuffle(s)
        n = len(s)
        n_val  = max(1, round(n * ratio[1]))
        n_test = max(1, round(n * ratio[2]))
        test  += s[:n_test]
        val   += s[n_test:n_test + n_val]
        train += s[n_test + n_val:]

    return train, val, test


# ─────────────────────────── Patch 提取 ──────────────────────────
def extract_patches(rgb_u8: np.ndarray,
                    hsi: np.ndarray,
                    valid_mask: np.ndarray,
                    colony_mask: np.ndarray,
                    size: int = 128,
                    min_valid: float = 0.7,
                    min_colony: float = 0.05):
    """
    非重叠滑窗切 patch。
    跳过条件：培养皿覆盖率 < min_valid，或菌落占比 < min_colony。
    rgb_u8:      (H, W, 3) uint8
    hsi:         (H, W, C) float32
    valid_mask:  (H, W) bool  — 培养皿区域
    colony_mask: (H, W) bool  — 菌落区域
    yield: (rgb_patch float32 [0,1], hsi_patch float32 [0,1])
    """
    H, W = hsi.shape[:2]
    for r in range(0, H - size + 1, size):
        for c in range(0, W - size + 1, size):
            patch_valid  = valid_mask [r:r+size, c:c+size]
            patch_colony = colony_mask[r:r+size, c:c+size]
            if patch_valid.mean() < min_valid:
                continue
            if patch_colony.mean() < min_colony:
                continue
            rgb_patch = rgb_u8[r:r+size, c:c+size, :].astype(np.float32) / 255.0
            hsi_patch = hsi   [r:r+size, c:c+size, :]
            yield rgb_patch, hsi_patch


# ─────────────────────────── 处理一个 split ──────────────────────
def process_split(names: list[str], split_name: str):
    out_dir = OUT_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    total_patches = 0
    for name in names:
        src = DATA_DIR / f'{name}.npz'
        if not src.exists():
            print(f'  [skip] {name}: npz 不存在')
            continue

        data = np.load(str(src))
        rgb_u8     = data['rgb']        # (H, W, 3) uint8
        hsi        = data['hsi']        # (H, W, C) float32
        valid_mask = data['valid_mask'] # (H, W) bool
        colony_mask = make_colony_mask(hsi, valid_mask)
        n_col = int(colony_mask.sum())
        n_dish = int(valid_mask.sum())
        print(f'    colony Otsu: {n_col:,}/{n_dish:,} ({100*n_col/max(n_dish,1):.1f}%)')

        count = 0
        for i, (rp, hp) in enumerate(
                extract_patches(rgb_u8, hsi, valid_mask, colony_mask,
                                PATCH_SIZE, MIN_VALID_FRAC, MIN_COLONY_FRAC)):
            np.savez_compressed(
                str(out_dir / f'{name}_p{i:03d}.npz'),
                rgb=rp,
                hsi=hp,
            )
            count += 1

        total_patches += count
        print(f'  {split_name}/{name}: {count} patches')

    print(f'  [{split_name}] 共 {total_patches} patches，来自 {len(names)} 个样本\n')
    return total_patches


# ─────────────────────────── 主函数 ──────────────────────────────
def main():
    all_npz = sorted(DATA_DIR.glob('*.npz'))
    if not all_npz:
        raise FileNotFoundError(f'未找到 npz 文件: {DATA_DIR}')

    names = [f.stem for f in all_npz]
    print(f'共找到 {len(names)} 个样本')

    train, val, test = split_samples(names, SPLIT, SEED)
    print(f'划分 → train:{len(train)}  val:{len(val)}  test:{len(test)}')

    SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_JSON.write_text(
        json.dumps({'train': train, 'val': val, 'test': test},
                   ensure_ascii=False, indent=2)
    )
    print(f'split.json 已保存\n')

    n_train = process_split(train, 'train')
    n_val   = process_split(val,   'val')
    n_test  = process_split(test,  'test')

    total = n_train + n_val + n_test
    print(f'完成。patches → {OUT_DIR}')
    print(f'  train:{n_train}  val:{n_val}  test:{n_test}  total:{total}')
    print(f'  每个 patch: RGB (128,128,3) float32 + HSI (128,128,176) float32')


if __name__ == '__main__':
    main()
