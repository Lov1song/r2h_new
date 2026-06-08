"""
RGB-Hyperspectral 空间配准 + 训练样本提取
------------------------------------------
输入:
  F:\\rgb\\{name}.bmp          — RGB 图像 (3536×3536)
  F:\\kejian\\{name}\\*.bmp    — 高光谱缩略图 (960×991, 配准参考)
  F:\\kejian\\{name}\\*.raw/.hdr — 高光谱立方体 (960×991×176, BIL uint16)
  F:\\kejian\\dark\\ / white\\ — 辐射校正参考

流程:
  灰度化 RGB → SIFT 特征匹配 → RANSAC Homography
  → warpPerspective 将 RGB 映射到高光谱坐标系
  → 逐像素配对 (RGB×3, 光谱×176) 保存为 .npz

输出:
  E:\\2026\\05\\r2h\\data\\{name}.npz
  E:\\2026\\05\\r2h\\data\\vis_{name}.png
"""

import os
import re
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
# ──────────────────────────── 路径配置 ────────────────────────────
RGB_DIR    = r"F:\rgb"
KEJIAN_DIR = r"F:\kejian"
OUTPUT_DIR = r"E:\2026\05\r2h\data"

SAMPLE_NAME = "cc-3"   # 修改此处测试不同样本
SAVE_NPZ    = True    # 是否保存 .npz 训练数据

# ──────────────────────────── ENVI 读取 ───────────────────────────
def parse_hdr(hdr_path: str) -> dict:
    """解析 ENVI .hdr 文件，返回元信息字典"""
    with open(hdr_path, "r") as f:
        text = f.read()

    info = {}
    for key in ["samples", "lines", "bands", "data type", "header offset"]:
        m = re.search(rf"{key}\s*=\s*(\d+)", text)
        if m:
            info[key.replace(" ", "_")] = int(m.group(1))

    m = re.search(r"interleave\s*=\s*(\w+)", text)
    if m:
        info["interleave"] = m.group(1).lower()

    m = re.search(r"wavelength\s*=\s*\{([^}]+)\}", text, re.DOTALL)
    if m:
        info["wavelengths"] = np.array(
            [float(x) for x in m.group(1).split(",") if x.strip()]
        )
    return info


def read_envi(raw_path: str, hdr_path: str):
    """读取 ENVI 原始文件 → (H, W, C) float32"""
    info = parse_hdr(hdr_path)
    H   = info["lines"]
    W   = info["samples"]
    C   = info["bands"]
    offset = info.get("header_offset", 0)

    dtype_map = {12: np.uint16, 4: np.float32, 2: np.int16, 1: np.uint8}
    dtype = dtype_map.get(info.get("data_type", 12), np.uint16)

    data = np.fromfile(raw_path, dtype=dtype, offset=offset)
    interleave = info.get("interleave", "bil")

    if interleave == "bil":
        data = data.reshape((H, C, W)).transpose(0, 2, 1)   # → (H, W, C)
    elif interleave == "bsq":
        data = data.reshape((C, H, W)).transpose(1, 2, 0)
    elif interleave == "bip":
        data = data.reshape((H, W, C))

    return data.astype(np.float32), info


def find_files(folder: str):
    """返回文件夹内第一个 (.bmp, .hdr, .raw) 路径"""
    def first(pattern):
        results = glob.glob(os.path.join(folder, pattern))
        return results[0] if results else None
    return first("*.bmp"), first("*.hdr"), first("*.raw")


# ──────────────────────────── 辐射校正 ────────────────────────────
def load_reference(folder: str) -> np.ndarray:
    """读取 dark/white 参考，取空间均值 → (1, 1, C)"""
    _, hdr, raw = find_files(folder)
    data, _ = read_envi(raw, hdr)          # (H, W, C)
    return data.mean(axis=(0, 1), keepdims=True)  # (1, 1, C)


def reflectance_correct(cube: np.ndarray,
                        dark: np.ndarray,
                        white: np.ndarray) -> np.ndarray:
    """反射率校正: (cube - dark) / (white - dark)，结果裁剪到 [0, 1]"""
    denom = white - dark
    denom[denom < 1e-4] = 1e-4
    ref = (cube - dark) / denom
    return np.clip(ref, 0.0, 1.0).astype(np.float32)


# ──────────────────────────── 培养皿掩码 ─────────────────────────
def create_dish_mask(hs_ref: np.ndarray, warp_valid: np.ndarray) -> np.ndarray:
    """
    生成有效训练像素的掩码：培养皿区域 AND warp 有效区域。

    策略：
      1. 对 HSI 反射率取各波段均值 → 培养皿区域明显亮于黑色背景
      2. Otsu 阈值分割前景（培养皿）与背景
      3. 形态学闭运算填补孔洞，开运算去除小噪点
      4. 保留最大连通域（培养皿本体）
      5. 与 warp 有效区域取交集

    返回: bool 掩码 (H, W)
    """
    # 各波段均值图，放大到 uint8 供 OpenCV 处理
    mean_img = hs_ref.mean(axis=2)                          # (H, W) float32
    norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min() + 1e-8)
    norm_u8 = (norm * 255).astype(np.uint8)

    # Otsu 自动阈值
    _, binary = cv2.threshold(norm_u8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学清理（椭圆核，尺寸根据图像分辨率自适应）
    ksize = max(7, min(norm_u8.shape) // 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=2)

    # 保留最大连通域（培养皿）
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_labels > 1:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = ((labels == largest_label) * 255).astype(np.uint8)

    dish_mask = binary.astype(bool)

    # 与 warp 有效区域（黑边排除）取交集
    valid = dish_mask & warp_valid
    print(f"  掩码统计: 培养皿区域={dish_mask.sum():,}  "
          f"warp有效={warp_valid.sum():,}  最终有效={valid.sum():,}")
    return valid


# ──────────────────────────── 配准 ────────────────────────────────
def register_rgb_to_hsi(rgb_bgr: np.ndarray,
                        hs_thumb_gray: np.ndarray):
    """
    计算 Homography H，使得
        warpPerspective(rgb_resized, H, (W_hs, H_hs))
    将缩放后的 RGB 变换到高光谱坐标系。

    返回: H (3×3) | None, kp_rgb, kp_hs, good_matches, rgb_resized_bgr
    """
    # 先将 RGB 缩放到与 HSI 缩略图相同的分辨率，消除尺度差异
    H_hs, W_hs = hs_thumb_gray.shape
    rgb_resized = cv2.resize(rgb_bgr, (W_hs, H_hs), interpolation=cv2.INTER_AREA)
    rgb_gray = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2GRAY)

    # CLAHE 增强局部对比度（HSI 缩略图往往很平坦，直接 SIFT 特征点稀少）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    rgb_eq  = clahe.apply(rgb_gray)
    hsi_eq  = clahe.apply(hs_thumb_gray)

    # contrastThreshold 降低到 0.02（默认 0.04），在低对比区域多检测特征点
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    kp_rgb, des_rgb = sift.detectAndCompute(rgb_eq, None)
    kp_hs,  des_hs  = sift.detectAndCompute(hsi_eq, None)

    print(f"  特征点数量  RGB: {len(kp_rgb)}  HSI: {len(kp_hs)}")

    if des_rgb is None or des_hs is None or len(kp_rgb) < 10 or len(kp_hs) < 10:
        print("  [!] 特征点不足，配准失败")
        return None, kp_rgb, kp_hs, [], rgb_resized

    # FLANN 匹配
    flann = cv2.FlannBasedMatcher(
        {"algorithm": 1, "trees": 5},
        {"checks": 50}
    )
    matches = flann.knnMatch(des_rgb, des_hs, k=2)

    # Lowe 比率测试
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    print(f"  Lowe 筛选后匹配数: {len(good)}")

    if len(good) < 10:
        print("  [!] 有效匹配不足，配准失败")
        return None, kp_rgb, kp_hs, good, rgb_resized

    pts_rgb = np.float32([kp_rgb[m.queryIdx].pt for m in good])
    pts_hs  = np.float32([kp_hs [m.trainIdx].pt for m in good])

    # pts_rgb → pts_hs 的单应矩阵
    H_mat, mask = cv2.findHomography(pts_rgb, pts_hs, cv2.RANSAC, 5.0)
    inliers = int(mask.ravel().sum())
    print(f"  RANSAC 内点: {inliers} / {len(good)}")

    return H_mat, kp_rgb, kp_hs, good, rgb_resized


# ──────────────────────────── 可视化 ──────────────────────────────
def _band_idx(wavelengths: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(wavelengths - target_nm)))


def visualize(rgb_bgr, hs_thumb_gray, hs_ref, wavelengths,
              kp_rgb, kp_hs, good_matches, H_mat, warped_rgb,
              sample_name: str, show: bool = True):

    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    H_hs, W_hs = hs_thumb_gray.shape

    # 假彩色合成 (800 / 670 / 550 nm)
    fc = np.stack([
        hs_ref[:, :, _band_idx(wavelengths, 800)],
        hs_ref[:, :, _band_idx(wavelengths, 670)],
        hs_ref[:, :, _band_idx(wavelengths, 550)],
    ], axis=2)
    fc = np.clip(fc / (fc.max() + 1e-6), 0, 1)

    fig = plt.figure(figsize=(22, 11))
    fig.suptitle(f"样本: {sample_name}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

    # ① 原始 RGB
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb_rgb)
    ax.set_title(f"RGB\n({rgb_rgb.shape[1]}×{rgb_rgb.shape[0]})")
    ax.axis("off")

    # ② 高光谱缩略图
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(hs_thumb_gray, cmap="gray")
    ax.set_title(f"HSI 缩略图\n({W_hs}×{H_hs})")
    ax.axis("off")

    # ③ 特征匹配图（对 RGB 按比例缩小后绘制）
    ax = fig.add_subplot(gs[0, 2:])
    scale = min(1.0, 800 / max(rgb_bgr.shape[:2]))
    rgb_small = cv2.resize(rgb_bgr, None, fx=scale, fy=scale)
    kp_rgb_scaled = [cv2.KeyPoint(k.pt[0] * scale, k.pt[1] * scale, k.size * scale)
                     for k in kp_rgb]
    hs_small_bgr = cv2.cvtColor(hs_thumb_gray, cv2.COLOR_GRAY2BGR)
    match_img = cv2.drawMatches(
        rgb_small, kp_rgb_scaled,
        hs_small_bgr, kp_hs,
        good_matches[:60], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    ax.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"特征匹配 (显示前60条，共 {len(good_matches)} 条)")
    ax.axis("off")

    # ④ 配准后 RGB（变换到 HSI 坐标系）
    ax = fig.add_subplot(gs[1, 0])
    if warped_rgb is not None:
        ax.imshow(warped_rgb)
        ax.set_title("配准后 RGB\n(→ HSI 坐标系)")
    else:
        ax.set_title("配准失败")
    ax.axis("off")

    # ⑤ 高光谱假彩色
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(fc)
    ax.set_title("HSI 假彩色\n(800 / 670 / 550 nm)")
    ax.axis("off")

    # ⑥ 叠加对比
    ax = fig.add_subplot(gs[1, 2])
    if warped_rgb is not None:
        rgb_f = warped_rgb.astype(np.float32) / 255.0
        blend = np.clip(rgb_f * 0.5 + fc * 0.5, 0, 1)
        ax.imshow(blend)
        ax.set_title("叠加对比\n(配准RGB × 0.5 + HSI假彩色 × 0.5)")
    ax.axis("off")

    # ⑦ 随机采样光谱曲线
    ax = fig.add_subplot(gs[1, 3])
    if warped_rgb is not None:
        valid_y, valid_x = np.where(warped_rgb[:, :, 0] > 5)
        if len(valid_y) > 0:
            n_samples = min(8, len(valid_y))
            chosen = np.random.choice(len(valid_y), n_samples, replace=False)
            for idx in chosen:
                y, x = valid_y[idx], valid_x[idx]
                spectrum = hs_ref[y, x, :]
                r, g, b  = warped_rgb[y, x] / 255.0
                ax.plot(wavelengths, spectrum, color=(r, g, b),
                        linewidth=1.5, alpha=0.85)
    ax.set_xlabel("波长 (nm)")
    ax.set_ylabel("反射率")
    ax.set_title("随机像素光谱\n(线条颜色 = 该像素RGB值)")
    ax.grid(True, alpha=0.3)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"vis_{sample_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"  可视化已保存: {save_path}")


# ──────────────────────────── 主流程 ──────────────────────────────
def process_sample(sample_name: str, save_npz: bool = True, show: bool = True):
    print(f"\n{'='*55}")
    print(f"  样本: {sample_name}")
    print(f"{'='*55}")

    # 1. 读取 RGB
    rgb_path = os.path.join(RGB_DIR, f"{sample_name}.bmp")
    rgb_bgr = cv2.imread(rgb_path)
    if rgb_bgr is None:
        print(f"  [!] 未找到 RGB: {rgb_path}")
        return
    print(f"  RGB 尺寸: {rgb_bgr.shape[1]}×{rgb_bgr.shape[0]}")

    # 2. 读取高光谱缩略图 + 立方体
    hs_folder = os.path.join(KEJIAN_DIR, sample_name)
    bmp_p, hdr_p, raw_p = find_files(hs_folder)
    if not all([bmp_p, hdr_p, raw_p]):
        print(f"  [!] 高光谱文件不完整: {hs_folder}")
        return

    hs_thumb = cv2.imread(bmp_p, cv2.IMREAD_GRAYSCALE)
    hs_cube, info = read_envi(raw_p, hdr_p)
    wavelengths = info.get("wavelengths", np.linspace(402.6, 1005.5, 176))
    print(f"  HSI 尺寸: {hs_cube.shape[1]}×{hs_cube.shape[0]}  波段: {hs_cube.shape[2]}")
    print(f"  波长范围: {wavelengths[0]:.1f} – {wavelengths[-1]:.1f} nm")

    # 3. 辐射校正
    dark  = load_reference(os.path.join(KEJIAN_DIR, "dark"))
    white = load_reference(os.path.join(KEJIAN_DIR, "white"))
    hs_ref = reflectance_correct(hs_cube, dark, white)
    print(f"  校正后反射率范围: [{hs_ref.min():.3f}, {hs_ref.max():.3f}]")

    # 4. 配准（RGB 先缩放到 HSI 分辨率再匹配，消除尺度差异）
    H_hs, W_hs = hs_thumb.shape
    H_mat, kp_rgb, kp_hs, good_matches, rgb_resized = register_rgb_to_hsi(rgb_bgr, hs_thumb)

    warped_rgb = None
    if H_mat is not None:
        warped_bgr = cv2.warpPerspective(rgb_resized, H_mat, (W_hs, H_hs))
        warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
        print(f"  配准成功，变换后 RGB 尺寸: {warped_rgb.shape}")
    else:
        print("  [!] 配准失败，跳过保存")

    # 5. 可视化（传入缩放后的 RGB，与特征点坐标一致）
    visualize(rgb_resized, hs_thumb, hs_ref, wavelengths,
              kp_rgb, kp_hs, good_matches, H_mat, warped_rgb, sample_name, show=show)

    # 6. 保存训练数据
    if save_npz and warped_rgb is not None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # 先用 warp 填充掩码排除透视变换黑边，再用 Otsu 阈值提取培养皿区域
        warp_valid = np.any(warped_rgb > 5, axis=2)
        valid_mask = create_dish_mask(hs_ref, warp_valid)
        out_path = os.path.join(OUTPUT_DIR, f"{sample_name}.npz")
        np.savez_compressed(
            out_path,
            rgb        = warped_rgb,    # (H_hs, W_hs, 3)   uint8
            hsi        = hs_ref,        # (H_hs, W_hs, 176) float32
            wavelengths= wavelengths,   # (176,)
            valid_mask = valid_mask,    # (H_hs, W_hs)      bool
            H          = H_mat,         # (3, 3)  备用
        )
        n_valid = int(valid_mask.sum())
        print(f"  有效像素: {n_valid:,}  → {out_path}")
        del hs_cube   # 释放最大的数组（991×960×176 float32 ≈ 671MB）

    return warped_rgb, hs_ref


# ──────────────────────────── 批量处理 ───────────────────────────
def process_all(save_npz: bool = True):
    """批量处理所有样本，跳过已完成的，记录成功/失败统计"""
    import gc
    all_bmp = glob.glob(os.path.join(RGB_DIR, "*.bmp"))
    names = sorted([os.path.splitext(os.path.basename(p))[0] for p in all_bmp])
    print(f"\n共找到 {len(names)} 个 RGB 样本，开始批量处理...\n")

    success, failed, skipped = [], [], []
    for name in names:
        npz_path = os.path.join(OUTPUT_DIR, f"{name}.npz")
        # 跳过已存在且大小正常（>50MB）的文件
        if os.path.exists(npz_path) and os.path.getsize(npz_path) > 50 * 1024 * 1024:
            print(f"  [skip] {name}  ({os.path.getsize(npz_path)//1024//1024} MB)")
            skipped.append(name)
            continue
        try:
            result = process_sample(name, save_npz=save_npz, show=False)
            if result[0] is not None:
                success.append(name)
            else:
                failed.append(name)
        except Exception as e:
            print(f"  [!] {name} 异常: {e}")
            failed.append(name)
        finally:
            gc.collect()   # 每个样本处理完强制释放内存

    print(f"\n{'='*55}")
    print(f"  批量完成: 成功={len(success)}  跳过={len(skipped)}  失败={len(failed)}")
    if failed:
        print(f"  失败样本: {failed}")
    print(f"{'='*55}")


# ──────────────────────────── 入口 ────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        # python register_and_extract.py all
        process_all(save_npz=True)
    else:
        # python register_and_extract.py  ← 单样本验证
        process_sample(SAMPLE_NAME, save_npz=SAVE_NPZ)
