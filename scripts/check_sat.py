import numpy as np
from pathlib import Path

files = sorted(Path('data').glob('*.npz'))
print(f'共 {len(files)} 个样本')
print(f'{"样本":<12} {"HSI_max":>8} {"HSI>0.99":>10} {"RGB>=254":>10}')
for f in files:
    d = np.load(str(f))
    mask = d['valid_mask']
    hsi = d['hsi'][mask]
    rgb = d['rgb'][mask]
    hsi_sat = (hsi > 0.99).mean()
    rgb_sat = (rgb >= 254).mean()
    print(f'{f.stem:<12} {hsi.max():>8.4f} {hsi_sat*100:>9.3f}% {rgb_sat*100:>9.3f}%')
