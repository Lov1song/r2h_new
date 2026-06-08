"""
PyTorch Dataset for blueberry hyperspectral reconstruction.

Each .npz patch contains:
  'rgb': (256, 256, 3)   float32 [0, 1]
  'hsi': (256, 256, 176) float32 [0, 1]

__getitem__ returns:
  rgb: (3, H, W) tensor  — network input
  hsi: (C, H, W) tensor — reconstruction target (C=176 or len(band_indices))
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, List, Optional


class BlueberryHSIDataset(Dataset):
    def __init__(self, patch_dir: Union[str, Path], augment: bool = False,
                 band_indices: Optional[List[int]] = None):
        self.files        = sorted(Path(patch_dir).glob('*.npz'))
        self.augment      = augment
        self.band_indices = band_indices  # None → all bands
        assert self.files, f'No .npz files found in {patch_dir}'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(str(self.files[idx]))
        rgb  = data['rgb']   # (H, W, 3)
        hsi  = data['hsi']   # (H, W, C)

        if self.band_indices is not None:
            hsi = hsi[:, :, self.band_indices]

        if self.augment:
            rgb, hsi = self._augment(rgb, hsi)

        # HWC → CHW
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        hsi = torch.from_numpy(hsi.transpose(2, 0, 1).copy())
        return rgb, hsi

    @staticmethod
    def _augment(rgb: np.ndarray, hsi: np.ndarray):
        if np.random.rand() > 0.5:
            rgb = np.fliplr(rgb)
            hsi = np.fliplr(hsi)
        if np.random.rand() > 0.5:
            rgb = np.flipud(rgb)
            hsi = np.flipud(hsi)
        k = np.random.randint(0, 4)
        if k:
            rgb = np.rot90(rgb, k)
            hsi = np.rot90(hsi, k)
        return rgb, hsi


def get_dataloaders(patch_root: Union[str, Path], batch_size: int = 8,
                    num_workers: int = 4,
                    band_indices: Optional[List[int]] = None):
    patch_root = Path(patch_root)
    train_ds = BlueberryHSIDataset(patch_root / 'train', augment=True,  band_indices=band_indices)
    val_ds   = BlueberryHSIDataset(patch_root / 'val',   augment=False, band_indices=band_indices)
    test_ds  = BlueberryHSIDataset(patch_root / 'test',  augment=False, band_indices=band_indices)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
