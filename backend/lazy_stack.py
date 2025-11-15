import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tifffile

from backend.volume_manager import _to_uint8


def _read_grayscale_slice(path: str) -> np.ndarray:
    """Load a single image slice from disk and normalize to uint8 grayscale."""
    ext = os.path.splitext(path.lower())[1]
    if ext in [".tif", ".tiff"]:
        arr = np.asarray(tifffile.imread(path))
        if arr.ndim == 3:
            # if multi-channel, collapse to first channel
            arr = arr[0] if arr.shape[0] <= 4 else arr.mean(axis=0)
    else:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Failed to read image: {path}")
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return _to_uint8(arr)


def _resize_if_needed(arr: np.ndarray, target_shape: Optional[Tuple[int, int]]) -> np.ndarray:
    if target_shape is None or arr.shape == target_shape:
        return arr
    return cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)


class LazySliceLoader:
    """Lazily load 2D slices from a list of file paths."""

    def __init__(self, files: Sequence[str]):
        if not files:
            raise ValueError("Empty slice list for lazy loader")
        self.files: List[str] = list(files)
        sample = _read_grayscale_slice(self.files[0])
        self.slice_shape: Tuple[int, int] = sample.shape[-2], sample.shape[-1]
        self.dtype = sample.dtype
        self.ndim = 3
        self._num_slices = len(self.files)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self._num_slices, self.slice_shape[0], self.slice_shape[1])

    def __len__(self) -> int:
        return self._num_slices

    def get_slice(self, index: int) -> np.ndarray:
        idx = max(0, min(index, self._num_slices - 1))
        slice_arr = _read_grayscale_slice(self.files[idx])
        return _resize_if_needed(slice_arr, self.slice_shape)


class LazyMaskLoader:
    """Lazily load mask slices paired with image files."""

    def __init__(self, mask_paths: Sequence[Optional[str]], target_shape: Tuple[int, int]):
        if not mask_paths:
            raise ValueError("Mask loader requires at least one entry")
        self.mask_paths: List[Optional[str]] = list(mask_paths)
        self.slice_shape = target_shape
        self._num_slices = len(self.mask_paths)

    def __len__(self) -> int:
        return self._num_slices

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self._num_slices, self.slice_shape[0], self.slice_shape[1])

    @property
    def ndim(self) -> int:
        return 3

    def _load_mask_slice(self, path: str) -> np.ndarray:
        ext = os.path.splitext(path.lower())[1]
        if ext in [".png", ".jpg", ".jpeg"]:
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError(f"Failed to read mask: {path}")
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            arr = np.asarray(tifffile.imread(path))
            if arr.ndim > 2:
                arr = arr[0]
        arr = _to_uint8(arr)
        return (arr > 0).astype(np.uint8)

    def get_slice(self, index: int) -> np.ndarray:
        idx = max(0, min(index, self._num_slices - 1))
        path = self.mask_paths[idx]
        if not path:
            return np.zeros(self.slice_shape, dtype=np.uint8)
        mask_arr = self._load_mask_slice(path)
        mask_arr = _resize_if_needed(mask_arr, self.slice_shape)
        return (mask_arr > 0).astype(np.uint8)

