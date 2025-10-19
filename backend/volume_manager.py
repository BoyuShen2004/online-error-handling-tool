"""
Proofreading Tool - Volume and Mask I/O Manager
-------------------------------------------
Handles loading, saving, and automatic alignment (transposing) of
2D and 3D biological image stacks and masks.

Supports:
- 2D images (.png, .jpg, .tif)
- 3D TIFF stacks (.tif/.tiff)
- Automatic orientation correction (Z, Y, X convention)
- Consistent _uploads/ directory handling for upload mode
"""

import numpy as np
import tifffile as tiff
import cv2
import os
from PIL import Image


# ------------------------------------------------------
# Utility: normalize 2D/3D array into uint8 grayscale
# ------------------------------------------------------
def _to_uint8(arr):
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - mn) / (mx - mn)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


# ------------------------------------------------------
# Core: load 2D or 3D image stack with auto axis correction
# ------------------------------------------------------
def load_image_or_stack(path):
    """
    Load an image or TIFF stack, automatically fixing axis orientation.

    Returns:
        np.ndarray with shape (Z, Y, X) for 3D stacks, (Y, X) for 2D images.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image/stack not found: {path}")

    ext = os.path.splitext(path)[-1].lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Failed to read image: {path}")
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        arr = _to_uint8(arr)
        print(f"✅ Loaded 2D image: shape={arr.shape}")
        return arr

    if ext in [".tif", ".tiff"]:
        arr = np.asarray(tiff.imread(path))
        arr = _to_uint8(arr)

        # Handle 2D single slice
        if arr.ndim == 2:
            print(f"✅ Loaded single-slice TIFF: shape={arr.shape}")
            return arr

        # Handle 3D stack (auto-detect orientation)
        if arr.ndim == 3:
            z, y, x = arr.shape
            print(f"📦 Raw TIFF shape: {arr.shape}")
            # Detect if middle dimension is smallest (likely Z)
            if y < z and y < x:
                arr = np.moveaxis(arr, 1, 0)
                print(f"🔄 Auto-transposed (H, Z, W) → (Z, H, W): {arr.shape}")
            elif x < y and x < z:
                arr = np.moveaxis(arr, 2, 0)
                print(f"🔄 Auto-transposed (W, H, Z) → (Z, H, W): {arr.shape}")
            else:
                print(f"✅ Orientation OK (Z, H, W): {arr.shape}")
            return arr

    raise ValueError(f"Unsupported file type: {ext}")


# ------------------------------------------------------
# Core: load or create binary mask matching the volume
# ------------------------------------------------------
def load_mask_like(mask_path, volume):
    """
    Load a binary mask that matches the given volume.
    Automatically corrects mismatched dimensions by transposing.

    Returns:
        np.ndarray mask with same shape as volume.
    """
    if mask_path is None or not os.path.exists(mask_path):
        print("ℹ️ No mask found — creating empty mask.")
        return np.zeros_like(volume, dtype=np.uint8)

    ext = os.path.splitext(mask_path)[-1].lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = _to_uint8(mask)
    else:
        mask = np.asarray(tiff.imread(mask_path))
        mask = _to_uint8(mask)

    print(f"📦 Raw mask shape: {mask.shape}")

    # If shapes match, done
    if mask.shape == volume.shape:
        print("✅ Mask shape matches volume.")
        return mask

    # --- Try automatic transpositions ---
    candidates = [
        (0, 1, 2),  # identity
        (1, 0, 2),  # swap first two
        (2, 1, 0),  # move last to first
        (1, 2, 0),  # H,W,Z → Z,H,W
        (2, 0, 1),  # W,H,Z → Z,H,W
        (0, 2, 1),  # Z,H,W → Z,W,H (rare)
    ]

    best = None
    for perm in candidates:
        if mask.ndim == 3 and tuple(np.array(mask.shape)[list(perm)]) == volume.shape:
            best = perm
            mask = np.transpose(mask, perm)
            print(f"🔄 Auto-transposed mask axes {perm} → {mask.shape}")
            break

    if best is None:
        print(f"⚠️ Could not automatically align mask. Resizing to volume shape...")
        mask2d = mask[0] if mask.ndim == 3 else mask
        mask2d = cv2.resize(mask2d, (volume.shape[-1], volume.shape[-2]), interpolation=cv2.INTER_NEAREST)
        if volume.ndim == 3:
            mask = np.stack([mask2d] * volume.shape[0], axis=0)
        else:
            mask = mask2d

    print(f"✅ Final mask shape: {mask.shape}")
    return (mask > 0).astype(np.uint8)


# ------------------------------------------------------
# Core: save mask to disk (auto extension)
# ------------------------------------------------------
def save_mask(mask, path):
    """
    Save binary mask to disk as TIFF or PNG depending on file extension.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mask = (mask > 0).astype(np.uint8)

    ext = os.path.splitext(path)[-1].lower()
    if ext in [".tif", ".tiff"]:
        tiff.imwrite(path, mask)
    elif ext in [".png", ".jpg", ".jpeg"]:
        im = Image.fromarray(mask * 255)
        im.save(path)
    else:
        # Default fallback to TIFF
        tiff.imwrite(path + ".tif", mask)
        path += ".tif"

    print(f"💾 Saved mask → {path} ({mask.shape})")


# ------------------------------------------------------
# Simple volume wrapper class (optional)
# ------------------------------------------------------
class Volume:
    """
    Wrapper class representing a loaded 2D/3D biological volume.
    Provides convenient properties for proofreading pipelines.
    """
    def __init__(self, path):
        self.path = path
        self.data = load_image_or_stack(path)
        self.shape = self.data.shape
        self.ndim = self.data.ndim

    def empty_mask(self):
        """Return an empty mask matching the volume shape."""
        return np.zeros_like(self.data, dtype=np.uint8)

    def save(self, out_path):
        """Save current volume data (mainly for debugging)."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tiff.imwrite(out_path, self.data)
        print(f"💾 Volume saved to {out_path}")
