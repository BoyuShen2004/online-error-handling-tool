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
import glob


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
    # Support: directory path or glob pattern to stack multiple 2D images
    if any(ch in str(path) for ch in ["*", "?", "["]):
        file_list = sorted(glob.glob(path))
        if not file_list:
            raise FileNotFoundError(f"No files match pattern: {path}")
        return _load_and_stack_2d_images(file_list)

    if os.path.isdir(path):
        # Stack all 2D images of the same type in the directory
        # Prefer common bio-imaging extensions
        candidates = []
        for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
            candidates = sorted([
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith(ext)
            ])
            if candidates:
                break
        if not candidates:
            raise FileNotFoundError(f"No supported images found in directory: {path}")
        return _load_and_stack_2d_images(candidates)

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
# Helper: load multiple 2D images and stack to (Z, Y, X)
# ------------------------------------------------------
def _load_and_stack_2d_images(file_list):
    if not file_list:
        raise ValueError("Empty file list for stacking")
    # Ensure consistent extension/type
    exts = {os.path.splitext(f)[-1].lower() for f in file_list}
    if len(exts) > 1:
        # Mixed types allowed but warn via print; continue by loading each
        print(f"ℹ️ Mixed extensions detected in stack: {sorted(list(exts))}")

    slices = []
    target_shape = None
    for idx, fp in enumerate(file_list):
        ext = os.path.splitext(fp)[-1].lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            arr = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError(f"Failed to read image: {fp}")
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif ext in [".tif", ".tiff"]:
            arr = np.asarray(tiff.imread(fp))
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D TIFF for stacking, got shape {arr.shape} at {fp}")
        else:
            raise ValueError(f"Unsupported file type in stack: {ext}")
        arr = _to_uint8(arr)
        if target_shape is None:
            target_shape = arr.shape
        elif arr.shape != target_shape:
            # Resize to match the first slice using nearest-neighbor
            arr = cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        slices.append(arr)

    stack = np.stack(slices, axis=0)
    print(f"✅ Loaded {len(file_list)} 2D images as stack: shape={stack.shape}")
    return stack


# ------------------------------------------------------
# Public wrapper: stack a list of 2D images to (Z, Y, X)
# ------------------------------------------------------
def stack_2d_images(file_list):
    """
    Public helper to stack multiple 2D image files into a 3D volume (Z,Y,X).
    """
    return _load_and_stack_2d_images(file_list)


# ------------------------------------------------------
# Helper: list image files for stacking from path/list/glob
# Excludes files that already look like masks (suffix "_mask").
# ------------------------------------------------------
def list_images_for_path(path):
    if isinstance(path, list):
        files = list(path)
    elif any(ch in str(path) for ch in ["*", "?", "["]):
        files = sorted(glob.glob(path))
    elif os.path.isdir(path):
        files = []
        for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
            cand = sorted([
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith(ext)
            ])
            if cand:
                files = cand
                break
    else:
        files = [path]

    # Exclude already-generated mask files (<base>_mask.*)
    filtered = []
    for fp in files:
        base, ext = os.path.splitext(os.path.basename(fp))
        if base.endswith("_mask") or base.endswith("_prediction"):
            continue
        filtered.append(fp)

    return filtered


# ------------------------------------------------------
# Helper: build mask stack for given image files from same or other dir
# Mask naming convention: <image_basename>_mask<ext>
# If a mask doesn't exist, returns a zero slice for that position
# ------------------------------------------------------
def build_mask_stack_from_pairs(image_files, mask_base_dir=None):
    if not image_files:
        return None
    mask_paths = build_mask_path_mapping(image_files, mask_base_dir=mask_base_dir)
    # Load first image to get target H,W
    first = image_files[0]
    ext_first = os.path.splitext(first)[-1].lower()
    # read first image dims
    if ext_first in [".png", ".jpg", ".jpeg"]:
        arr0 = cv2.imread(first, cv2.IMREAD_UNCHANGED)
        if arr0 is None:
            raise ValueError(f"Failed to read image: {first}")
        if arr0.ndim == 3:
            arr0 = cv2.cvtColor(arr0, cv2.COLOR_BGR2GRAY)
    else:
        arr0 = np.asarray(tiff.imread(first))
        if arr0.ndim != 2:
            raise ValueError(f"Expected 2D image for mask pairing, got {arr0.shape} at {first}")
    h, w = arr0.shape[-2], arr0.shape[-1]

    mask_slices = []
<<<<<<< HEAD
    for mask_fp in mask_paths:
=======
    for img_fp in image_files:
        base, ext = os.path.splitext(os.path.basename(img_fp))
        # Prefer same extension mask
        candidates = []
        if mask_base_dir is None:
            search_dir = os.path.dirname(img_fp)
        else:
            search_dir = mask_base_dir

        # try same ext first, then fallback to common ones
        try_exts = [ext.lower()]
        for e in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
            if e not in try_exts:
                try_exts.append(e)

        mask_fp = None
        # 1) Conventional: <base>_mask<ext>
        for e in try_exts:
            candidate = os.path.join(search_dir, f"{base}_mask{e}")
            if os.path.exists(candidate):
                mask_fp = candidate
                break

        # 1b) Model prediction naming: <base>_prediction<ext>
        if mask_fp is None:
            for e in try_exts:
                candidate = os.path.join(search_dir, f"{base}_prediction{e}")
                if os.path.exists(candidate):
                    mask_fp = candidate
                    break

        # 2) nnUNet preprocessed style: mask name equals image name without trailing "_0000"
        if mask_fp is None and base.endswith("_0000"):
            trimmed = base[:-5]  # remove suffix "_0000"
            for e in try_exts:
                candidate = os.path.join(search_dir, f"{trimmed}{e}")
                if os.path.exists(candidate):
                    mask_fp = candidate
                    break

>>>>>>> 3f56e9f990b99809b40b97f25513b326deedeb57
        if mask_fp is None:
            mask_slices.append(np.zeros((h, w), dtype=np.uint8))
            continue

        mext = os.path.splitext(mask_fp)[-1].lower()
        if mext in [".png", ".jpg", ".jpeg"]:
            marr = cv2.imread(mask_fp, cv2.IMREAD_UNCHANGED)
            if marr is None:
                marr = np.zeros((h, w), dtype=np.uint8)
            if marr.ndim == 3:
                marr = cv2.cvtColor(marr, cv2.COLOR_BGR2GRAY)
        else:
            marr = np.asarray(tiff.imread(mask_fp))
            if marr.ndim != 2:
                # take first plane if needed
                marr = marr[0] if marr.ndim > 2 else marr
        # normalize and resize to target
        marr = _to_uint8(marr)
        if marr.shape != (h, w):
            marr = cv2.resize(marr, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_slices.append((marr > 0).astype(np.uint8))

    return np.stack(mask_slices, axis=0)


def build_mask_path_mapping(image_files, mask_base_dir=None):
    if not image_files:
        return []

    mask_paths = []
    for img_fp in image_files:
        base, ext = os.path.splitext(os.path.basename(img_fp))
        if mask_base_dir is None:
            search_dir = os.path.dirname(img_fp)
        else:
            search_dir = mask_base_dir

        try_exts = [ext.lower()]
        for e in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
            if e not in try_exts:
                try_exts.append(e)

        mask_fp = None
        for e in try_exts:
            candidate = os.path.join(search_dir, f"{base}_mask{e}")
            if os.path.exists(candidate):
                mask_fp = candidate
                break

        if mask_fp is None:
            for e in try_exts:
                candidate = os.path.join(search_dir, f"{base}_prediction{e}")
                if os.path.exists(candidate):
                    mask_fp = candidate
                    break

        if mask_fp is None and base.endswith("_0000"):
            trimmed = base[:-5]  # remove suffix "_0000"
            for e in try_exts:
                candidate = os.path.join(search_dir, f"{trimmed}{e}")
                if os.path.exists(candidate):
                    mask_fp = candidate
                    break

        mask_paths.append(mask_fp)

    return mask_paths



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
