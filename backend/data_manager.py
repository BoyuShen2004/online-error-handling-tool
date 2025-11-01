import os
import numpy as np
import cv2
import tifffile
from PIL import Image
from typing import Tuple, Optional, List, Dict, Any, Union
import io
import base64

class DataManager:
    """
    Handles loading, processing, and managing image and mask data.
    Supports both 2D and 3D datasets.
    """

    def __init__(self):
        self.current_volume = None
        self.current_mask = None
        self.volume_info = {}
        self.mask_info = {}

    def load_image(self, image_path: Union[str, List[str]]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load image data from file path, directory/glob, or list of files.
        Returns: (image_array, metadata_dict)
        """
        # If list of files is provided, stack them
        if isinstance(image_path, list):
            from backend.volume_manager import _to_uint8  # reuse normalization
            import cv2
            if not image_path:
                raise FileNotFoundError("No files provided for loading")
            files = list(image_path)
            slices = []
            target_shape = None
            for fp in files:
                ext = os.path.splitext(fp.lower())[1]
                if ext in ['.tif', '.tiff']:
                    arr = tifffile.imread(fp)
                    if arr.ndim != 2:
                        raise ValueError(f"Expected 2D TIFF for stacking, got {arr.shape}")
                else:
                    img = Image.open(fp)
                    arr = np.array(img)
                    if arr.ndim == 3:
                        if arr.shape[2] == 3:
                            arr = np.mean(arr, axis=2).astype(arr.dtype)
                        elif arr.shape[2] == 4:
                            arr = np.mean(arr[:, :, :3], axis=2).astype(arr.dtype)
                        else:
                            arr = arr[:, :, 0]
                    elif arr.ndim != 2:
                        raise ValueError(f"Unsupported image dims {arr.ndim} for {fp}")
                arr = _to_uint8(arr)
                if target_shape is None:
                    target_shape = arr.shape
                elif arr.shape != target_shape:
                    arr = cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
                slices.append(arr)
            volume = np.stack(slices, axis=0)
            info = {
                "shape": volume.shape,
                "dtype": str(volume.dtype),
                "ndim": volume.ndim,
                "is_3d": True,
                "num_slices": volume.shape[0]
            }
            self.current_volume = volume
            self.volume_info = info
            return volume, info

        # Allow directory or glob pattern
        if any(ch in str(image_path) for ch in ['*', '?', '[']) or os.path.isdir(image_path):
            from backend.volume_manager import load_image_or_stack
            volume = load_image_or_stack(image_path)
            info = {
                "shape": volume.shape,
                "dtype": str(volume.dtype),
                "ndim": volume.ndim,
                "is_3d": volume.ndim == 3,
                "num_slices": volume.shape[0] if volume.ndim == 3 else 1
            }
            self.current_volume = volume
            self.volume_info = info
            return volume, info

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        ext = os.path.splitext(image_path.lower())[1]
        
        if ext in ['.tif', '.tiff']:
            # Load TIFF stack
            volume = tifffile.imread(image_path)
            info = {
                "shape": volume.shape,
                "dtype": str(volume.dtype),
                "ndim": volume.ndim,
                "is_3d": volume.ndim == 3,
                "num_slices": volume.shape[0] if volume.ndim == 3 else 1
            }
        else:
            # Load 2D image
            img = Image.open(image_path)
            volume = np.array(img)
            
            # Handle different image formats properly
            if volume.ndim == 3:
                # If it's a color image, convert to grayscale
                if volume.shape[2] == 3:  # RGB
                    volume = np.mean(volume, axis=2).astype(volume.dtype)
                elif volume.shape[2] == 4:  # RGBA
                    volume = np.mean(volume[:, :, :3], axis=2).astype(volume.dtype)
                else:
                    # Other multi-channel formats, take first channel
                    volume = volume[:, :, 0]
            elif volume.ndim == 2:
                # Already grayscale, keep as is
                pass
            else:
                raise ValueError(f"Unsupported image format with {volume.ndim} dimensions")
            
            info = {
                "shape": volume.shape,
                "dtype": str(volume.dtype),
                "ndim": volume.ndim,
                "is_3d": False,
                "num_slices": 1
            }

        self.current_volume = volume
        self.volume_info = info
        return volume, info

    def load_mask(self, mask_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load mask data from file path.
        Returns: (mask_array, metadata_dict)
        """
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        ext = os.path.splitext(mask_path.lower())[1]
        
        if ext in ['.tif', '.tiff']:
            # Load TIFF stack
            mask = tifffile.imread(mask_path)
            info = {
                "shape": mask.shape,
                "dtype": str(mask.dtype),
                "ndim": mask.ndim,
                "is_3d": mask.ndim == 3,
                "num_slices": mask.shape[0] if mask.ndim == 3 else 1
            }
        else:
            # Load 2D image
            img = Image.open(mask_path)
            mask = np.array(img)
            
            # Handle different image formats properly
            if mask.ndim == 3:
                # If it's a color image, convert to grayscale
                if mask.shape[2] == 3:  # RGB
                    mask = np.mean(mask, axis=2).astype(mask.dtype)
                elif mask.shape[2] == 4:  # RGBA
                    mask = np.mean(mask[:, :, :3], axis=2).astype(mask.dtype)
                else:
                    # Other multi-channel formats, take first channel
                    mask = mask[:, :, 0]
            elif mask.ndim == 2:
                # Already grayscale, keep as is
                pass
            else:
                raise ValueError(f"Unsupported mask format with {mask.ndim} dimensions")
            
            info = {
                "shape": mask.shape,
                "dtype": str(mask.dtype),
                "ndim": mask.ndim,
                "is_3d": False,
                "num_slices": 1
            }

        self.current_mask = mask
        self.mask_info = info
        return mask, info

    def get_slice(self, z: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get a specific slice from the volume and mask.
        Returns: (image_slice, mask_slice)
        """
        if self.current_volume is None:
            raise ValueError("No volume loaded")

        if self.current_volume.ndim == 2:
            image_slice = self.current_volume
            mask_slice = self.current_mask if self.current_mask is not None else None
        else:
            z = max(0, min(z, self.current_volume.shape[0] - 1))
            image_slice = self.current_volume[z]
            mask_slice = self.current_mask[z] if self.current_mask is not None else None

        return image_slice, mask_slice

    def create_overlay(self, image_slice: np.ndarray, mask_slice: Optional[np.ndarray], 
                      alpha: float = 0.4) -> np.ndarray:
        """
        Create an overlay of image and mask.
        Returns: RGB overlay image
        """
        # Ensure image_slice is 2D
        if image_slice.ndim != 2:
            if image_slice.ndim == 3:
                # If 3D, take the first slice or convert to grayscale
                if image_slice.shape[2] == 1:
                    image_slice = image_slice[:, :, 0]
                else:
                    image_slice = np.mean(image_slice, axis=2)
            else:
                raise ValueError(f"Unsupported image slice dimensions: {image_slice.ndim}")
        
        # Normalize image to 0-255
        if image_slice.max() > 1:
            img_norm = image_slice.astype(np.float32)
        else:
            img_norm = (image_slice * 255).astype(np.float32)
        
        img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)
        
        # Convert to RGB
        img_rgb = np.stack([img_norm] * 3, axis=-1)

        if mask_slice is not None:
            # Ensure mask_slice is 2D and has same dimensions as image
            if mask_slice.ndim != 2:
                if mask_slice.ndim == 3:
                    if mask_slice.shape[2] == 1:
                        mask_slice = mask_slice[:, :, 0]
                    else:
                        mask_slice = np.mean(mask_slice, axis=2)
                else:
                    raise ValueError(f"Unsupported mask slice dimensions: {mask_slice.ndim}")
            
            # Ensure mask has same dimensions as image
            if mask_slice.shape != image_slice.shape:
                # Resize mask to match image dimensions
                mask_slice = cv2.resize(mask_slice, (image_slice.shape[1], image_slice.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            
            # Create colored mask overlay
            mask_binary = (mask_slice > 0).astype(np.uint8)
            
            # Create colored overlay (red for mask)
            overlay = img_rgb.copy()
            
            # Apply mask overlay channel by channel
            mask_indices = mask_binary > 0
            if np.any(mask_indices):
                # Red channel: increase red
                overlay[mask_indices, 0] = np.minimum(255, overlay[mask_indices, 0] + 100)
                # Green channel: decrease green
                overlay[mask_indices, 1] = np.maximum(0, overlay[mask_indices, 1] - 50)
                # Blue channel: decrease blue
                overlay[mask_indices, 2] = np.maximum(0, overlay[mask_indices, 2] - 50)
            
            # Blend with original image
            img_rgb = cv2.addWeighted(img_rgb, 1-alpha, overlay, alpha, 0)

        return img_rgb

    def array_to_base64(self, arr: np.ndarray, format: str = "PNG") -> str:
        """Convert numpy array to base64 string."""
        if len(arr.shape) == 2:
            img = Image.fromarray(arr)
        else:
            img = Image.fromarray(arr)
        
        bio = io.BytesIO()
        img.save(bio, format=format)
        bio.seek(0)
        return base64.b64encode(bio.getvalue()).decode()

    def create_layer_data(self, z: int, image_slice: np.ndarray, 
                         mask_slice: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Create layer data for a specific slice.
        Returns: layer data dictionary
        """
        overlay = self.create_overlay(image_slice, mask_slice)
        
        return {
            "z": z,
            "image_slice": self.array_to_base64(image_slice),
            "mask_slice": self.array_to_base64(mask_slice) if mask_slice is not None else None,
            "overlay": self.array_to_base64(overlay),
            "has_mask": mask_slice is not None,
            "mask_coverage": float(np.sum(mask_slice > 0) / mask_slice.size) if mask_slice is not None else 0.0
        }

    def generate_layer_for_z(self, z: int) -> Dict[str, Any]:
        """
        Generate full layer data for a specific z index using current volume/mask.
        """
        if self.current_volume is None:
            raise ValueError("No volume loaded")
        image_slice, mask_slice = self.get_slice(z)
        return self.create_layer_data(z, image_slice, mask_slice)

    def generate_layers_range(self, start_z: int, end_z: int) -> List[Dict[str, Any]]:
        """
        Generate layer data for z in [start_z, end_z) (end exclusive).
        """
        if self.current_volume is None:
            raise ValueError("No volume loaded")
        if self.current_volume.ndim == 2:
            start_z, end_z = 0, 1
        end_z = min(end_z, self.current_volume.shape[0] if self.current_volume.ndim == 3 else 1)
        start_z = max(0, start_z)
        layers: List[Dict[str, Any]] = []
        for z in range(start_z, end_z):
            layers.append(self.generate_layer_for_z(z))
        return layers

    def generate_all_layers(self) -> List[Dict[str, Any]]:
        """
        Generate layer data for all slices.
        Returns: list of layer data dictionaries
        """
        if self.current_volume is None:
            raise ValueError("No volume loaded")

        layers = []
        
        if self.current_volume.ndim == 2:
            # 2D case
            image_slice, mask_slice = self.get_slice(0)
            layer_data = self.create_layer_data(0, image_slice, mask_slice)
            layers.append(layer_data)
        else:
            # 3D case
            for z in range(self.current_volume.shape[0]):
                image_slice, mask_slice = self.get_slice(z)
                layer_data = self.create_layer_data(z, image_slice, mask_slice)
                layers.append(layer_data)

        return layers

    def validate_data_compatibility(self) -> bool:
        """
        Validate that image and mask data are compatible.
        Returns: True if compatible, False otherwise
        """
        if self.current_volume is None or self.current_mask is None:
            return True  # No mask is valid

        # Check dimensions
        if self.current_volume.ndim != self.current_mask.ndim:
            return False

        if self.current_volume.ndim == 2:
            return self.current_volume.shape == self.current_mask.shape
        else:
            return self.current_volume.shape == self.current_mask.shape

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data."""
        summary = {
            "volume_loaded": self.current_volume is not None,
            "mask_loaded": self.current_mask is not None,
            "compatible": self.validate_data_compatibility()
        }
        
        if self.current_volume is not None:
            summary.update({
                "volume_shape": self.current_volume.shape,
                "volume_dtype": str(self.current_volume.dtype),
                "is_3d": self.current_volume.ndim == 3,
                "num_slices": self.current_volume.shape[0] if self.current_volume.ndim == 3 else 1
            })
        
        if self.current_mask is not None:
            summary.update({
                "mask_shape": self.current_mask.shape,
                "mask_dtype": str(self.current_mask.dtype),
                "mask_is_3d": self.current_mask.ndim == 3
            })

        return summary
