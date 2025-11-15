"""
Error Handling Tool - Proofreading Routes
----------------------------------------
Handles the integrated proofreading interface for correcting incorrect layers.
"""

import os
import io
import base64
import numpy as np
from flask import Blueprint, render_template, request, current_app, jsonify, send_file, redirect, url_for
from PIL import Image
from backend.volume_manager import list_images_for_path, load_image_or_stack, load_mask_like, save_mask, stack_2d_images
def _cached_detection_image_files():
    files = current_app.config.get("DETECTION_IMAGE_FILES")
    if files:
        return files
    session_manager = current_app.session_manager
    session_state = session_manager.snapshot()
    image_path = session_state.get("image_path", "")
    files = list_images_for_path(image_path)
    current_app.config["DETECTION_IMAGE_FILES"] = files
    return files


def _resolve_detection_mask(img_fp, idx):
    mask_files = current_app.config.get("DETECTION_MASK_FILES")
    if mask_files and idx < len(mask_files):
        mask_fp = mask_files[idx]
        if mask_fp:
            return mask_fp

    def build_exts(ext):
        exts = [ext.lower()]
        for e in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
            if e not in exts:
                exts.append(e)
        return exts

    def try_dir(dir_path, base_name, extensions):
        if not dir_path or not os.path.isdir(dir_path):
            return None
        for e in extensions:
            cand = os.path.join(dir_path, f"{base_name}_mask{e}")
            if os.path.exists(cand):
                return cand
        for e in extensions:
            cand = os.path.join(dir_path, f"{base_name}_prediction{e}")
            if os.path.exists(cand):
                return cand
        if base_name.endswith("_0000"):
            trimmed = base_name[:-5]
            for e in extensions:
                cand = os.path.join(dir_path, f"{trimmed}{e}")
                if os.path.exists(cand):
                    return cand
        return None

    if not img_fp:
        return None

    base, ext = os.path.splitext(os.path.basename(img_fp))
    extensions = build_exts(ext)
    session_state = current_app.session_manager.snapshot()
    mask_path = session_state.get("mask_path", "")
    search_dirs = []
    if mask_path and os.path.isdir(mask_path):
        search_dirs.append(mask_path)
    img_dir = os.path.dirname(img_fp)
    if img_dir:
        search_dirs.append(img_dir)

    for d in search_dirs:
        result = try_dir(d, base, extensions)
        if result:
            return result

    if mask_path and os.path.isfile(mask_path):
        return mask_path
    return None
from backend.utils import jsonify_dimensions

bp = Blueprint("proofreading", __name__, url_prefix="")

def register_proofreading_routes(app):
    app.register_blueprint(bp)

@bp.route("/proofreading")
def proofreading():
    """Layer selection page for incorrect layers."""
    session_manager = current_app.session_manager
    session_state = session_manager.snapshot()
    
    if not session_state.get("layers"):
        return redirect(url_for("landing.landing"))
    
    # Get incorrect layers for proofreading
    incorrect_layers = session_manager.get_incorrect_layers()
    
    if not incorrect_layers:
        all_layers = session_state.get("layers", [])
        return render_template(
            "proofreading_selection.html",
            layers=all_layers,  # Pass all layers for navigation
            incorrect_layers=[],  # No incorrect layers
            progress=session_manager.get_progress_stats(),
            mode3d=session_state.get("mode3d", False),
            image_path=session_state.get("image_path", ""),
            mask_path=session_state.get("mask_path", ""),
            warning="No incorrect layers found for proofreading."
        )
    
    # Pass all layers to ensure navigation is visible
    all_layers = session_state.get("layers", [])
    
    return render_template(
        "proofreading_selection.html",
        layers=all_layers,  # Pass all layers for navigation
        incorrect_layers=incorrect_layers,  # Pass incorrect layers for selection
        progress=session_manager.get_progress_stats(),
        mode3d=session_state.get("mode3d", False),
        image_path=session_state.get("image_path", ""),
        mask_path=session_state.get("mask_path", "")
    )

@bp.route("/proofreading/edit/<layer_id>")
def proofreading_edit(layer_id):
    """Proofreading editor for a specific incorrect layer."""
    session_manager = current_app.session_manager
    session_state = session_manager.snapshot()
    
    if not session_state.get("layers"):
        return redirect(url_for("landing.landing"))
    
    # Get all incorrect layers
    incorrect_layers = session_manager.get_incorrect_layers()
    
    # Find the specific layer
    current_layer = None
    layer_index = -1
    for i, layer in enumerate(incorrect_layers):
        if layer["id"] == layer_id:
            current_layer = layer
            layer_index = i
            break
    
    if not current_layer:
        return redirect(url_for("proofreading.proofreading"))
    
    # Load volume and mask for proofreading
    try:
        # Try to get data from detection workflow first
        data_manager = current_app.config.get("DETECTION_DATA_MANAGER")
        volume = current_app.config.get("DETECTION_VOLUME")
        mask = current_app.config.get("DETECTION_MASK")
        
        # Get the specific slice for this incorrect layer
        slice_idx = current_layer.get("z", 0)
        print(f"Loading incorrect layer {current_layer['id']} at slice index {slice_idx}")
        
        image_slice = None
        mask_slice = None
        if data_manager is not None:
            image_slice, mask_slice = data_manager.get_slice(slice_idx)
        else:
            # Fallback to eager loading when data manager is unavailable
            if volume is None:
                ipath = session_state.get("image_path", "")
                volume = stack_2d_images(ipath) if isinstance(ipath, list) else load_image_or_stack(ipath)
            if mask is None:
                mask = load_mask_like(session_state.get("mask_path"), volume)

            if volume.ndim == 3:
                if slice_idx >= volume.shape[0]:
                    raise ValueError(f"Slice index {slice_idx} out of range for volume with {volume.shape[0]} slices")
                image_slice = volume[slice_idx]
                mask_slice = mask[slice_idx] if mask is not None and getattr(mask, 'ndim', 0) == 3 else mask
            else:
                image_slice = volume
                mask_slice = mask
        
        print(f"Image slice shape: {image_slice.shape}")
        print(f"Mask slice shape: {mask_slice.shape if mask_slice is not None else 'None'}")
        print(f"Mask slice type: {type(mask_slice)}")
        
        # Store only the specific slice in app config
        current_app.config["INTEGRATED_VOLUME"] = image_slice
        current_app.config["INTEGRATED_MASK"] = mask_slice
        current_app.config["CURRENT_SLICE_INDEX"] = slice_idx
        current_app.config["CURRENT_LAYER_ID"] = layer_id
        
        # For 2D mode, we only have 1 slice
        num_slices = 1
        
        # Pass all layers to ensure navigation is visible
        all_layers = session_state.get("layers", [])
        
        return render_template(
            "proofreading.html",
            layers=all_layers,  # Pass all layers for navigation
            current_layer=current_layer,
            layer_index=layer_index,
            total_incorrect=len(incorrect_layers),
            incorrect_layers=incorrect_layers,
            progress=session_manager.get_progress_stats(),
            mode3d=False,  # Always treat as 2D for incorrect layer editing
            image_path=session_state.get("image_path", ""),
            mask_path=session_state.get("mask_path", ""),
            num_slices=1,  # Only editing one slice
            volume_shape=image_slice.shape,
            mask_shape=mask_slice.shape if mask_slice is not None else None,
            slice_index=slice_idx
        )
        
    except Exception as e:
        # Pass all layers to ensure navigation is visible
        all_layers = session_state.get("layers", [])
        
        return render_template(
            "proofreading.html",
            layers=all_layers,  # Pass all layers for navigation
            current_layer=current_layer,
            layer_index=layer_index,
            total_incorrect=len(incorrect_layers),
            incorrect_layers=incorrect_layers,
            progress=session_manager.get_progress_stats(),
            mode3d=session_state.get("mode3d", False),
            image_path=session_state.get("image_path", ""),
            mask_path=session_state.get("mask_path", ""),
            warning=f"Error loading data for proofreading: {e}"
        )

@bp.route("/api/proofreading_layer/<layer_id>")
def api_proofreading_layer(layer_id):
    """Get layer data for proofreading interface."""
    try:
        session_manager = current_app.session_manager
        layers = session_manager.get("layers", [])
        
        # Find the layer
        layer = None
        for l in layers:
            if l["id"] == layer_id:
                layer = l
                break
        
        if not layer:
            return jsonify({"error": "Layer not found"}), 404
        
        # Get volume and mask
        volume = current_app.config.get("INTEGRATED_VOLUME")
        mask = current_app.config.get("INTEGRATED_MASK")
        
        if volume is None or mask is None:
            return jsonify({"error": "Volume or mask not loaded"}), 400
        
        # Get layer slice
        if volume.ndim == 3:
            slice_idx = layer.get("slice_index", 0)
            image_slice = volume[slice_idx]
            mask_slice = mask[slice_idx] if mask.ndim == 3 else mask
        else:
            image_slice = volume
            mask_slice = mask
        
        # Convert to base64 for display
        image_pil = Image.fromarray(image_slice)
        mask_pil = Image.fromarray(mask_slice * 255)
        
        # Create overlay
        overlay = Image.blend(image_pil.convert('RGB'), mask_pil.convert('RGB'), 0.3)
        
        # Convert to base64
        def image_to_base64(img):
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            "layer": layer,
            "image_base64": image_to_base64(image_pil),
            "mask_base64": image_to_base64(mask_pil),
            "overlay_base64": image_to_base64(overlay),
            "slice_index": layer.get("slice_index", 0),
            "total_slices": volume.shape[0] if volume.ndim == 3 else 1
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/api/save_proofreading", methods=["POST"])
def api_save_proofreading():
    """Save proofreading changes for a layer."""
    try:
        data = request.get_json()
        layer_id = data.get("layer_id")
        mask_data = data.get("mask_data")  # Base64 encoded mask
        
        if not layer_id or not mask_data:
            return jsonify({"success": False, "error": "Missing layer_id or mask_data"}), 400
        
        # Decode mask data
        mask_bytes = base64.b64decode(mask_data.split(',')[1])
        mask_pil = Image.open(io.BytesIO(mask_bytes))
        mask_array = np.array(mask_pil.convert('L')) > 128
        
        # Get current volume and mask
        volume = current_app.config.get("INTEGRATED_VOLUME")
        mask = current_app.config.get("INTEGRATED_MASK")
        
        if volume is None or mask is None:
            return jsonify({"success": False, "error": "Volume or mask not loaded"}), 400
        
        # Update mask for this layer
        session_manager = current_app.session_manager
        layers = session_manager.get("layers", [])
        
        for layer in layers:
            if layer["id"] == layer_id:
                slice_idx = layer.get("slice_index", 0)
                if volume.ndim == 3:
                    mask[slice_idx] = mask_array.astype(np.uint8)
                else:
                    mask[:] = mask_array.astype(np.uint8)
                break
        
        # Update mask in app config
        current_app.config["INTEGRATED_MASK"] = mask
        
        # Save mask to file
        session_state = session_manager.snapshot()
        mask_path = session_state.get("mask_path", "")
        if mask_path:
            save_mask(mask, mask_path)
        
        return jsonify({"success": True})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/mark_corrected", methods=["POST"])
def api_mark_corrected():
    """Mark a layer as corrected after proofreading."""
    try:
        data = request.get_json()
        layer_id = data.get("layer_id")
        
        if not layer_id:
            return jsonify({"success": False, "error": "Missing layer_id"}), 400
        
        # Update layer status to correct
        session_manager = current_app.session_manager
        session_manager.update_layer_status(layer_id, "correct", {"proofread": True})
        
        return jsonify({"success": True})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/slice/<int:slice_idx>")
def api_slice(slice_idx):
    """Get image slice for proofreading."""
    try:
        # For incorrect layer editing, we only have one slice
        volume = current_app.config.get("INTEGRATED_VOLUME")
        if volume is None:
            return jsonify({"error": "Volume not loaded"}), 400
        
        # Since we're editing a specific incorrect layer, we only have one slice
        if slice_idx != 0:
            return jsonify({"error": "Only slice 0 available for incorrect layer editing"}), 400
        
        # Convert to PIL Image and return as PNG
        from PIL import Image
        img = Image.fromarray(volume)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return send_file(img_bytes, mimetype='image/png')
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/api/mask/<int:slice_idx>")
def api_mask(slice_idx):
    """Get mask slice for proofreading."""
    try:
        # For incorrect layer editing, we only have one slice
        mask = current_app.config.get("INTEGRATED_MASK")
        if mask is None:
            return jsonify({"error": "Mask not loaded"}), 400
        
        # Since we're editing a specific incorrect layer, we only have one slice
        if slice_idx != 0:
            return jsonify({"error": "Only slice 0 available for incorrect layer editing"}), 400
        
        # Convert to PIL Image and return as PNG
        from PIL import Image
        img = Image.fromarray(mask * 255)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return send_file(img_bytes, mimetype='image/png')
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/api/names_current")
def api_names_current():
    """Return the current image filename and paired mask filename for the incorrect-layer proofreading view."""
    try:
        slice_idx = current_app.config.get("CURRENT_SLICE_INDEX", 0)

        images = _cached_detection_image_files()
        if not images:
            session_state = current_app.session_manager.snapshot()
            image_path = session_state.get("image_path", "")
            img_fp = image_path if isinstance(image_path, str) else None
        else:
            img_fp = images[slice_idx] if slice_idx < len(images) else images[-1]

        img_name = os.path.basename(img_fp) if img_fp else None
        mask_fp = _resolve_detection_mask(img_fp, slice_idx)
        mask_name = os.path.basename(mask_fp) if mask_fp else None

        return jsonify(image=img_name, mask=mask_name)
    except Exception as e:
        return jsonify(error=str(e)), 500

@bp.route("/api/mask/update", methods=["POST"])
def api_mask_update():
    """Update mask with edited slices."""
    try:
        data = request.get_json()
        batch = data.get("full_batch", [])
        
        if not batch:
            return jsonify({"success": True, "message": "No changes to save"})
        
        # Get current mask (single slice for incorrect layer editing)
        mask = current_app.config.get("INTEGRATED_MASK")
        if mask is None:
            return jsonify({"success": False, "error": "Mask not loaded"}), 400
        
        # Update mask with edited slice (only slice 0 for incorrect layer editing)
        for item in batch:
            slice_idx = item.get("z", 0)
            png_data = item.get("png", "")
            
            if png_data and slice_idx == 0:  # Only allow slice 0 for incorrect layer editing
                # Decode base64 PNG data
                import base64
                img_data = base64.b64decode(png_data)
                img = Image.open(io.BytesIO(img_data))
                img_array = np.array(img.convert('L')) > 128
                
                # Update the single slice mask
                mask[:] = img_array.astype(np.uint8)
        
        # Update mask in app config
        current_app.config["INTEGRATED_MASK"] = mask
        
        return jsonify({"success": True})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/save", methods=["POST"])
def api_save():
    """Save mask to file."""
    try:
        mask = current_app.config.get("INTEGRATED_MASK")
        volume = current_app.config.get("INTEGRATED_VOLUME")
        slice_idx = current_app.config.get("CURRENT_SLICE_INDEX", 0)
        
        if mask is None and volume is not None:
            if volume.ndim == 2:
                mask = np.zeros_like(volume, dtype=np.uint8)
            elif volume.ndim == 3:
                mask = np.zeros_like(volume, dtype=np.uint8)
            current_app.config["INTEGRATED_MASK"] = mask
        elif mask is None:
            return jsonify({"success": False, "error": "No mask or image loaded"}), 400
        
        # Get session data
        session_manager = current_app.session_manager
        session_state = session_manager.snapshot()
        mask_path = session_state.get("mask_path", "")
        image_path = session_state.get("image_path", "")
        load_mode = session_state.get("load_mode", "path")

        # If working from a folder/glob/multi-file
        src_is_dir = bool(image_path and isinstance(image_path, str) and os.path.isdir(image_path))
        src_is_glob = bool(image_path and isinstance(image_path, str) and any(ch in image_path for ch in ['*','?','[']) and not os.path.exists(image_path))
        src_is_list = isinstance(image_path, list)
        # Case A: user provided a mask folder → save current slice into that folder
        if (src_is_dir or src_is_glob or src_is_list) and mask_path and os.path.isdir(mask_path):
            mask_dir = mask_path
            os.makedirs(mask_dir, exist_ok=True)

            # Determine source filename for current slice
            src_fp = None
            try:
                if src_is_list:
                    src_fp = image_path[slice_idx] if slice_idx < len(image_path) else image_path[-1]
                elif src_is_dir:
                    candidates = []
                    for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                        candidates = sorted([
                            os.path.join(image_path, f)
                            for f in os.listdir(image_path)
                            if f.lower().endswith(ext)
                        ])
                        if candidates:
                            break
                    if candidates:
                        src_fp = candidates[slice_idx] if slice_idx < len(candidates) else candidates[-1]
                else:
                    import glob as _glob
                    files = sorted(_glob.glob(image_path))
                    if files:
                        src_fp = files[slice_idx] if slice_idx < len(files) else files[-1]
            except Exception:
                src_fp = None

            if src_fp is None:
                out_fp = os.path.join(mask_dir, f"slice_{slice_idx:04d}_mask.tif")
            else:
                base = os.path.splitext(os.path.basename(src_fp))[0]
                ext_out = os.path.splitext(src_fp)[-1].lower()
                out_fp = os.path.join(mask_dir, f"{base}_mask{ext_out}")

            save_mask(mask, out_fp)
            return jsonify({"success": True, "message": f"Mask slice saved to {out_fp}"})

        # Case B: no mask folder → create sibling folder and save current slice
        if not mask_path and (src_is_dir or src_is_glob or src_is_list):
            # Determine mask directory next to uploaded folder
            if src_is_list:
                first_dir = os.path.dirname(image_path[0]) if image_path else os.path.abspath(".")
                parent = os.path.dirname(first_dir)
                dir_name = os.path.basename(first_dir)
                mask_dir = os.path.join(parent, f"{dir_name}_mask")
            elif src_is_dir:
                parent = os.path.dirname(image_path)
                dir_name = os.path.basename(image_path)
                mask_dir = os.path.join(parent, f"{dir_name}_mask")
            else:
                base_dir = os.path.dirname(image_path)
                dir_name = os.path.basename(base_dir)
                parent = os.path.dirname(base_dir)
                mask_dir = os.path.join(parent, f"{dir_name}_mask")
            os.makedirs(mask_dir, exist_ok=True)

            # Determine source filename for current slice if possible
            src_fp = None
            try:
                if src_is_list:
                    src_fp = image_path[slice_idx] if slice_idx < len(image_path) else image_path[-1]
                elif src_is_dir:
                    candidates = []
                    for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                        candidates = sorted([
                            os.path.join(image_path, f)
                            for f in os.listdir(image_path)
                            if f.lower().endswith(ext)
                        ])
                        if candidates:
                            break
                    if candidates:
                        src_fp = candidates[slice_idx] if slice_idx < len(candidates) else candidates[-1]
                else:
                    import glob as _glob
                    files = sorted(_glob.glob(image_path))
                    if files:
                        src_fp = files[slice_idx] if slice_idx < len(files) else files[-1]
            except Exception:
                src_fp = None

            if src_fp is None:
                # Fallback generic name
                out_fp = os.path.join(mask_dir, f"slice_{slice_idx:04d}_mask.tif")
            else:
                base = os.path.splitext(os.path.basename(src_fp))[0]
                ext_out = os.path.splitext(src_fp)[-1].lower()
                out_fp = os.path.join(mask_dir, f"{base}_mask{ext_out}")

            save_mask(mask, out_fp)
            session_manager.update(mask_path=mask_dir)
            return jsonify({"success": True, "message": f"Mask slice saved to {out_fp}"})

        # Generate mask path if not provided (file-based)
        if not mask_path:
            if load_mode == "upload" or not image_path or not os.path.exists(image_path):
                base_dir = os.path.abspath("./_uploads")
                os.makedirs(base_dir, exist_ok=True)
                image_name = session_state.get("image_name", "image")
                if not image_name or image_name == "image":
                    image_name = "image"
                original_base = os.path.splitext(os.path.basename(image_name))[0]
            else:
                base_dir = os.path.dirname(image_path)
                original_base = os.path.splitext(os.path.basename(image_path))[0]

            # Detect extension
            ext = ".tif"
            if isinstance(image_path, str) and image_path:
                _, src_ext = os.path.splitext(image_path.lower())
                if src_ext in [".png", ".jpg", ".jpeg"]:
                    ext = src_ext

            mask_path = os.path.join(base_dir, f"{original_base}_mask{ext}")
            session_manager.update(mask_path=mask_path)
        
        try:
            # Load the full volume and mask to update the specific slice
            full_volume = load_image_or_stack(image_path)
            full_mask = load_mask_like(mask_path, full_volume)
            
            if full_mask is not None:
                # Update the specific slice in the full mask
                if full_mask.ndim == 3 and slice_idx < full_mask.shape[0]:
                    full_mask[slice_idx] = mask
                elif full_mask.ndim == 2:
                    full_mask[:] = mask
                
                # Save the updated full mask
                save_mask(full_mask, mask_path)
                return jsonify({"success": True, "message": f"Mask slice {slice_idx} saved to {mask_path}"})
            else:
                # If no existing mask, create one with the edited slice
                save_mask(mask, mask_path)
                return jsonify({"success": True, "message": f"New mask saved to {mask_path}"})
                
        except Exception as e:
            print(f"Save error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({"success": False, "error": f"Failed to save mask: {str(e)}"}), 500
        
    except Exception as e:
        print(f"Save error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": f"Failed to save mask: {str(e)}"}), 500

@bp.route("/api/dims", methods=["POST"])
def api_dims():
    """Get dimensions of uploaded file."""
    return jsonify_dimensions(request.files.get("file"), use_temp_file=False)