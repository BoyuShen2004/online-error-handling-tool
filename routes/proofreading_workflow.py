"""
Error Handling Tool - Standalone Proofreading Workflow Routes
-----------------------------------------------------------
Handles the standalone proofreading workflow (similar to PFTool).
"""

import os
import io
import base64
import numpy as np
import cv2
from flask import Blueprint, render_template, request, current_app, jsonify, send_file, redirect, url_for
from PIL import Image
from backend.volume_manager import load_image_or_stack, load_mask_like, save_mask, list_images_for_path, build_mask_stack_from_pairs, stack_2d_images
from backend.utils import jsonify_dimensions

bp = Blueprint("proofreading_workflow", __name__, url_prefix="/standalone_proofreading")

def register_proofreading_workflow_routes(app):
    app.register_blueprint(bp)

@bp.route("/load")
def proofreading_load():
    """Load dataset for standalone proofreading workflow."""
    # Check if there's existing data
    volume = current_app.config.get("PROOFREADING_VOLUME")
    mask = current_app.config.get("PROOFREADING_MASK")
    image_path = current_app.config.get("PROOFREADING_IMAGE_PATH")
    mask_path = current_app.config.get("PROOFREADING_MASK_PATH")
    
    has_existing_data = volume is not None and image_path is not None
    
    if has_existing_data:
        return render_template("proofreading_load.html",
                             has_existing_data=True,
                             existing_image_path=os.path.basename(image_path),
                             existing_mask_path=os.path.basename(mask_path) if mask_path else None,
                             existing_shape=" × ".join(map(str, volume.shape)),
                             existing_mode3d=volume.ndim == 3)
    else:
        return render_template("proofreading_load.html")

@bp.route("/clear", methods=["POST"])
def clear_data():
    """Clear existing data."""
    try:
        # Clear app config
        current_app.config.pop("PROOFREADING_VOLUME", None)
        current_app.config.pop("PROOFREADING_MASK", None)
        current_app.config.pop("PROOFREADING_IMAGE_PATH", None)
        current_app.config.pop("PROOFREADING_MASK_PATH", None)
        current_app.config.pop("PROOFREADING_EDITED_SLICES", None)
        
        # Clear session
        session_manager = current_app.session_manager
        session_manager.reset_session()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/load", methods=["POST"])
def proofreading_load_post():
    """Handle dataset loading for standalone proofreading."""
    try:
        # Clear previously loaded data before loading new data
        session_manager = current_app.session_manager
        session_manager.reset_session()
        current_app.config.pop("PROOFREADING_VOLUME", None)
        current_app.config.pop("PROOFREADING_MASK", None)
        current_app.config.pop("PROOFREADING_IMAGE_PATH", None)
        current_app.config.pop("PROOFREADING_MASK_PATH", None)
        
        # Get form data
        load_mode = request.form.get("load_mode", "path")
        image_path = request.form.get("image_path", "").strip()
        mask_path = request.form.get("mask_path", "").strip()
        
        if load_mode == "path":
            if not image_path:
                return render_template("proofreading_load.html", 
                                    error="Image path is required",
                                    image_path=image_path,
                                    mask_path=mask_path)
            
            if not os.path.exists(image_path):
                return render_template("proofreading_load.html", 
                                    error="Image file not found",
                                    image_path=image_path,
                                    mask_path=mask_path)
            
            if mask_path and not os.path.exists(mask_path):
                return render_template("proofreading_load.html", 
                                    error="Mask file not found",
                                    image_path=image_path,
                                    mask_path=mask_path)
        
        else:  # upload mode
            # Allow multiple image files to form a stack
            image_files = request.files.getlist("image_file")
            mask_file = request.files.get("mask_file")

            image_files = [f for f in image_files if f and f.filename]
            if not image_files:
                return render_template("proofreading_load.html", 
                                    error="At least one image file is required")

            # Save uploaded files temporarily
            upload_dir = "_uploads"
            os.makedirs(upload_dir, exist_ok=True)

            saved_image_paths = []
            for f in image_files:
                dst = os.path.join(upload_dir, f.filename)
                f.save(dst)
                saved_image_paths.append(dst)

            image_path = saved_image_paths[0] if len(saved_image_paths) == 1 else saved_image_paths

            if mask_file and mask_file.filename:
                mask_path = os.path.join(upload_dir, mask_file.filename)
                mask_file.save(mask_path)
            else:
                mask_path = ""
        
        # Store paths in session manager
        session_manager = current_app.session_manager
        
        # Determine original filename based on load mode
        if load_mode == "upload":
            if isinstance(image_path, list):
                original_filename = f"{len(image_path)}_files_stack"
            else:
                original_filename = os.path.basename(image_path)
        else:
            if isinstance(image_path, list) or any(ch in str(image_path) for ch in ['*','?','[']) or os.path.isdir(image_path):
                original_filename = "stack"
            else:
                original_filename = os.path.basename(image_path)
        
        session_manager.update(
            image_path=image_path,
            mask_path=mask_path,
            load_mode=load_mode,
            image_name=original_filename,  # Store original filename
            mode3d=False  # Will be updated after loading
        )
        
        # Prepare driver images when working from dir/glob/list and masks live in same folder
        prepared_image_source = image_path
        is_dir_or_glob_or_list = isinstance(image_path, list) or (isinstance(image_path, str) and (os.path.isdir(image_path) or any(ch in image_path for ch in ['*','?','['])))
        if is_dir_or_glob_or_list:
            try:
                # Same-folder pairing if mask_path is same dir as images (or mask not provided)
                img_dir = image_path if (isinstance(image_path, str) and os.path.isdir(image_path)) else None
                mask_dir_candidate = mask_path if (mask_path and os.path.isdir(mask_path)) else img_dir
                if img_dir and mask_dir_candidate and os.path.abspath(img_dir) == os.path.abspath(mask_dir_candidate):
                    all_images = list_images_for_path(image_path)
                    drivers = [fp for fp in all_images if os.path.splitext(os.path.basename(fp))[0].endswith('_0000')]
                    if drivers:
                        prepared_image_source = drivers
            except Exception:
                pass

        # Load volume using prepared source
        if isinstance(prepared_image_source, list):
            # Stack list of 2D files into a volume
            volume = stack_2d_images(prepared_image_source)
        else:
            volume = load_image_or_stack(prepared_image_source)
        # Optional: build mask per-file pairing when working with folders/globs/lists
        mask = None
        if is_dir_or_glob_or_list:
            # Determine mask directory: same as images if mask_path is empty or is same dir
            mask_dir = None
            if mask_path and os.path.isdir(mask_path):
                mask_dir = mask_path
            elif not mask_path:
                mask_dir = None  # same directory as each image
            else:
                # mask provided as single file; fallback to generic alignment
                mask = load_mask_like(mask_path, volume)

            if mask is None:
                # Pair using the same drivers as volume load
                if isinstance(prepared_image_source, list):
                    image_files = prepared_image_source
                else:
                    image_files = list_images_for_path(prepared_image_source)
                mask = build_mask_stack_from_pairs(image_files, mask_dir)
        else:
            mask = load_mask_like(mask_path, volume) if mask_path else None
        
        print(f"DEBUG: Loaded volume shape: {volume.shape if volume is not None else 'None'}")
        print(f"DEBUG: Loaded mask shape: {mask.shape if mask is not None else 'None'}")
        
        # Store in app config for the session
        current_app.config["PROOFREADING_VOLUME"] = volume
        current_app.config["PROOFREADING_MASK"] = mask
        current_app.config["PROOFREADING_IMAGE_PATH"] = image_path
        current_app.config["PROOFREADING_MASK_PATH"] = mask_path
        current_app.config["PROOFREADING_EDITED_SLICES"] = set()
        
        # Update session with 3D info
        mode3d = volume.ndim == 3
        session_manager.update(mode3d=mode3d)
        
        print(f"DEBUG: Stored in config - volume: {current_app.config.get('PROOFREADING_VOLUME') is not None}, mask: {current_app.config.get('PROOFREADING_MASK') is not None}")
        
        # Determine if 3D
        mode3d = volume.ndim == 3
        num_slices = volume.shape[0] if mode3d else 1
        
        # Redirect to proofreading editor
        return redirect(url_for("proofreading_workflow.proofreading_editor"))
        
    except Exception as e:
        return render_template("proofreading_load.html", 
                            error=f"Error loading dataset: {str(e)}")

@bp.route("/editor")
def proofreading_editor():
    """Standalone proofreading editor."""
    volume = current_app.config.get("PROOFREADING_VOLUME")
    mask = current_app.config.get("PROOFREADING_MASK")
    
    if volume is None:
        return redirect(url_for("proofreading_workflow.proofreading_load"))
    
    # Ensure mask exists (create empty mask if none provided)
    if mask is None:
        if volume.ndim == 2:
            mask = np.zeros_like(volume, dtype=np.uint8)
        elif volume.ndim == 3:
            mask = np.zeros_like(volume, dtype=np.uint8)
        current_app.config["PROOFREADING_MASK"] = mask
        print(f"DEBUG: Created empty mask with shape {mask.shape}")
    
    mode3d = volume.ndim == 3
    num_slices = volume.shape[0] if mode3d else 1
    
    return render_template("proofreading_standalone.html",
                         mode3d=mode3d,
                         num_slices=num_slices,
                         volume_shape=volume.shape,
                         mask_shape=mask.shape if mask is not None else None,
                         z=0,  # Current slice index (always 0 for standalone)
                         slice_index=0)  # For template compatibility

@bp.route("/api/slice/<int:z>")
def api_slice(z):
    """Get image slice for standalone proofreading."""
    try:
        volume = current_app.config.get("PROOFREADING_VOLUME")
        print(f"DEBUG: api_slice called with z={z}, volume is None: {volume is None}")
        
        # If volume is not loaded, try to reload from session
        if volume is None:
            session_manager = current_app.session_manager
            session_state = session_manager.snapshot()
            image_path = session_state.get("image_path")
            
            if image_path:
                print(f"DEBUG: Reloading volume from {image_path}")
                if isinstance(image_path, list):
                    volume = stack_2d_images(image_path)
                else:
                    volume = load_image_or_stack(image_path)
                current_app.config["PROOFREADING_VOLUME"] = volume
                
                # Also reload mask if available
                mask_path = session_state.get("mask_path")
                if mask_path and os.path.exists(mask_path):
                    mask = load_mask_like(mask_path, volume)
                    current_app.config["PROOFREADING_MASK"] = mask
                    current_app.config["PROOFREADING_MASK_PATH"] = mask_path
            else:
                print("DEBUG: No volume loaded in api_slice")
                return jsonify(error="No volume loaded"), 404
        
        if volume.ndim == 2:
            sl = volume
        else:
            z = int(np.clip(z, 0, volume.shape[0] - 1))
            sl = volume[z]
        
        print(f"DEBUG: Processing slice shape: {sl.shape}")
        
        # Convert to RGB with consistent normalization
        arr = np.asarray(sl)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            rgb = arr.astype(np.uint8)
        else:
            # Consistent normalization for all image types
            if arr.max() > 0:
                arr = (arr / arr.max() * 255.0)
            else:
                arr = arr.astype(np.float64)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            rgb = np.stack([arr] * 3, axis=-1)
        
        # Ensure consistent data type and range
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        
        print(f"DEBUG: RGB shape: {rgb.shape}")
        
        bio = io.BytesIO()
        Image.fromarray(rgb).save(bio, format="PNG")
        bio.seek(0)
        return send_file(bio, mimetype="image/png")
        
    except Exception as e:
        print(f"DEBUG: Error in api_slice: {e}")
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500

@bp.route("/api/mask/<int:z>")
def api_mask(z):
    """Get mask slice for standalone proofreading."""
    try:
        mask = current_app.config.get("PROOFREADING_MASK")
        volume = current_app.config.get("PROOFREADING_VOLUME")
        print(f"DEBUG: api_mask called with z={z}, volume is None: {volume is None}, mask is None: {mask is None}")
        
        # If volume is not loaded, try to reload from session
        if volume is None:
            session_manager = current_app.session_manager
            session_state = session_manager.snapshot()
            image_path = session_state.get("image_path")
            
            if image_path:
                print(f"DEBUG: Reloading volume from {image_path}")
                if isinstance(image_path, list):
                    volume = stack_2d_images(image_path)
                else:
                    volume = load_image_or_stack(image_path)
                current_app.config["PROOFREADING_VOLUME"] = volume
                
                # Also reload mask if available
                mask_path = session_state.get("mask_path")
                if mask_path and os.path.exists(mask_path):
                    mask = load_mask_like(mask_path, volume)
                    current_app.config["PROOFREADING_MASK"] = mask
                    current_app.config["PROOFREADING_MASK_PATH"] = mask_path

        # if no mask loaded but an image exists, create a blank one
        if mask is None and volume is not None:
            if volume.ndim == 2:
                mask = np.zeros_like(volume, dtype=np.uint8)
            elif volume.ndim == 3:
                mask = np.zeros_like(volume, dtype=np.uint8)
            current_app.config["PROOFREADING_MASK"] = mask

        if mask is None:
            return jsonify(error="No mask loaded"), 404

        if mask.ndim == 2:
            sl = mask
        else:
            z = int(np.clip(z, 0, mask.shape[0] - 1))
            sl = mask[z]
        
        print(f"DEBUG: Processing mask slice shape: {sl.shape}")
        
        # Consistent mask processing
        mask_binary = (sl > 0).astype(np.uint8) * 255
        im = Image.fromarray(mask_binary)
        bio = io.BytesIO()
        im.save(bio, format="PNG")
        bio.seek(0)
        return send_file(bio, mimetype="image/png")
        
    except Exception as e:
        print(f"DEBUG: Error in api_mask: {e}")
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500

@bp.route("/api/names/<int:z>")
def api_names(z):
    """Return source image filename and paired mask filename for current slice.
    When images and masks share the same folder and a pair is found, suppress mask filename display.
    Supports nnUNet-style pairing (image *_0000, mask without suffix).
    """
    try:
        session_manager = current_app.session_manager
        session_state = session_manager.snapshot()
        image_path = session_state.get("image_path", "")
        mask_path = session_state.get("mask_path", "")

        # Build driver list consistent with loader
        images = list_images_for_path(image_path)
        try:
            img_dir = image_path if (isinstance(image_path, str) and os.path.isdir(image_path)) else None
            mask_dir_candidate = mask_path if (mask_path and os.path.isdir(mask_path)) else img_dir
            if img_dir and mask_dir_candidate and os.path.abspath(img_dir) == os.path.abspath(mask_dir_candidate):
                drivers = [fp for fp in images if os.path.splitext(os.path.basename(fp))[0].endswith('_0000')]
                if drivers:
                    images = drivers
        except Exception:
            pass
        if not images:
            return jsonify(image=None, mask=None)
        idx = int(np.clip(z, 0, len(images) - 1))
        img_fp = images[idx]

        # Resolve mask name with both conventions
        base, ext = os.path.splitext(os.path.basename(img_fp))
        mdir = mask_path if (mask_path and os.path.isdir(mask_path)) else os.path.dirname(img_fp)

        mask_fp = None
        found_path = None
        for e in [ext.lower(), ".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
            cand = os.path.join(mdir, f"{base}_mask{e}")
            if os.path.exists(cand):
                found_path = cand
                mask_fp = cand
                break
        if mask_fp is None and base.endswith("_0000"):
            trimmed = base[:-5]
            for e in [ext.lower(), ".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                cand = os.path.join(mdir, f"{trimmed}{e}")
                if os.path.exists(cand):
                    found_path = cand
                    mask_fp = cand
                    break

        # If same folder and a pair was found, suppress independent mask display
        try:
            same_folder = (mask_path and os.path.isdir(mask_path) and os.path.abspath(mask_path) == os.path.abspath(os.path.dirname(img_fp))) or \
                          (isinstance(image_path, str) and os.path.isdir(image_path) and mask_path and os.path.isdir(mask_path) and os.path.abspath(image_path) == os.path.abspath(mask_path))
        except Exception:
            same_folder = False
        mask_name = None if (same_folder and found_path is not None) else (os.path.basename(mask_fp) if mask_fp else None)

        return jsonify(image=os.path.basename(img_fp), mask=mask_name)
    except Exception as e:
        return jsonify(error=str(e)), 500

@bp.route("/api/mask/update", methods=["POST"])
def api_mask_update():
    """Update mask for standalone proofreading."""
    data = request.get_json(force=True)
    mask = current_app.config.get("PROOFREADING_MASK")
    volume = current_app.config.get("PROOFREADING_VOLUME")

    # --- ensure mask exists for 2D/3D cases ---
    if mask is None and volume is not None:
        if volume.ndim == 2:
            mask = np.zeros_like(volume, dtype=np.uint8)
        elif volume.ndim == 3:
            mask = np.zeros_like(volume, dtype=np.uint8)
        current_app.config["PROOFREADING_MASK"] = mask
    elif mask is None:
        return jsonify(success=False, error="No mask or image loaded"), 404

    # --- Batch updates ---
    if "full_batch" in data:
        edited = set(current_app.config.get("PROOFREADING_EDITED_SLICES", set()))
        for item in data["full_batch"]:
            z = int(item["z"])
            png_bytes = base64.b64decode(item["png"])
            img = Image.open(io.BytesIO(png_bytes)).convert("L")
            arr = (np.array(img) > 127).astype(np.uint8)

            if mask.ndim == 2:
                arr = cv2.resize(arr, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask[:, :] = arr
            else:
                arr = cv2.resize(arr, (mask.shape[2], mask.shape[1]), interpolation=cv2.INTER_NEAREST)
                mask[z] = arr
            edited.add(z)
        current_app.config["PROOFREADING_MASK"] = mask
        current_app.config["PROOFREADING_EDITED_SLICES"] = edited
        print(f"✅ Batch updated {len(data['full_batch'])} slice(s)")
        return jsonify(success=True)

    # --- Single slice update ---
    if "full_png" in data:
        z = int(data.get("z", 0))
        png_bytes = base64.b64decode(data["full_png"])
        img = Image.open(io.BytesIO(png_bytes)).convert("L")
        arr = (np.array(img) > 127).astype(np.uint8)

        if mask.ndim == 2:
            arr = cv2.resize(arr, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask[:, :] = arr
        else:
            arr = cv2.resize(arr, (mask.shape[2], mask.shape[1]), interpolation=cv2.INTER_NEAREST)
            mask[z] = arr

        current_app.config["PROOFREADING_MASK"] = mask
        edited = set(current_app.config.get("PROOFREADING_EDITED_SLICES", set()))
        edited.add(z)
        current_app.config["PROOFREADING_EDITED_SLICES"] = edited
        print(f"✅ Replaced full slice {z}")
        return jsonify(success=True)

    return jsonify(success=False, error="Invalid data"), 400

@bp.route("/api/save", methods=["POST"])
def api_save():
    """Save mask for standalone proofreading."""
    mask = current_app.config.get("PROOFREADING_MASK")
    volume = current_app.config.get("PROOFREADING_VOLUME")

    if mask is None and volume is not None:
        if volume.ndim == 2:
            mask = np.zeros_like(volume, dtype=np.uint8)
        elif volume.ndim == 3:
            mask = np.zeros_like(volume, dtype=np.uint8)
        current_app.config["PROOFREADING_MASK"] = mask
    elif mask is None:
        return jsonify(success=False, error="No mask or image loaded"), 404

    # Get session data
    session_manager = current_app.session_manager
    session_state = session_manager.snapshot()
    img_path = session_state.get("image_path", "")
    mask_path = session_state.get("mask_path", "")
    load_mode = session_state.get("load_mode", "path")

    # Generate save destination
    # If the dataset came from a folder/glob/multiple files, save per-slice masks into a sibling "<dir>_mask" folder
    src_is_dir = bool(img_path and isinstance(img_path, str) and os.path.isdir(img_path))
    src_is_glob = bool(img_path and isinstance(img_path, str) and any(ch in img_path for ch in ['*','?','[']) and not os.path.exists(img_path))
    src_is_list = isinstance(img_path, list)

    # Folder/glob/list datasets: always save per-slice files inside mask folder
    # Case A: a mask folder was provided by user
    if (src_is_dir or src_is_glob or src_is_list) and mask_path and os.path.isdir(mask_path):
        mask_dir = mask_path
        os.makedirs(mask_dir, exist_ok=True)

        # Only save edited slice(s)
        edited = current_app.config.get("PROOFREADING_EDITED_SLICES", set())
        edited_list = sorted(list(edited))
        if not edited_list:
            return jsonify(success=True, message="No edited slices to save"), 200

        # Build source file list for naming
        image_path = session_state.get("image_path", "")
        image_files = list_images_for_path(image_path)
        if not image_files:
            return jsonify(success=False, error="No source files found to derive mask names"), 400

        for z in edited_list:
            if z < 0:
                continue
            src_fp = image_files[z if z < len(image_files) else -1]
            base = os.path.splitext(os.path.basename(src_fp))[0]
            ext_out = os.path.splitext(src_fp)[-1].lower()
            out_fp = os.path.join(mask_dir, f"{base}_mask{ext_out}")
            sl = mask[z] if (mask.ndim == 3 and z < mask.shape[0]) else (mask if mask.ndim == 2 else None)
            if sl is None:
                continue
            save_mask(sl, out_fp)

        # Keep mask_path as folder
        session_manager.update(mask_path=mask_dir)
        current_app.config["PROOFREADING_MASK_PATH"] = mask_dir
        current_app.config["PROOFREADING_EDITED_SLICES"] = set()
        return jsonify(success=True, message=f"Mask slices saved to {mask_dir}")

    # Case B: user didn't supply a mask folder; create sibling folder
    if (src_is_dir or src_is_glob or src_is_list) and not mask_path:
        # Build ordered source file list consistent with loader
        def list_source_files():
            if src_is_list:
                return list(img_path)
            if src_is_dir:
                chosen = []
                for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                    chosen = sorted([
                        os.path.join(img_path, f)
                        for f in os.listdir(img_path)
                        if f.lower().endswith(ext)
                    ])
                    if chosen:
                        break
                return chosen
            # glob string
            import glob as _glob
            return sorted(_glob.glob(img_path))

        src_files = list_source_files()
        if not src_files:
            return jsonify(success=False, error="No source files found to derive mask names"), 400

        # Determine destination mask folder next to the uploaded folder
        if src_is_list:
            first_dir = os.path.dirname(src_files[0]) if src_files else os.path.abspath(".")
            parent = os.path.dirname(first_dir)
            dir_name = os.path.basename(first_dir)
            mask_dir = os.path.join(parent, f"{dir_name}_mask")
        elif src_is_dir:
            parent = os.path.dirname(img_path)
            dir_name = os.path.basename(img_path)
            mask_dir = os.path.join(parent, f"{dir_name}_mask")
        else:
            # glob: use base directory of the glob and folder name
            base_dir = os.path.dirname(img_path)
            dir_name = os.path.basename(base_dir)
            parent = os.path.dirname(base_dir)
            mask_dir = os.path.join(parent, f"{dir_name}_mask")

        os.makedirs(mask_dir, exist_ok=True)

        # Only save edited slice(s)
        edited = current_app.config.get("PROOFREADING_EDITED_SLICES", set())
        edited_list = sorted(list(edited))
        if not edited_list:
            return jsonify(success=True, message="No edited slices to save"), 200
        for z in edited_list:
            if z < 0:
                continue
            src_fp = src_files[z if z < len(src_files) else -1]
            base = os.path.splitext(os.path.basename(src_fp))[0]
            ext_out = os.path.splitext(src_fp)[-1].lower()
            out_fp = os.path.join(mask_dir, f"{base}_mask{ext_out}")
            sl = mask[z] if (mask.ndim == 3 and z < mask.shape[0]) else (mask if mask.ndim == 2 else None)
            if sl is None:
                continue
            save_mask(sl, out_fp)

        # Update session to reference the mask directory
        session_manager.update(mask_path=mask_dir)
        current_app.config["PROOFREADING_MASK_PATH"] = mask_dir
        # Clear edited set after successful save
        current_app.config["PROOFREADING_EDITED_SLICES"] = set()
        return jsonify(success=True, message=f"Mask slices saved to {mask_dir}")

    # Default behavior: single file path (TIFF stack or 2D image)
    # Determine save directory and filename
    if load_mode == "upload" or not img_path or not os.path.exists(img_path):
        base_dir = os.path.abspath("./_uploads")
        os.makedirs(base_dir, exist_ok=True)
        image_name = session_state.get("image_name", "image")
        if not image_name or image_name == "image":
            image_name = "image"
        base_name = os.path.splitext(os.path.basename(image_name))[0]
    else:
        base_dir = os.path.dirname(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

    ext = ".tif"
    if isinstance(img_path, str) and img_path:
        _, src_ext = os.path.splitext(img_path.lower())
        if src_ext in [".png", ".jpg", ".jpeg"]:
            ext = src_ext

    if not mask_path:
        mask_path = os.path.join(base_dir, f"{base_name}_mask{ext}")
        session_manager.update(mask_path=mask_path)
        current_app.config["PROOFREADING_MASK_PATH"] = mask_path

    try:
        print(f"DEBUG: Saving mask with shape {mask.shape if mask is not None else 'None'}")
        print(f"DEBUG: Saving to path: {mask_path}")
        print(f"DEBUG: Mask path exists: {os.path.exists(mask_path) if mask_path else 'No path'}")
        
        save_mask(mask, mask_path)
        print(f"DEBUG: Save completed successfully")
        return jsonify(success=True, message=f"Mask saved to {mask_path}")
    except Exception as e:
        print(f"Save error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify(success=False, error=f"Failed to save mask: {str(e)}"), 500

@bp.route("/api/dims", methods=["POST"])
def api_dims():
    """Get dimensions of uploaded file for standalone proofreading."""
    return jsonify_dimensions(request.files.get("file"), use_temp_file=True)
