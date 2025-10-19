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
from backend.volume_manager import load_image_or_stack, load_mask_like, save_mask

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
            image_file = request.files.get("image_file")
            mask_file = request.files.get("mask_file")
            
            if not image_file or image_file.filename == "":
                return render_template("proofreading_load.html", 
                                    error="Image file is required")
            
            # Save uploaded files temporarily
            upload_dir = "_uploads"
            os.makedirs(upload_dir, exist_ok=True)
            
            image_path = os.path.join(upload_dir, image_file.filename)
            image_file.save(image_path)
            
            if mask_file and mask_file.filename:
                mask_path = os.path.join(upload_dir, mask_file.filename)
                mask_file.save(mask_path)
            else:
                mask_path = ""
        
        # Store paths in session manager
        session_manager = current_app.session_manager
        
        # Determine original filename based on load mode
        if load_mode == "upload":
            original_filename = image_file.filename
        else:
            original_filename = os.path.basename(image_path)
        
        session_manager.update(
            image_path=image_path,
            mask_path=mask_path,
            load_mode=load_mode,
            image_name=original_filename,  # Store original filename
            mode3d=False  # Will be updated after loading
        )
        
        # Load volume and mask
        volume = load_image_or_stack(image_path)
        mask = load_mask_like(mask_path, volume) if mask_path else None
        
        print(f"DEBUG: Loaded volume shape: {volume.shape if volume is not None else 'None'}")
        print(f"DEBUG: Loaded mask shape: {mask.shape if mask is not None else 'None'}")
        
        # Store in app config for the session
        current_app.config["PROOFREADING_VOLUME"] = volume
        current_app.config["PROOFREADING_MASK"] = mask
        current_app.config["PROOFREADING_IMAGE_PATH"] = image_path
        current_app.config["PROOFREADING_MASK_PATH"] = mask_path
        
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
            
            if image_path and os.path.exists(image_path):
                print(f"DEBUG: Reloading volume from {image_path}")
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
        
        # Convert to RGB
        arr = np.asarray(sl)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            rgb = arr.astype(np.uint8)
        else:
            arr = (arr / arr.max() * 255.0) if arr.max() > 0 else arr
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            rgb = np.stack([arr] * 3, axis=-1)
        
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
            
            if image_path and os.path.exists(image_path):
                print(f"DEBUG: Reloading volume from {image_path}")
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
        
        im = Image.fromarray((sl > 0).astype(np.uint8) * 255)
        bio = io.BytesIO()
        im.save(bio, format="PNG")
        bio.seek(0)
        return send_file(bio, mimetype="image/png")
        
    except Exception as e:
        print(f"DEBUG: Error in api_mask: {e}")
        import traceback
        traceback.print_exc()
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
        current_app.config["PROOFREADING_MASK"] = mask
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

    # Determine save directory and filename (like PFTool)
    if load_mode == "upload" or not img_path or not os.path.exists(img_path):
        base_dir = os.path.abspath("./_uploads")
        os.makedirs(base_dir, exist_ok=True)
        image_name = session_state.get("image_name", "image")
        if not image_name or image_name == "image":
            image_name = "image"  # Default fallback
        base_name = os.path.splitext(os.path.basename(image_name))[0]
    else:
        base_dir = os.path.dirname(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Detect extension
    ext = ".tif"
    if img_path:
        _, src_ext = os.path.splitext(img_path.lower())
        if src_ext in [".png", ".jpg", ".jpeg"]:
            ext = src_ext

    # Generate mask path if not provided
    if not mask_path:
        # Get the original filename without extension
        if load_mode == "upload" or not img_path or not os.path.exists(img_path):
            original_name = session_state.get("image_name", "image")
            if not original_name or original_name == "image":
                original_name = "image"  # Default fallback
            original_base = os.path.splitext(os.path.basename(original_name))[0]
        else:
            original_base = os.path.splitext(os.path.basename(img_path))[0]
        
        mask_path = os.path.join(base_dir, f"{original_base}_mask{ext}")
        # Update session with generated mask path
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
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        # Save temporarily
        temp_path = f"temp_{file.filename}"
        file.save(temp_path)
        
        try:
            volume = load_image_or_stack(temp_path)
            shape = volume.shape
            return jsonify({"shape": list(shape)})
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500
