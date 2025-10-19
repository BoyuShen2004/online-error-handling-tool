"""
Error Handling Tool - Detection Workflow Routes
---------------------------------------------
Handles the error detection workflow (original functionality).
"""

import os
from flask import Blueprint, render_template, request, redirect, url_for, current_app, jsonify
from backend.data_manager import DataManager
from backend.session_manager import SessionManager

bp = Blueprint("detection_workflow", __name__, url_prefix="")

def register_detection_workflow_routes(app):
    app.register_blueprint(bp)

@bp.route("/detection/load")
def detection_load():
    """Load dataset for error detection workflow."""
    # Check if there's existing data
    volume = current_app.config.get("DETECTION_VOLUME")
    mask = current_app.config.get("DETECTION_MASK")
    image_path = current_app.config.get("DETECTION_IMAGE_PATH")
    mask_path = current_app.config.get("DETECTION_MASK_PATH")
    session_manager = current_app.session_manager
    session_state = session_manager.snapshot()
    layers = session_state.get("layers", [])
    
    has_existing_data = volume is not None and image_path is not None and len(layers) > 0
    
    if has_existing_data:
        return render_template("detection_load.html",
                             has_existing_data=True,
                             existing_image_path=os.path.basename(image_path),
                             existing_mask_path=os.path.basename(mask_path) if mask_path else None,
                             existing_shape=" × ".join(map(str, volume.shape)),
                             existing_mode3d=volume.ndim == 3,
                             existing_layers_count=len(layers))
    else:
        return render_template("detection_load.html")

@bp.route("/detection/clear", methods=["POST"])
def detection_clear():
    """Clear previously loaded data for error detection."""
    try:
        session_manager = current_app.session_manager
        
        # Clear session data
        session_manager.reset_session()
        
        # Clear app config
        current_app.config.pop("DETECTION_VOLUME", None)
        current_app.config.pop("DETECTION_MASK", None)
        current_app.config.pop("DETECTION_IMAGE_PATH", None)
        current_app.config.pop("DETECTION_MASK_PATH", None)
        current_app.config.pop("DETECTION_DATA_MANAGER", None)
        
        return jsonify({"success": True, "message": "Data cleared successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/detection/load", methods=["POST"])
def detection_load_post():
    """Handle dataset loading for error detection."""
    try:
        session_manager = current_app.session_manager
        
        # Clear previously loaded data before loading new data
        session_manager.reset_session()
        current_app.config.pop("DETECTION_VOLUME", None)
        current_app.config.pop("DETECTION_MASK", None)
        current_app.config.pop("DETECTION_IMAGE_PATH", None)
        current_app.config.pop("DETECTION_MASK_PATH", None)
        current_app.config.pop("DETECTION_DATA_MANAGER", None)
        
        # Get form data
        load_mode = request.form.get("load_mode", "path")
        image_path = request.form.get("image_path", "").strip()
        mask_path = request.form.get("mask_path", "").strip()
        
        if load_mode == "path":
            if not image_path:
                return render_template("detection_load.html", 
                                    error="Image path is required",
                                    image_path=image_path,
                                    mask_path=mask_path)
            
            if not os.path.exists(image_path):
                return render_template("detection_load.html", 
                                    error="Image file not found",
                                    image_path=image_path,
                                    mask_path=mask_path)
            
            if mask_path and not os.path.exists(mask_path):
                return render_template("detection_load.html", 
                                    error="Mask file not found",
                                    image_path=image_path,
                                    mask_path=mask_path)
        
        else:  # upload mode
            image_file = request.files.get("image_file")
            mask_file = request.files.get("mask_file")
            
            if not image_file or image_file.filename == "":
                return render_template("detection_load.html", 
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
        
        # Load data
        data_manager = DataManager()
        
        # Load image
        volume, volume_info = data_manager.load_image(image_path)
        
        # Load mask if provided
        mask = None
        mask_info = {}
        if mask_path:
            mask, mask_info = data_manager.load_mask(mask_path)
        
        # Validate compatibility
        if not data_manager.validate_data_compatibility():
            return render_template("detection_load.html", 
                                error="Image and mask dimensions are incompatible",
                                image_path=image_path if load_mode == "path" else "",
                                mask_path=mask_path if load_mode == "path" else "")
        
        # Generate layers
        layers = data_manager.generate_all_layers()
        
        # Update session
        session_manager.update(
            mode3d=volume_info["is_3d"],
            image_path=image_path,
            mask_path=mask_path,
            load_mode=load_mode,
            image_name=image_file.filename if load_mode == "upload" else os.path.basename(image_path)
        )
        session_manager.set_image_info(image_path, load_mode)
        
        # Add layers to session
        for layer_data in layers:
            session_manager.add_layer(layer_data)
        
        # Store data in app config for API access
        current_app.config["DETECTION_VOLUME"] = volume
        current_app.config["DETECTION_MASK"] = mask
        current_app.config["DETECTION_IMAGE_PATH"] = image_path
        current_app.config["DETECTION_MASK_PATH"] = mask_path
        current_app.config["DETECTION_DATA_MANAGER"] = data_manager
        
        # Redirect to detection page
        return redirect(url_for("detection.detection"))
        
    except Exception as e:
        return render_template("detection_load.html", 
                            error=f"Error loading dataset: {str(e)}")

@bp.route("/api/dims", methods=["POST"])
def api_dims():
    """Get dimensions of uploaded file."""
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        # Save temporarily
        temp_path = f"temp_{file.filename}"
        file.save(temp_path)
        
        try:
            from backend.data_manager import DataManager
            data_manager = DataManager()
            volume, volume_info = data_manager.load_image(temp_path)
            
            shape = volume.shape
            return jsonify({"shape": list(shape)})
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500
