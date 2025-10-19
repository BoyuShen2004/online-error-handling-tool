"""
Error Handling Tool (EHTool)
============================
Integrated application combining error detection and proofreading functionality.
Users can select layers as correct, incorrect, or unsure, with incorrect layers
sent to proofreading tool and unsure layers available for later review.
"""

import os
import atexit
import signal
import sys
from flask import Flask
from routes.landing import register_landing_routes
from routes.detection import register_detection_routes
from routes.review import register_review_routes
from routes.proofreading import register_proofreading_routes
from routes.export import register_export_routes
from routes.detection_workflow import register_detection_workflow_routes
from routes.proofreading_workflow import register_proofreading_workflow_routes
from backend.session_manager import SessionManager

def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("EHTool_SECRET", "dev-secret")
    
    # Attach a global session manager to app
    app.session_manager = SessionManager()
    
    # Register routes
    register_landing_routes(app)
    register_detection_routes(app)
    register_review_routes(app)
    register_proofreading_routes(app)
    register_export_routes(app)
    register_detection_workflow_routes(app)
    register_proofreading_workflow_routes(app)
    
    return app

def cleanup():
    """Cleanup function to handle resource cleanup."""
    try:
        # Suppress multiprocessing warnings
        import warnings
        warnings.filterwarnings("ignore", category=ResourceWarning)
    except:
        pass

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\n🛑 Shutting down Error Handling Tool...")
    sys.exit(0)

if __name__ == "__main__":
    # Register cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    host = os.environ.get("EHTool_HOST", "0.0.0.0")
    port = int(os.environ.get("EHTool_PORT", "5004"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app = create_app()
    print(f"✅ Error Handling Tool running on http://{host}:{port}  (debug={debug})")
    app.run(host=host, port=port, debug=debug)
