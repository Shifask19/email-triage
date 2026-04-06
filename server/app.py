# Re-export the FastAPI app from the root app.py
# Required by openenv validate for multi-mode deployment
from app import app

__all__ = ["app"]
