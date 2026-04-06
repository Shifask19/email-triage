"""
Server entry point for openenv validate multi-mode deployment.
"""
import os
import uvicorn
from app import app


def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
