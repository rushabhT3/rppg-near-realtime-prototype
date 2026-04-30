import os
import uuid
import shutil
import logging
from datetime import datetime

from fastapi import UploadFile, File, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.logging import get_logger
from app.services.processor import VideoProcessor

logger = get_logger()


def setup_routes(app):
    """Register all routes with the FastAPI application."""

    @app.get("/health")
    async def health_check():
        """Health check endpoint for monitoring and load balancers."""
        return {
            "status": "healthy",
            "service": "vitalis-rppg",
            "version": settings.APP_VERSION,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": "running",
        }

    @app.get("/ping")
    async def ping():
        """Simple ping endpoint for basic connectivity test."""
        return {"pong": True, "timestamp": datetime.utcnow().isoformat()}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        session_id = str(uuid.uuid4())
        path = os.path.join(settings.upload_dir, f"{session_id}_{file.filename}")
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"session_id": session_id, "file_path": path}

    @app.websocket("/ws/process/{session_id}")
    async def websocket_handler(websocket: WebSocket, session_id: str):
        await websocket.accept()

        # Locate session file
        file_path = next(
            (
                os.path.join(settings.upload_dir, f)
                for f in os.listdir(settings.upload_dir)
                if f.startswith(session_id)
            ),
            None,
        )

        if not file_path:
            await websocket.send_json({"error": "Session context not found"})
            await websocket.close()
            return

        try:
            processor = VideoProcessor(file_path, websocket)
            await processor.run()
        except WebSocketDisconnect:
            logger.info(f"Client disconnected: {session_id}")
        except Exception as e:
            logger.error(f"E2E Processing Error: {e}")
            await websocket.send_json({"error": "Internal processing failure"})
        finally:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
