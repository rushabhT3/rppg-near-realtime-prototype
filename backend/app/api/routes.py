import os
import uuid
import shutil
import asyncio
import logging
from datetime import datetime

from fastapi import UploadFile, File, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse

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

    @app.get("/api/upload-url")
    async def get_upload_url(
        filename: str = Query(..., description="Name of the video file")
    ):
        """Get a signed URL for uploading video directly to GCS.
        This bypasses Cloud Run's 32MB request body limit."""
        try:
            from app.services.storage import generate_signed_upload_url

            result = generate_signed_upload_url(filename)
            return result
        except Exception as e:
            logger.error(f"GCS upload URL generation failed: {e}")
            return JSONResponse(
                status_code=503,
                content={"error": "GCS storage unavailable", "fallback": "/api/upload"},
            )

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        """Upload video directly (for local dev or GCS fallback)."""
        session_id = str(uuid.uuid4())
        path = os.path.join(settings.upload_dir, f"{session_id}_{file.filename}")
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"session_id": session_id, "file_path": path}

    @app.websocket("/ws/process/{session_id}")
    async def websocket_handler(
        websocket: WebSocket,
        session_id: str,
        object_name: str = Query(default=None),
    ):
        await websocket.accept()

        file_path = None
        is_gcs = object_name is not None

        try:
            if is_gcs:
                from app.services.storage import download_to_local

                file_path = await asyncio.to_thread(download_to_local, object_name)
            else:
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
            if is_gcs and object_name:
                try:
                    from app.services.storage import delete_object

                    delete_object(object_name)
                except Exception:
                    pass
