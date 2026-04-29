import os
import sys
import uuid
import time
import cv2
import numpy as np
import shutil
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import welch, detrend

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s"
)
logger = logging.getLogger("VITALIS_BACKEND")

# Add parent directory to path to import rppg engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rppg


# --- Domain Models ---
@dataclass
class BiometricChunk:
    chunk_index: int
    start_time: float
    end_time: float
    bpm: Optional[float]
    sqi: float
    respiratory_rate: float
    latency_ms: float
    processing_speed: float


# --- Services ---


class SignalAnalyzer:
    """Handles biometric signal extraction and analysis using the rPPG engine."""

    def __init__(self, model_name: str = "FacePhys.rlap"):
        self.model_name = model_name
        self._model = self._initialize_model()

    def _initialize_model(self) -> Optional[rppg.Model]:
        logger.info(f"Initializing rPPG Engine: {self.model_name} (JIT Warming up)...")
        try:
            return rppg.Model(self.model_name)
        except Exception as e:
            logger.error(f"Critical Engine Failure: {e}")
            return None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def get_model(self) -> rppg.Model:
        if not self._model:
            raise RuntimeError("Engine not initialized")
        return self._model

    def estimate_respiratory_rate(self, bvp_signal: Any, fps: float = 30.0) -> float:
        """Pure function to estimate RR from BVP signal using PSD analysis."""
        if bvp_signal is None or (
            isinstance(bvp_signal, (list, np.ndarray)) and len(bvp_signal) < 100
        ):
            return 0.0

        try:
            bvp_arr = np.array(bvp_signal)
            bvp_detrended = detrend(bvp_arr)
            freqs, psd = welch(bvp_detrended, fs=fps, nperseg=len(bvp_detrended))

            # Respiratory range: 0.1Hz to 0.5Hz (6-30 Br/m)
            mask = (freqs >= 0.1) & (freqs <= 0.5)
            if not np.any(mask):
                return 15.0  # Normal fallback

            rr_freq = freqs[mask][np.argmax(psd[mask])]
            return float(rr_freq * 60)
        except Exception as e:
            logger.debug(f"RR Estimation error: {e}")
            return 0.0

    async def wait_for_inference(self, target_time: float, max_wait: float = 60.0):
        """Deep synchronization: Guarantees AI buffer has results for the target window."""
        model = self.get_model()
        wait_start = time.time()

        # We need the signal buffer to be AT LEAST as long as the target window
        # fps * target_time gives the absolute sample count needed
        required_samples = int(target_time * model.fps)

        while model.n_signal < required_samples:
            elapsed = time.time() - wait_start
            if elapsed > max_wait:
                logger.warning(
                    f"DEEP SYNC TIMEOUT: {model.n_signal}/{required_samples} samples at {target_time}s."
                )
                break

            # Use longer sleep to allow AI thread more CPU cycles
            await asyncio.sleep(0.5)

        logger.info(
            f"Deep Sync Ready: {model.n_signal} samples for {target_time}s window."
        )


# Initialize global singleton analyzer
analyzer = SignalAnalyzer()

# --- Application Layer ---

app = FastAPI(title="VITALIS Near Real-Time rPPG API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class DataSanitizer:
    """Handles JSON serialization safety for NumPy types."""

    @staticmethod
    def sanitize(data: Any) -> Any:
        if isinstance(data, dict):
            return {k: DataSanitizer.sanitize(v) for k, v in data.items()}
        if isinstance(data, list):
            return [DataSanitizer.sanitize(v) for v in data]
        if isinstance(data, (np.integer, np.int64)):
            return int(data)
        if isinstance(data, (np.floating, np.float32, np.float64)):
            return float(data)
        if isinstance(data, np.ndarray):
            return data.tolist()
        return data


class VideoProcessor:
    """Orchestrates the incremental video processing workflow."""

    def __init__(self, file_path: str, websocket: WebSocket):
        self.file_path = file_path
        self.websocket = websocket
        self.chunk_size_sec = 5.0
        self.chunk_bpms = []

    async def run(self):
        if not analyzer.ready:
            await self._fail("AI Engine offline")
            return

        cap = cv2.VideoCapture(self.file_path)
        try:
            if not cap.isOpened():
                await self._fail("Video source unreadable")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start_time = time.time()

            current_window_start = 0.0
            frame_idx = 0
            model = analyzer.get_model()

            with model:
                logger.info(f"Analysis started: {total_frames} frames @ {fps}fps")
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    ts = frame_idx / fps
                    model.update_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ts)

                    # BACKPRESSURE: If ingestion is >5s ahead of inference, wait
                    if frame_idx % 60 == 0:
                        inference_ts = model.n_signal / model.fps
                        if ts - inference_ts > 5.0:
                            logger.debug(
                                f"Backpressure: Ingestion {ts:.1f}s vs Inference {inference_ts:.1f}s. Waiting..."
                            )
                            await asyncio.sleep(0.5)

                    if ts >= current_window_start + self.chunk_size_sec:
                        await self._process_window(current_window_start, ts, start_time)
                        current_window_start = ts

                    frame_idx += 1
                    if frame_idx % 30 == 0:
                        await asyncio.sleep(0.01)

                # Finalize remaining data
                await self._finalize(
                    current_window_start, frame_idx / fps, total_frames, start_time
                )
        finally:
            cap.release()
            logger.debug("VideoCapture released")

    async def _process_window(self, start: float, end: float, session_start: float):
        await analyzer.wait_for_inference(end)

        model = analyzer.get_model()
        metrics = model.hr(start=start, end=end) or {}
        bvp, _ = model.bvp(start=start, end=end)

        bpm = metrics.get("hr")
        if bpm is not None:
            self.chunk_bpms.append(float(bpm))

        chunk_data = {
            "type": "chunk_update",
            "chunk_index": int(start // self.chunk_size_sec),
            "start_time": round(start, 1),
            "end_time": round(end, 1),
            "bpm": round(bpm, 1) if bpm is not None else None,
            "sqi": round(float(metrics.get("SQI") or 0.0), 2),
            "respiratory_rate": round(
                float(analyzer.estimate_respiratory_rate(bvp, model.fps) or 0.0), 1
            ),
            "latency_ms": round(float(metrics.get("latency") or 0.0) * 1000, 0),
            "processing_speed": end / (time.time() - session_start),
        }
        await self.websocket.send_json(DataSanitizer.sanitize(chunk_data))

    async def _finalize(
        self, last_start: float, last_ts: float, total_frames: int, session_start: float
    ):
        model = analyzer.get_model()

        # Send final partial chunk if significant
        if last_ts > last_start + 1.0:
            await self._process_window(last_start, last_ts, session_start)

        # Final Sync: Be very patient for 1min+ uncompressed videos
        logger.info(
            f"Awaiting final AI completion ({model.n_signal}/{total_frames})..."
        )
        final_wait_start = time.time()
        while model.n_signal < total_frames - 10:
            if time.time() - final_wait_start > 60.0:
                logger.warning("Final synchronization timeout reached.")
                break
            await asyncio.sleep(0.5)

        final_metrics = model.hr() or {}
        bvp, _ = model.bvp()
        overall_bpm = (
            float(np.median(self.chunk_bpms))
            if self.chunk_bpms
            else final_metrics.get("hr")
        )

        final_data = {
            "type": "final_result",
            "overall_bpm": round(overall_bpm, 1) if overall_bpm is not None else None,
            "overall_sqi": round(float(final_metrics.get("SQI") or 0.0), 2),
            "overall_respiratory_rate": round(
                float(analyzer.estimate_respiratory_rate(bvp, model.fps) or 0.0), 1
            ),
            "total_processing_time_sec": round(time.time() - session_start, 1),
            "average_latency_ms": round(
                float(final_metrics.get("latency") or 0.0) * 1000, 0
            ),
            "video_duration_sec": round(last_ts, 1),
        }
        await self.websocket.send_json(DataSanitizer.sanitize(final_data))
        logger.info("Session finalized.")

    async def _fail(self, message: str):
        logger.error(f"Session failed: {message}")
        await self.websocket.send_json({"error": message})


# --- Routes ---


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"session_id": session_id, "file_path": path}


@app.websocket("/ws/process/{session_id}")
async def websocket_handler(websocket: WebSocket, session_id: str):
    await websocket.accept()

    # Locate session file
    file_path = next(
        (
            os.path.join(UPLOAD_DIR, f)
            for f in os.listdir(UPLOAD_DIR)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
