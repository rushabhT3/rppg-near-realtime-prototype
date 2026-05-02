import time
import asyncio
from dataclasses import asdict
from typing import List

import cv2
import numpy as np
from fastapi import WebSocket

from .analyzer import get_analyzer
from .sanitizer import DataSanitizer
from app.models.biometric import BiometricChunk
from app.core.logging import get_logger

logger = get_logger()


class VideoProcessor:
    """Orchestrates the incremental video processing workflow."""

    def __init__(self, file_path: str, websocket: WebSocket):
        self.file_path = file_path
        self.websocket = websocket
        self.chunk_size_sec = 5.0
        self.chunk_bpms: List[float] = []
        self.chunk_sqis: List[float] = []

    async def run(self):
        # Load model in thread pool to avoid blocking the event loop
        analyzer = await asyncio.to_thread(get_analyzer)
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
            session_start = time.time()

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
                        await self._process_window(
                            current_window_start, ts, session_start
                        )
                        current_window_start = ts

                    frame_idx += 1
                    if frame_idx % 30 == 0:
                        await asyncio.sleep(0.01)

                await self._finalize(
                    current_window_start, frame_idx / fps, total_frames, session_start
                )
        finally:
            cap.release()
            logger.debug("VideoCapture released")

    async def _process_window(self, start: float, end: float, session_start: float):
        chunk_wall_start = time.time()

        analyzer = get_analyzer()
        await analyzer.wait_for_inference(end)

        model = analyzer.get_model()
        metrics = model.hr(start=start, end=end, return_hrv=False) or {}
        bvp, _ = model.bvp(start=start, end=end)

        bpm = metrics.get("hr")
        sqi = float(metrics.get("SQI") or 0.0)
        if bpm is not None:
            self.chunk_bpms.append(float(bpm))
            self.chunk_sqis.append(sqi)

        chunk_wall_ms = round((time.time() - chunk_wall_start) * 1000, 1)
        chunk_duration = end - start
        processing_speed = (
            round(chunk_duration / (chunk_wall_ms / 1000.0), 2)
            if chunk_wall_ms > 0
            else 0.0
        )

        chunk = BiometricChunk(
            chunk_index=int(start // self.chunk_size_sec),
            start_time=round(start, 1),
            end_time=round(end, 1),
            bpm=round(bpm, 1) if bpm is not None else None,
            sqi=round(float(metrics.get("SQI") or 0.0), 2),
            respiratory_rate=round(
                float(analyzer.estimate_respiratory_rate(bvp, model.fps) or 0.0), 1
            ),
            latency_ms=chunk_wall_ms,
            processing_speed=processing_speed,
        )
        chunk_data = {"type": "chunk_update", **asdict(chunk)}
        await self.websocket.send_json(DataSanitizer.sanitize(chunk_data))

    async def _finalize(
        self, last_start: float, last_ts: float, total_frames: int, session_start: float
    ):
        analyzer = get_analyzer()
        model = analyzer.get_model()

        # Send final partial chunk if significant
        if last_ts > last_start + 1.0:
            await self._process_window(last_start, last_ts, session_start)

        # Wait for AI inference to complete for full video
        logger.info(
            f"Awaiting final AI completion ({model.n_signal}/{total_frames})..."
        )
        final_wait_start = time.time()
        while model.n_signal < total_frames - 10:
            if time.time() - final_wait_start > 60.0:
                logger.warning("Final synchronization timeout reached.")
                break
            await asyncio.sleep(0.5)

        final_metrics = model.hr(return_hrv=False) or {}
        bvp, _ = model.bvp()

        # SQI-weighted median: filter out low-quality chunks (SQI < 0.3)
        SQI_THRESHOLD = 0.3
        filtered_bpms = [
            bpm
            for bpm, sqi in zip(self.chunk_bpms, self.chunk_sqis)
            if sqi >= SQI_THRESHOLD
        ]
        overall_bpm = (
            float(np.median(filtered_bpms))
            if filtered_bpms
            else (
                float(np.median(self.chunk_bpms))
                if self.chunk_bpms
                else final_metrics.get("hr")
            )
        )

        total_time = round(time.time() - session_start, 1)

        final_data = {
            "type": "final_result",
            "overall_bpm": round(overall_bpm, 1) if overall_bpm is not None else None,
            "overall_sqi": round(float(final_metrics.get("SQI") or 0.0), 2),
            "overall_respiratory_rate": round(
                float(analyzer.estimate_respiratory_rate(bvp, model.fps) or 0.0), 1
            ),
            "video_duration_sec": round(last_ts, 1),
            "total_processing_time_sec": total_time,
            "average_latency_ms": round(
                (
                    (time.time() - session_start)
                    / (last_ts / self.chunk_size_sec)
                    * 1000
                    if last_ts > 0
                    else 0.0
                ),
                1,
            ),
        }
        await self.websocket.send_json(DataSanitizer.sanitize(final_data))
        logger.info("Session finalized.")

    async def _fail(self, message: str):
        logger.error(f"Session failed: {message}")
        await self.websocket.send_json({"error": message})
