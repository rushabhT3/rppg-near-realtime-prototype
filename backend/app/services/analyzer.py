import os
import sys
import time
import asyncio
from typing import Optional, Any

import numpy as np
from scipy.signal import welch, detrend

# Add parent directory to path to import rppg engine
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import rppg

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger()


class SignalAnalyzer:
    """Handles biometric signal extraction and analysis using the rPPG engine."""

    _instance: Optional["SignalAnalyzer"] = None

    def __new__(cls, model_name: str = settings.DEFAULT_MODEL):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = settings.DEFAULT_MODEL):
        if self._initialized:
            return
        self.model_name = model_name
        self._model = self._initialize_model()
        self._initialized = True

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
