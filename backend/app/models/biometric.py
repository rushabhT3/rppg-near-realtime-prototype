from dataclasses import dataclass, asdict
from typing import Optional


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
