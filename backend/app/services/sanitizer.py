from typing import Any
import numpy as np


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
