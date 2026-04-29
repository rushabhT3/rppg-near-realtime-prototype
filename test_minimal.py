#!/usr/bin/env python3
"""Minimal test without pkg_resources dependency"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set Keras backend to JAX BEFORE importing keras
os.environ["KERAS_BACKEND"] = "jax"

# Test basic imports first
try:
    print("Testing basic imports...")
    import numpy as np
    import jax
    import keras

    print("✓ Core ML libraries imported")

    import cv2
    import scipy

    print("✓ CV/Signal processing libraries imported")

    print("\nTesting rPPG import...")
    import rppg

    print("✓ rPPG imported successfully!")

    print("\nTesting model creation...")
    model = rppg.Model()
    print("✓ Model created successfully!")

    print(f"Model info:")
    print(f"  - FPS: {model.fps}")
    print(f"  - Input shape: {model.input}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
