#!/usr/bin/env python3
"""Test JAX backend specifically"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force JAX backend
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

try:
    print("Testing JAX backend...")
    import jax

    print(f"✓ JAX version: {jax.__version__}")

    import keras

    print(f"✓ Keras backend: {keras.config.backend()}")

    # Test simple JAX operation
    import numpy as np

    x = jax.numpy.array([1, 2, 3])
    print(f"✓ JAX operation works: {x}")

    # Test Keras with JAX
    from keras import layers, ops

    dense = layers.Dense(10)
    print("✓ Keras layer created with JAX backend")

    print("\n✓ JAX + Keras working correctly!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
