#!/usr/bin/env python3
"""Direct test bypassing pkg_resources"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test if we can at least import the core components
try:
    print("Testing direct imports...")

    # Test if we can import the models directly
    from rppg.models import InfinitePulse

    print("✓ InfinitePulse imported")

    # Test JAX/Keras
    import jax
    import keras

    print("✓ JAX/Keras working")

    # Test if weights directory exists
    weights_dir = os.path.join(os.path.dirname(__file__), "rppg", "weights")
    if os.path.exists(weights_dir):
        weight_files = [f for f in os.listdir(weights_dir) if f.endswith(".h5")]
        print(f"✓ Found {len(weight_files)} weight files")
        print(f"  Sample: {weight_files[:3]}")
    else:
        print("❌ Weights directory not found")

    print("\n✓ Core components working!")
    print("The pkg_resources issue is just for file path resolution.")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
