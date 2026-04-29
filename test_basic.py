#!/usr/bin/env python3
"""Basic test script for rPPG backend"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing rPPG import...")
    import rppg

    print("✓ rPPG imported successfully")

    print("\nTesting model initialization...")
    model = rppg.Model()
    print("✓ Model initialized successfully")

    print("\nTesting model info...")
    print(f"Model FPS: {model.fps}")
    print(f"Model input shape: {model.input}")
    print(f"Available models: {len(rppg.supported_models)} models")

    print("\n✓ All basic tests passed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
