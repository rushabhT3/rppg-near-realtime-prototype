#!/usr/bin/env python3
"""Test rPPG with video processing"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rppg
import numpy as np


def test_with_tensor():
    """Test with synthetic video data"""
    print("Testing with synthetic video tensor...")

    # Create a synthetic video (100 frames, 128x128, RGB)
    video_tensor = np.random.randint(0, 255, (100, 128, 128, 3), dtype="uint8")

    model = rppg.Model()

    try:
        # Process the synthetic video
        result = model.process_video_tensor(video_tensor, fps=30.0)
        print(f"✓ Video processed successfully")
        print(f"Result: {result}")

        if result and result.get("hr"):
            print(f"Estimated Heart Rate: {result['hr']:.1f} BPM")
        else:
            print("No valid heart rate detected (expected with random data)")

    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback

        traceback.print_exc()


def test_different_models():
    """Test different model architectures"""
    models_to_test = ["FacePhys.rlap", "PhysMamba.pure", "TSCAN.pure"]

    for model_name in models_to_test:
        try:
            print(f"\nTesting model: {model_name}")
            model = rppg.Model(model_name)
            print(f"✓ {model_name} loaded successfully")
            print(f"  FPS: {model.fps}, Input: {model.input}")
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")


if __name__ == "__main__":
    print("=== rPPG Video Processing Test ===")

    # Test different models
    test_different_models()

    # Test with synthetic data
    test_with_tensor()

    print("\n=== Test Complete ===")
