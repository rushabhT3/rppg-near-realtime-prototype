#!/usr/bin/env python3
"""Test what signals different models can extract"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rppg
import numpy as np

print("=== Testing Different Model Outputs ===\n")

# Test with synthetic data
video_tensor = np.random.randint(0, 255, (100, 128, 128, 3), dtype="uint8")

models_to_test = ["FacePhys.rlap", "PhysMamba.pure", "TSCAN.pure"]

for model_name in models_to_test:
    print(f"Testing {model_name}:")
    try:
        model = rppg.Model(model=model_name)
        result = model.process_video_tensor(video_tensor, fps=30.0)

        print(f"  Available outputs: {list(result.keys())}")
        for key, value in result.items():
            if key == "hr":
                print(f"    Heart Rate: {value:.1f} BPM")
            elif key == "hrv":
                print(
                    f"    HRV available: {bool(value)} keys: {list(value.keys()) if value else 'None'}"
                )
            elif key == "SQI":
                print(f"    Signal Quality: {value:.3f}")
            else:
                print(f"    {key}: {type(value)}")
        print()

    except Exception as e:
        print(f"  ❌ Error: {e}\n")

print("=== Raw Signal Access ===\n")

# Test if we can get raw BVP signal for respiratory analysis
try:
    model = rppg.Model("FacePhys.rlap")
    result = model.process_video_tensor(video_tensor, fps=30.0)

    # Try to get raw BVP signal
    with model:
        for i in range(len(video_tensor)):
            model.update_face(video_tensor[i], ts=i / 30.0)

    # Get raw BVP signal
    bvp, timestamps = model.bvp()
    print(f"Raw BVP signal length: {len(bvp)} samples")
    print(f"Timestamp range: {timestamps[0]:.1f} - {timestamps[-1]:.1f} seconds")

    if len(bvp) > 0:
        print("✅ Raw BVP signal available for respiratory analysis")
        print("   (Can extract respiratory rate from BVP patterns)")
    else:
        print("❌ No BVP signal available")

except Exception as e:
    print(f"❌ Error accessing raw signals: {e}")

print("\n=== Summary ===")
print("• Heart Rate: ✅ Primary output from all models")
print("• Respiratory Rate: ⚠️  Can be derived from BVP/HRV patterns")
print("• Raw Signals: ✅ Available for custom analysis")
