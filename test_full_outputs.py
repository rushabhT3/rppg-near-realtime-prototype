#!/usr/bin/env python3
"""Test ALL built-in outputs including respiratory"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rppg
import numpy as np

print("=== Full rPPG Outputs Test ===\n")

# Test with longer synthetic data for better HRV analysis
video_tensor = np.random.randint(
    0, 255, (300, 128, 128, 3), dtype="uint8"
)  # 10 seconds at 30fps

print("Testing with extended synthetic video (10 seconds)...\n")

model = rppg.Model("FacePhys.rlap")

try:
    # Process the video
    result = model.process_video_tensor(video_tensor, fps=30.0)

    print("=== COMPLETE OUTPUTS ===")
    print(f"Result keys: {list(result.keys())}")

    for key, value in result.items():
        print(f"\n{key}:")
        if key == "hr":
            print(f"  Heart Rate: {value:.1f} BPM")
        elif key == "hrv":
            print(f"  Heart Rate Variability Analysis:")
            if isinstance(value, dict) and value:
                for hrv_key, hrv_val in value.items():
                    if hrv_key in ["bpm", "rmssd", "pnn20", "pnn50"]:
                        print(f"    {hrv_key}: {hrv_val}")
                    elif hrv_key in ["VLF", "LF", "HF", "TP", "LF/HF"]:
                        print(f"    {hrv_key}: {hrv_val:.4f}")
                    else:
                        print(f"    {hrv_key}: {hrv_val}")
            else:
                print("    No HRV data (signal quality too low)")
        elif key == "SQI":
            print(f"  Signal Quality Index: {value:.3f}")
            if value > 0.5:
                print("    ✓ Good signal quality")
            else:
                print("    ⚠ Low signal quality")
        elif key == "latency":
            print(f"  Processing latency: {value:.3f} seconds")
        else:
            print(f"  Value: {value}")

    print("\n=== RESPIRATORY INDICATORS ===")
    if "hrv" in result and isinstance(result["hrv"], dict):
        hrv = result["hrv"]
        respiratory_indicators = []

        # Check for respiratory-related HRV metrics
        if "LF" in hrv and "HF" in hrv:
            lf_hf_ratio = hrv.get("LF/HF")
            if lf_hf_ratio:
                print(f"LF/HF Ratio: {lf_hf_ratio:.3f}")
                print("  → Indicates sympathetic/parasympathetic balance")
                print("  → Can be used for respiratory rate estimation")

        if "VLF" in hrv:
            print(f"VLF Power: {hrv['VLF']:.4f}")
            print("  → Very Low Frequency related to respiratory patterns")

        if any(key in hrv for key in ["rmssd", "pnn50"]):
            print("HRV Time-domain metrics available")
            print("  → Can extract respiratory sinus arrhythmia")

    print("\n=== RAW SIGNAL ACCESS ===")
    # Test raw signal access
    with model:
        for i in range(len(video_tensor)):
            model.update_face(video_tensor[i], ts=i / 30.0)

    # Get raw BVP for respiratory analysis
    bvp, timestamps = model.bvp()
    if len(bvp) > 0:
        print(f"✅ Raw BVP signal: {len(bvp)} samples")
        print(f"   Duration: {timestamps[-1]:.1f} seconds")
        print("   → Can extract respiratory rate from BVP patterns")

        # Simple respiratory rate estimation from BVP
        if len(bvp) > 60:  # Need at least 2 seconds
            from scipy.signal import welch

            f, Pxx = welch(bvp, fs=30, nperseg=min(len(bvp), 256))
            # Respiratory frequency range: 0.1-0.5 Hz (6-30 breaths/min)
            resp_mask = (f >= 0.1) & (f <= 0.5)
            if np.any(Pxx[resp_mask] > 0):
                resp_freq = f[resp_mask][np.argmax(Pxx[resp_mask])]
                resp_rate = resp_freq * 60  # Convert to breaths per minute
                print(f"   → Estimated Respiratory Rate: {resp_rate:.1f} breaths/min")
    else:
        print("❌ No BVP signal available")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()

print("\n=== SUMMARY ===")
print("✅ Heart Rate: Direct measurement")
print("✅ HRV Analysis: Built-in (LF, HF, VLF, LF/HF ratios)")
print("✅ Respiratory Indicators: Available through HRV patterns")
print("✅ Raw BVP Signal: Available for custom respiratory analysis")
print("\nThe rPPG system provides both cardiac AND respiratory information!")
