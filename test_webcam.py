#!/usr/bin/env python3
"""Test rPPG with real-time webcam"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rppg
import time
import cv2


def test_webcam():
    """Test real-time webcam processing"""
    print("Testing real-time webcam processing...")
    print("Press 'q' to quit")

    try:
        model = rppg.Model()

        # Open webcam (index 0 is default)
        with model.video_capture(0):
            last_process_time = 0
            current_hr = None
            frame_count = 0

            # Iterate through the preview generator
            for frame, box in model.preview:
                frame_count += 1

                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Calculate HR every 2 seconds
                now = time.time()
                if now - last_process_time > 2.0:
                    result = model.hr(start=-10)  # Last 10 seconds
                    if result and result["hr"]:
                        current_hr = result["hr"]
                        sqi = result.get("SQI", 0)
                        print(f"HR: {current_hr:.1f} BPM (SQI: {sqi:.2f})")
                    last_process_time = now

                # Draw face detection box
                if box is not None:
                    y1, y2 = box[0]
                    x1, x2 = box[1]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Display HR if available
                    if current_hr is not None:
                        cv2.putText(
                            frame,
                            f"HR: {current_hr:.1f} BPM",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                # Display frame info
                cv2.putText(
                    frame,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("rPPG Real-time Monitor", frame)

                # Quit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except Exception as e:
        print(f"❌ Error with webcam: {e}")
        import traceback

        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=== rPPG Webcam Test ===")
    test_webcam()
    print("=== Test Complete ===")
