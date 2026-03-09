#!/usr/bin/env python3
"""
Script to identify which /dev/video device corresponds to which physical camera.
Press 's' to save a screenshot, 'q' to move to the next camera, 'Esc' to exit.
"""

import cv2
import os
import glob
import numpy as np
import time

def list_video_devices():
    """List all available video devices."""
    devices = []
    for device in glob.glob('/dev/video*'):
        # Extract device number
        num = int(device.replace('/dev/video', ''))
        devices.append((num, device))
    devices.sort()
    return devices

def is_valid_color_image(frame):
    """Check if frame is a valid color image."""
    if frame is None or frame.size == 0:
        return False
    
    # Check if image has color (not grayscale)
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        return False
    
    # Check if image has meaningful variance (not all black/white)
    std_dev = np.std(frame)
    if std_dev < 5:  # Very low variance indicates blank image
        return False
    
    return True

def test_camera(device_path):
    """Test a camera device and display its feed."""
    print(f"\n{'='*60}")
    print(f"Testing: {device_path}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"❌ Cannot open {device_path}")
        return True  # Continue to next camera instead of stopping
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"✓ Camera opened successfully")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    
    # Try to read a few frames to check validity
    valid_frame = None
    print(f"  Checking for valid color images...")
    for i in range(10):  # Try up to 10 frames
        ret, frame = cap.read()
        if ret and is_valid_color_image(frame):
            valid_frame = frame
            break
        time.sleep(0.1)
    
    if valid_frame is None:
        print(f"❌ No valid color images detected (likely metadata stream)")
        cap.release()
        return True  # Continue to next camera
    
    print(f"✓ Valid color image detected!")
    
    print(f"\nControls:")
    print(f"  - Press 's' to save a screenshot")
    print(f"  - Press 'q' or 'n' to test next camera")
    print(f"  - Press 'Esc' to exit")
    
    window_name = f"Camera Test: {device_path}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print(f"❌ Failed to read frame from {device_path}")
            break
        
        if not is_valid_color_image(frame):
            continue
            
        frame_count += 1
        
        # Add text overlay
        cv2.putText(frame, f"Device: {device_path}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Resolution: {width}x{height}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' for next, 'Esc' to exit", (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('n'):
            print("Moving to next camera...")
            break
        elif key == 27:  # Esc
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            return False
        elif key == ord('s'):
            device_num = device_path.replace('/dev/video', '')
            filename = f"camera_video{device_num}_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def main():
    print("Camera Device Identifier")
    print("=" * 60)
    
    devices = list_video_devices()
    
    if not devices:
        print("No video devices found!")
        return
    
    print(f"\nFound {len(devices)} video device(s):")
    for num, path in devices:
        print(f"  - {path}")
    
    print("\nStarting camera testing...")
    print("Note: Press 's' to save a screenshot of any camera.")
    print("Metadata streams will be automatically skipped.\n")
    
    for num, device_path in devices:
        continue_testing = test_camera(device_path)
        if not continue_testing:
            break
    
    print("\n" + "=" * 60)
    print("Camera identification complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
