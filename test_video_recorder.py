"""
Test script for the VideoRecorder class
"""

import cv2
import time
from video_recorder import VideoRecorder

def test_video_recorder():
    """Test the VideoRecorder with a simple recording"""
    
    print("=" * 50)
    print("VideoRecorder Test")
    print("=" * 50)
    
    # Create recorder
    recorder = VideoRecorder()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return False
    
    print("\n1. Initializing recorder...")
    success = recorder.initialize_recorder(cap)
    if not success:
        print("   FAILED to initialize")
        cap.release()
        return False
    print("   ✓ Initialized successfully")
    print(f"   State: {recorder.get_state()}")
    
    print("\n2. Starting 3-second recording...")
    test_file = "test_recording.mp4"
    success = recorder.start_recording(test_file, duration_seconds=3)
    if not success:
        print("   FAILED to start recording")
        cap.release()
        return False
    print("   ✓ Recording started")
    print(f"   State: {recorder.get_state()}")
    
    # Wait for recording to complete (auto-stops after 3 seconds)
    print("\n3. Recording in progress...")
    for i in range(3):
        time.sleep(1)
        print(f"   {i+1} seconds...")
    
    # Give a bit more time for the thread to finalize
    time.sleep(0.5)
    
    print(f"\n4. Final state: {recorder.get_state()}")
    
    # Clean up
    recorder.release()
    cap.release()
    
    print("\n5. Checking if file exists...")
    import os
    if os.path.exists(test_file):
        file_size = os.path.getsize(test_file)
        print(f"   ✓ File created: {test_file}")
        print(f"   ✓ File size: {file_size / 1024:.1f} KB")
        
        # Try to read the video
        print("\n6. Verifying video file...")
        test_cap = cv2.VideoCapture(test_file)
        if test_cap.isOpened():
            frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = test_cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            print(f"   ✓ Video readable")
            print(f"   ✓ Frames: {frame_count}")
            print(f"   ✓ FPS: {fps:.1f}")
            print(f"   ✓ Duration: {duration:.2f} seconds")
            test_cap.release()
        else:
            print("   ✗ Could not read video file")
            return False
    else:
        print("   ✗ File not created")
        return False
    
    print("\n" + "=" * 50)
    print("✓ ALL TESTS PASSED")
    print("=" * 50)
    return True

if __name__ == "__main__":
    try:
        test_video_recorder()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
