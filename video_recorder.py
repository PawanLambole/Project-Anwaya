"""
Video Recorder Module
Implements a robust video recording system with proper frame timing
Based on professional video recording architecture
"""

import cv2
import time
import threading
from queue import Queue, Empty
from PyQt5.QtCore import QObject, pyqtSignal


class VideoRecorder(QObject):
    """
    Professional video recorder with separate capture and encoding threads
    for smooth, real-time video recording at consistent frame rates.
    """
    
    # Signals for status updates
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal(str)  # Emits file path
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # System state
        self.state = "IDLE"  # IDLE, READY, RECORDING, STOPPED
        
        # Video source
        self.camera = None
        self.camera_index = 0
        
        # Recording parameters
        self.width = 640
        self.height = 480
        self.fps = 30.0
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # File handling
        self.file_path = None
        self.video_writer = None
        
        # Threading
        self.capture_thread = None
        self.frame_queue = Queue(maxsize=60)  # Buffer for smooth recording
        self.is_capturing = False
        
        # Frame timing
        self.frame_interval = 1.0 / self.fps  # Time between frames
        self.last_frame_time = 0
        self.frames_written = 0
        self.start_time = 0
        
    def initialize_recorder(self, camera_source=None):
        """
        Initialize the video recording system
        
        Args:
            camera_source: cv2.VideoCapture object or camera index
        
        Returns:
            bool: True if initialized successfully
        """
        try:
            # Set up camera input source
            if camera_source is not None:
                if isinstance(camera_source, cv2.VideoCapture):
                    self.camera = camera_source
                else:
                    self.camera_index = camera_source
                    self.camera = cv2.VideoCapture(self.camera_index)
            
            if self.camera is None or not self.camera.isOpened():
                self.error_occurred.emit("Could not open camera")
                return False
            
            # Get camera properties
            self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Try to get camera FPS, fallback to 30
            camera_fps = self.camera.get(cv2.CAP_PROP_FPS)
            if camera_fps > 0:
                self.fps = camera_fps
            else:
                self.fps = 30.0
            
            self.frame_interval = 1.0 / self.fps
            
            # Set state to READY
            self.state = "READY"
            print(f"[VideoRecorder] Initialized: {self.width}x{self.height} @ {self.fps:.1f} FPS")
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Initialization error: {str(e)}")
            return False
    
    def start_recording(self, file_path, duration_seconds=None):
        """
        Start the recording process
        
        Args:
            file_path: Path where video will be saved
            duration_seconds: Optional duration limit
        
        Returns:
            bool: True if recording started successfully
        """
        if self.state != "READY":
            self.error_occurred.emit("System not ready or already recording")
            return False
        
        try:
            # Open file for writing
            self.file_path = file_path
            self.video_writer = cv2.VideoWriter(
                self.file_path,
                self.fourcc,
                self.fps,
                (self.width, self.height)
            )
            
            if not self.video_writer.isOpened():
                self.error_occurred.emit("Could not open file for writing")
                return False
            
            # Initialize recording state
            self.state = "RECORDING"
            self.is_capturing = True
            self.frames_written = 0
            self.start_time = time.time()
            self.last_frame_time = self.start_time
            
            # Clear frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    break
            
            # Start capture and encode thread
            self.capture_thread = threading.Thread(
                target=self._capture_and_encode_loop,
                args=(duration_seconds,),
                daemon=True
            )
            self.capture_thread.start()
            
            print(f"[VideoRecorder] Recording started: {file_path}")
            self.recording_started.emit()
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Start recording error: {str(e)}")
            self.state = "READY"
            return False
    
    def _capture_and_encode_loop(self, duration_seconds=None):
        """
        Main loop for capturing and encoding frames
        Runs in separate thread for consistent timing
        
        Args:
            duration_seconds: Optional duration limit
        """
        print(f"[VideoRecorder] Capture loop started (duration: {duration_seconds}s)")
        
        while self.is_capturing and self.state == "RECORDING":
            try:
                current_time = time.time()
                
                # Check if duration limit reached
                if duration_seconds is not None:
                    elapsed = current_time - self.start_time
                    if elapsed >= duration_seconds:
                        print(f"[VideoRecorder] Duration limit reached: {elapsed:.2f}s")
                        break
                
                # Capture frame from camera
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    print("[VideoRecorder] Failed to capture frame")
                    time.sleep(0.001)
                    continue
                
                # Write frame at consistent intervals
                time_since_last_frame = current_time - self.last_frame_time
                
                if time_since_last_frame >= self.frame_interval:
                    # Write frame to video file
                    self.video_writer.write(frame)
                    self.frames_written += 1
                    self.last_frame_time = current_time
                    
                    # Optional: Add to queue for preview (not used currently)
                    # if not self.frame_queue.full():
                    #     self.frame_queue.put(frame.copy())
                
                # Small sleep to prevent CPU overuse
                time.sleep(0.001)
                
            except Exception as e:
                print(f"[VideoRecorder] Capture loop error: {e}")
                break
        
        # Auto-stop if loop ended
        if self.state == "RECORDING":
            self._finalize_recording()
    
    def stop_recording(self):
        """
        Stop the recording process
        
        Returns:
            str: Path to saved video file, or None if error
        """
        if self.state != "RECORDING":
            self.error_occurred.emit("Not currently recording")
            return None
        
        print("[VideoRecorder] Stopping recording...")
        self.is_capturing = False
        
        # Wait for capture thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Finalize and close file
        return self._finalize_recording()
    
    def _finalize_recording(self):
        """
        Finalize the recording and close file
        
        Returns:
            str: Path to saved video file
        """
        try:
            # Release video writer
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            # Calculate statistics
            elapsed = time.time() - self.start_time
            actual_fps = self.frames_written / elapsed if elapsed > 0 else 0
            
            print(f"[VideoRecorder] Recording saved: {self.file_path}")
            print(f"[VideoRecorder] Stats: {self.frames_written} frames in {elapsed:.2f}s ({actual_fps:.1f} FPS)")
            
            # Update state
            saved_path = self.file_path
            self.state = "READY"
            self.file_path = None
            
            # Emit signal
            self.recording_stopped.emit(saved_path)
            
            return saved_path
            
        except Exception as e:
            print(f"[VideoRecorder] Finalize error: {e}")
            self.error_occurred.emit(f"Finalize error: {str(e)}")
            self.state = "READY"
            return None
    
    def is_recording(self):
        """Check if currently recording"""
        return self.state == "RECORDING"
    
    def is_ready(self):
        """Check if ready to record"""
        return self.state == "READY"
    
    def get_state(self):
        """Get current state"""
        return self.state
    
    def release(self):
        """Release all resources"""
        if self.state == "RECORDING":
            self.stop_recording()
        
        self.state = "IDLE"
        
        # Note: Camera is managed externally, don't release it here
        # if self.camera:
        #     self.camera.release()
        #     self.camera = None
