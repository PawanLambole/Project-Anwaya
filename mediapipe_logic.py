import cv2
import numpy as np
import os
import mediapipe as mp
from PyQt5.QtCore import QThread, pyqtSignal

# --- 1. SCRIPT PARAMETERS ---
DATA_PATH = os.path.join('ISL_Data')
OUTPUT_PATH = os.path.join('ISL_Processed')
SEQUENCE_LENGTH = 30  # Number of frames per sequence
RECORD_SECONDS = 3     # Duration of each recording
KEYPOINT_SIZE = 1662   # Total features: 132(pose) + 1404(face) + 63(left_hand) + 63(right_hand)

# MediaPipe model quality settings
MIN_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for detection
MIN_TRACKING_CONFIDENCE = 0.5   # Minimum confidence for tracking
MODEL_COMPLEXITY = 1            # 0=Lite, 1=Full, 2=Heavy

# --- 2. MEDIAPIPE SETUP ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1)) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1))

def check_detection_quality(results):
    """
    Check the quality of MediaPipe detection results.
    Returns tuple: (has_hands, has_pose, quality_score)
    """
    has_hands = bool(results.left_hand_landmarks or results.right_hand_landmarks)
    has_pose = bool(results.pose_landmarks)
    
    # Calculate quality score based on detected landmarks
    quality_score = 0
    if results.pose_landmarks:
        quality_score += 0.3
    if results.face_landmarks:
        quality_score += 0.2
    if results.left_hand_landmarks:
        quality_score += 0.25
    if results.right_hand_landmarks:
        quality_score += 0.25
    
    return has_hands, has_pose, quality_score

def extract_keypoints(results):
    """
    Extract and flatten keypoints from MediaPipe Holistic results.
    Returns a 1662-element array with pose, face, and hand landmarks.
    """
    # Pose: 33 landmarks × 4 (x, y, z, visibility) = 132 features
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                    for res in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33*4)
    
    # Face: 468 landmarks × 3 (x, y, z) = 1404 features  
    face = np.array([[res.x, res.y, res.z] 
                    for res in results.face_landmarks.landmark]).flatten() \
           if results.face_landmarks else np.zeros(468*3)
    
    # Left Hand: 21 landmarks × 3 (x, y, z) = 63 features
    lh = np.array([[res.x, res.y, res.z] 
                  for res in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21*3)
    
    # Right Hand: 21 landmarks × 3 (x, y, z) = 63 features
    rh = np.array([[res.x, res.y, res.z] 
                  for res in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21*3)
    
    # Total: 132 + 1404 + 63 + 63 = 1662 features
    return np.concatenate([pose, face, lh, rh])

# --- 3. PROCESSING THREAD ---
class ProcessingThread(QThread):
    finished = pyqtSignal(str)
    def __init__(self, video_path, output_folder, holistic_model):
        super().__init__()
        self.video_path = video_path
        self.output_folder = output_folder
        self.holistic_model = holistic_model # Note: We create a new one in-thread

    def run(self):
        """
        Process video file to extract keypoint sequences.
        Improved with better error handling and optimized MediaPipe usage.
        """
        print(f"  [Thread] Processing {self.video_path}...")
        cap_proc = cv2.VideoCapture(self.video_path)
        
        if not cap_proc.isOpened():
            print(f"  [Thread] ERROR: Could not open video file: {self.video_path}")
            # Create empty files as fallback
            for j in range(SEQUENCE_LENGTH):
                npy_path = os.path.join(self.output_folder, f"{j}.npy")
                np.save(npy_path, np.zeros(KEYPOINT_SIZE))
            self.finished.emit(self.output_folder)
            return
        
        all_video_keypoints_raw = []
        frame_count = 0
        
        # Create MediaPipe model once for the thread (more efficient)
        with mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            model_complexity=1  # 0=Lite, 1=Full, 2=Heavy (Full is good balance)
        ) as thread_holistic:
            
            while cap_proc.isOpened():
                ret, frame = cap_proc.read()
                if not ret:
                    break
                
                # Process frame with MediaPipe
                image, results = mediapipe_detection(frame, thread_holistic)
                keypoints = extract_keypoints(results)
                all_video_keypoints_raw.append(keypoints)
                frame_count += 1
        
        cap_proc.release()
        print(f"  [Thread] Extracted {frame_count} frames from video")
        
        # Check if we got any keypoints
        if not all_video_keypoints_raw or frame_count == 0: 
            print(f"  [Thread] WARNING: No keypoints extracted from {self.video_path}.")
            print(f"  [Thread] Creating {SEQUENCE_LENGTH} empty landmark files as fallback...")
            for j in range(SEQUENCE_LENGTH):
                npy_path = os.path.join(self.output_folder, f"{j}.npy")
                np.save(npy_path, np.zeros(KEYPOINT_SIZE))
            self.finished.emit(self.output_folder)
            return 

        num_frames = len(all_video_keypoints_raw)
        
        # Sample frames evenly across the video to get exactly SEQUENCE_LENGTH frames
        if num_frames >= SEQUENCE_LENGTH:
            # Use linspace for even sampling
            indices = np.linspace(0, num_frames - 1, SEQUENCE_LENGTH, dtype=int)
        else:
            # If video has fewer frames than needed, repeat frames to fill
            print(f"  [Thread] WARNING: Video has only {num_frames} frames, expected {SEQUENCE_LENGTH}")
            indices = np.linspace(0, num_frames - 1, SEQUENCE_LENGTH, dtype=int)
        
        # Save sampled keypoints
        for j, frame_index in enumerate(indices):
            keypoints_to_save = all_video_keypoints_raw[frame_index]
            npy_path = os.path.join(self.output_folder, f"{j}.npy")
            np.save(npy_path, keypoints_to_save)
        
        print(f"  [Thread] ✓ Successfully saved {SEQUENCE_LENGTH} landmark files to {self.output_folder}/")
        self.finished.emit(self.output_folder)