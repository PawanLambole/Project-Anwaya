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

# Reverted model_complexity to 0 (Lite) to completely eliminate all lag.
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 0  # 0=Lite (Fastest), 1=Full, 2=Heavy

# Quality gate: if more than this ratio of sampled frames have zero hand landmarks,
# a quality_warning signal is emitted to the UI.
BAD_FRAME_THRESHOLD = 0.50  # 50%

# Hand keypoint slice indices in the 1662-element array
HAND_LH_START = 132 + 1404          # 1536
HAND_LH_END   = HAND_LH_START + 63  # 1599
HAND_RH_START = HAND_LH_END         # 1599
HAND_RH_END   = HAND_RH_START + 63  # 1662

# --- 2. MEDIAPIPE SETUP ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    # --- FAST Zero-Lag Low Light Enhancement ---
    # Instantly check brightness using a fast grayscale mean
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    # If the room is dark (value under 90)
    if avg_brightness < 90:
        # Boost contrast (alpha) and brightness (beta) completely instantly in C++
        alpha = 1.3  # Contrast control
        beta = 40    # Brightness control
        enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    else:
        enhanced_image = image
    
    # --- MediaPipe Processing ---
    image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    
    image_bgr_out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr_out, results

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

def _frame_has_no_hands(keypoints):
    """Return True if both left and right hand keypoints are all zeros (no hands detected)."""
    lh_kp = keypoints[HAND_LH_START:HAND_LH_END]
    rh_kp = keypoints[HAND_RH_START:HAND_RH_END]
    return np.all(lh_kp == 0) and np.all(rh_kp == 0)


# --- 3. PROCESSING THREAD ---
class ProcessingThread(QThread):
    finished = pyqtSignal(str)
    # NEW: emits (output_folder, missed_hand_frame_count) when quality is poor
    quality_warning = pyqtSignal(str, int)

    def __init__(self, video_path, output_folder, holistic_model):
        super().__init__()
        self.video_path = video_path
        self.output_folder = output_folder
        self.holistic_model = holistic_model  # Note: We create a new one in-thread

    def run(self):
        """
        Process video file to extract keypoint sequences.
        Uses model_complexity=2 (Heavy) for maximum landmark accuracy.
        Emits quality_warning if >50% of sampled frames have no hand detections.
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
        
        # Use Heavy model (model_complexity=2) for maximum accuracy during offline processing.
        # This is slower than the live camera model but produces far better keypoints.
        with mp_holistic.Holistic(
            static_image_mode=True,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            model_complexity=MODEL_COMPLEXITY
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
        print(f"  [Thread] Extracted {frame_count} raw frames from video")
        
        # Handle empty video
        if not all_video_keypoints_raw or frame_count == 0: 
            print(f"  [Thread] WARNING: No keypoints extracted from {self.video_path}.")
            print(f"  [Thread] Creating {SEQUENCE_LENGTH} empty landmark files as fallback...")
            for j in range(SEQUENCE_LENGTH):
                npy_path = os.path.join(self.output_folder, f"{j}.npy")
                np.save(npy_path, np.zeros(KEYPOINT_SIZE))
            self.quality_warning.emit(self.output_folder, SEQUENCE_LENGTH)
            self.finished.emit(self.output_folder)
            return 

        num_frames = len(all_video_keypoints_raw)
        
        # --- Smart Cropping: Find active sign boundaries ---
        valid_indices = [i for i, kp in enumerate(all_video_keypoints_raw) if not _frame_has_no_hands(kp)]
        
        first_valid_idx = 0
        last_valid_idx = num_frames - 1
        
        if valid_indices:
            first_valid_idx = valid_indices[0]
            last_valid_idx = valid_indices[-1]
            # Extra safety padding (add 1 frame of context on edges if possible)
            first_valid_idx = max(0, first_valid_idx - 1)
            last_valid_idx = min(num_frames - 1, last_valid_idx + 1)
        
        # Sample frames evenly across ONLY the active video portion (normalizes frame rate and drops dead space)
        indices = np.linspace(first_valid_idx, last_valid_idx, SEQUENCE_LENGTH, dtype=int)
        
        # Save sampled keypoints and count bad frames (no hands detected)
        missed_hand_frames = 0
        for j, frame_index in enumerate(indices):
            keypoints_to_save = all_video_keypoints_raw[frame_index]
            
            # Quality check: count frames where no hands were detected
            if _frame_has_no_hands(keypoints_to_save):
                missed_hand_frames += 1

            npy_path = os.path.join(self.output_folder, f"{j}.npy")
            np.save(npy_path, keypoints_to_save)
        
        # --- Post-save quality validation ---
        bad_ratio = missed_hand_frames / SEQUENCE_LENGTH
        if bad_ratio > BAD_FRAME_THRESHOLD:
            print(f"  [Thread] ⚠️  Quality Warning: {missed_hand_frames}/{SEQUENCE_LENGTH} sampled frames had NO hand detection ({bad_ratio*100:.0f}%). This video may produce bad training data.")
            self.quality_warning.emit(self.output_folder, missed_hand_frames)
        else:
            print(f"  [Thread] ✓ Quality OK: Only {missed_hand_frames}/{SEQUENCE_LENGTH} frames had missing hands ({bad_ratio*100:.0f}%).")

        print(f"  [Thread] ✓ Successfully saved {SEQUENCE_LENGTH} landmark files to {self.output_folder}/")
        self.finished.emit(self.output_folder)