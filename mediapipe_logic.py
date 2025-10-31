import cv2
import numpy as np
import os
import mediapipe as mp
from PyQt5.QtCore import QThread, pyqtSignal

# --- 1. SCRIPT PARAMETERS ---
DATA_PATH = os.path.join('ISL_Data')
OUTPUT_PATH = os.path.join('ISL_Processed')
SEQUENCE_LENGTH = 30
RECORD_SECONDS = 3
KEYPOINT_SIZE = 1662

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

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
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
        print(f"  [Thread] Processing {self.video_path}...")
        cap_proc = cv2.VideoCapture(self.video_path)
        all_video_keypoints_raw = []

        while(cap_proc.isOpened()):
            ret, frame = cap_proc.read()
            if not ret:
                break 
            # Create a new holistic model instance *within the thread*
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as thread_holistic:
                image, results = mediapipe_detection(frame, thread_holistic)
                keypoints = extract_keypoints(results)
                all_video_keypoints_raw.append(keypoints)
        cap_proc.release()
        
        if not all_video_keypoints_raw: 
            print(f"  [Thread] Warning: No keypoints from {self.video_path}.")
            for j in range(SEQUENCE_LENGTH):
                npy_path = os.path.join(self.output_folder, f"{j}.npy")
                np.save(npy_path, np.zeros(KEYPOINT_SIZE))
            self.finished.emit(self.output_folder)
            return 

        num_frames = len(all_video_keypoints_raw)
        indices = np.linspace(0, num_frames - 1, SEQUENCE_LENGTH, dtype=int)
        
        for j, frame_index in enumerate(indices):
            keypoints_to_save = all_video_keypoints_raw[frame_index]
            npy_path = os.path.join(self.output_folder, f"{j}.npy")
            np.save(npy_path, keypoints_to_save)
        
        print(f"  [Thread] Successfully saved {SEQUENCE_LENGTH} landmark files.")
        self.finished.emit(self.output_folder)