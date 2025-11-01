import cv2
import numpy as np
import mediapipe as mp
import pickle
from keras.models import load_model
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

class RecognitionWorker(QThread):
    """Worker thread for real-time ISL recognition"""
    frame_ready = pyqtSignal(QImage)
    prediction_ready = pyqtSignal(str, float)  # (action, confidence)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.model = None
        self.label_encoder = None
        self.cap = None
        self.MAX_SEQUENCE_LENGTH = 30
        self.sequence = []
        self.threshold = 0.5
        
        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
    def load_model_and_encoder(self):
        """Load the trained model and label encoder"""
        try:
            print("Attempting to load model from 'model/isl_model.keras'...")
            # Load model from model folder
            self.model = load_model('model/isl_model.keras')
            print("✓ Model loaded successfully!")
            
            print("Attempting to load label encoder from 'model/label_encoder.pkl'...")
            with open('model/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.actions = self.label_encoder.classes_
            print(f"✓ Label encoder loaded successfully! Actions: {self.actions}")
            return True
        except FileNotFoundError as e:
            error_msg = f"Model or Label Encoder not found in 'model/' folder.\nError: {str(e)}\nPlease ensure 'isl_model.keras' and 'label_encoder.pkl' exist in the 'model' directory."
            print(f"✗ FileNotFoundError: {error_msg}")
            self.error_occurred.emit(error_msg)
            return False
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}\nType: {type(e).__name__}"
            print(f"✗ Exception: {error_msg}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(error_msg)
            return False
    
    def mediapipe_detection(self, image, model):
        """Process image with MediaPipe"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    
    def draw_styled_landmarks(self, image, results):
        """Draw MediaPipe landmarks on image"""
        # Pose connections (Green)
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)
        )
        
        # Left hand connections (Blue)
        self.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1)
        )
        
        # Right hand connections (Red)
        self.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)
        )
    
    def extract_keypoints(self, results):
        """Extract keypoints from MediaPipe results"""
        pose = np.array([[res.x, res.y, res.z, res.visibility] 
                        for res in results.pose_landmarks.landmark]).flatten() \
               if results.pose_landmarks else np.zeros(33*4)
        
        face = np.array([[res.x, res.y, res.z] 
                        for res in results.face_landmarks.landmark]).flatten() \
               if results.face_landmarks else np.zeros(468*3)
        
        lh = np.array([[res.x, res.y, res.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten() \
             if results.left_hand_landmarks else np.zeros(21*3)
        
        rh = np.array([[res.x, res.y, res.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten() \
             if results.right_hand_landmarks else np.zeros(21*3)
        
        return np.concatenate([pose, face, lh, rh])

    def run(self):
        """Main recognition loop"""
        if not self.load_model_and_encoder():
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.error_occurred.emit("Could not open webcam")
            return
        
        self.running = True
        self.status_update.emit("Camera started")
        
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        ) as holistic:
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Mirror flip
                frame = cv2.flip(frame, 1)
                
                # Process with MediaPipe
                image, results = self.mediapipe_detection(frame, holistic)
                self.draw_styled_landmarks(image, results)
                
                # Extract keypoints and build sequence
                keypoints = self.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-self.MAX_SEQUENCE_LENGTH:]
                
                # Make prediction when sequence is full
                if len(self.sequence) == self.MAX_SEQUENCE_LENGTH:
                    input_data = np.expand_dims(self.sequence, axis=0)
                    prediction = self.model.predict(input_data, verbose=0)[0]
                    top_prediction_index = np.argmax(prediction)
                    confidence = prediction[top_prediction_index]
                    
                    if confidence > self.threshold:
                        predicted_action = self.actions[top_prediction_index]
                        self.prediction_ready.emit(predicted_action, confidence)
                    else:
                        self.prediction_ready.emit("Waiting...", 0.0)
                else:
                    self.prediction_ready.emit("Collecting frames...", 0.0)
                
                # Convert frame to QImage for display
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                self.frame_ready.emit(qt_image)
        
        self.cap.release()
        self.status_update.emit("Camera stopped")
    
    def stop(self):
        """Stop the recognition loop"""
        self.running = False
        self.wait()
