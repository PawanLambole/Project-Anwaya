import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
from keras.models import load_model
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS   # ~0.0333 s

class RecognitionWorker(QThread):
    """Worker thread for real-time ISL recognition"""
    frame_ready = pyqtSignal(QImage)
    prediction_ready = pyqtSignal(str, float)  # (action, confidence)
    word_committed = pyqtSignal(str)            # emitted when a word is confirmed
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, cam_index=0, dataset_name="Default"):
        super().__init__()
        self.cam_index = cam_index
        self.dataset_name = dataset_name
        self.running = False
        self.model = None
        self.label_encoder = None
        self.cap = None
        self._current_frame = None  # Hold reference to prevent GC
        self.MAX_SEQUENCE_LENGTH = 30
        self.sequence = []
        self.threshold = 0.75          # minimum confidence to count

        # --- Word commit state ---
        self.CONSECUTIVE_NEEDED = 3    # same word must appear N times in a row
        self.IDLE_TO_COMMIT   = 2.0   # seconds of idle before committing
        self._consec_word     = None   # word being tracked
        self._consec_count    = 0      # how many times seen in a row
        self._candidate_word  = None   # word ready to be committed
        self._idle_since      = None   # when idle period started
        
        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
    def load_model_and_encoder(self):
        """Load the trained model and label encoder"""
        
        model_path = f'model/{self.dataset_name}/{self.dataset_name}_model.keras'
        encoder_path = f'model/{self.dataset_name}/{self.dataset_name}_label_encoder.pkl'
        
        try:
            print(f"Attempting to load model from '{model_path}'...")
            self.model = load_model(model_path)
            print("✓ Model loaded successfully!")
            
            print(f"Attempting to load label encoder from '{encoder_path}'...")
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.actions = self.label_encoder.classes_
            print(f"✓ Label encoder loaded successfully! Actions: {self.actions}")
            return True
        except FileNotFoundError as e:
            error_msg = f"Model or Label Encoder not found for dataset '{self.dataset_name}'.\nError: {str(e)}\nPlease ensure '{self.dataset_name}_model.keras' and '{self.dataset_name}_label_encoder.pkl' exist in the 'model/{self.dataset_name}' directory. You may need to train a model for this dataset first."
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

    def _emit_frame(self, frame):
        """Convert a BGR frame to QImage and emit it to the UI."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        self._current_frame = rgb_image
        qt_image = QImage(
            self._current_frame.data, w, h, bytes_per_line,
            QImage.Format_RGB888
        ).copy()
        self.frame_ready.emit(qt_image)

    def run(self):
        """Main recognition loop"""
        # ── Step 1: Open camera FIRST so LED turns on immediately ──────────
        self.status_update.emit("Opening camera...")

        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("CAP_DSHOW failed, trying default backend...")
            self.cap = cv2.VideoCapture(self.cam_index)

        if not self.cap.isOpened():
            self.error_occurred.emit(f"Could not open webcam {self.cam_index}.")
            return

        # Warm-up: Windows cameras return black frames on the first few reads
        print("Warming up camera (discarding initial frames)...")
        for _ in range(10):
            self.cap.read()

        # Reduce the camera driver buffer so we always get the LATEST frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        self.status_update.emit("Camera started")
        print("Camera opened successfully.")

        # ── Step 2: Load model WHILE streaming live preview to avoid black screen ──
        # We load the model in a background thread so the camera feed stays alive.
        self.status_update.emit("Loading model, please wait...")
        self._model_loaded = False
        self._model_load_failed = False

        import threading as _threading

        def _load_model_bg():
            ok = self.load_model_and_encoder()
            if ok:
                self._model_loaded = True
            else:
                self._model_load_failed = True

        model_thread = _threading.Thread(target=_load_model_bg, daemon=True)
        model_thread.start()

        # Stream raw frames while the model loads so UI shows live video
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            # ── Preview loop (model still loading) ─────────────────────────
            while self.running and not self._model_loaded and not self._model_load_failed:
                t0 = time.monotonic()

                ret, frame = self.cap.read()
                if not ret:
                    print("[RecognitionWorker] cap.read() returned False during preview.")
                    break
                frame = cv2.flip(frame, 1)
                # Show a plain preview — no MediaPipe yet (keeps it fast)
                self._emit_frame(frame)

                # Sleep for the remainder of the frame interval to cap at TARGET_FPS
                elapsed = time.monotonic() - t0
                sleep_time = FRAME_INTERVAL - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if not self.running or self._model_load_failed:
                self.cap.release()
                self.status_update.emit("Camera stopped")
                return

            self.status_update.emit("Recognition active")
            print("Starting recognition loop...")

            # ── Main recognition loop (model is ready) ──────────────────────
            while self.running:
                t0 = time.monotonic()

                # Drain any buffered/stale frames so we always process the latest
                # This prevents the jitter caused by MediaPipe being slower than
                # the camera's capture rate, which queues up old frames.
                for _ in range(2):
                    self.cap.grab()

                ret, frame = self.cap.read()
                if not ret:
                    print("[RecognitionWorker] cap.read() returned False, stopping.")
                    break

                # Mirror flip
                frame = cv2.flip(frame, 1)

                # Process with MediaPipe
                image, results = self.mediapipe_detection(frame, holistic)
                self.draw_styled_landmarks(image, results)

                # ── Motion/Hand Gate ──────────────────────────────────────────
                # Only collect keypoints when the person is actually signing.
                # Criteria: at least one hand landmark detected  OR
                #           significant wrist movement compared to last frame.
                hands_visible = (results.left_hand_landmarks is not None or
                                 results.right_hand_landmarks is not None)

                # Compute wrist movement from pose landmarks (L wrist=15, R wrist=16)
                wrist_moved = False
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    # Use left+right wrist normalised coords for motion delta
                    curr_wrists = np.array([
                        lm[15].x, lm[15].y,   # left wrist
                        lm[16].x, lm[16].y    # right wrist
                    ])
                    if hasattr(self, '_prev_wrists'):
                        delta = np.linalg.norm(curr_wrists - self._prev_wrists)
                        wrist_moved = delta > 0.015   # ~1.5% of frame width
                    self._prev_wrists = curr_wrists

                is_active = hands_visible or wrist_moved

                if is_active:
                    self._last_active_time = time.monotonic()

                    # ── Key fix: don't reset the idle/commit timer once a
                    # candidate is locked. Micro-movements while the user
                    # holds still would otherwise restart the 2s countdown.
                    if self._candidate_word is None:
                        self._idle_since = None

                    keypoints = self.extract_keypoints(results)
                    self.sequence.append(keypoints)
                    self.sequence = self.sequence[-self.MAX_SEQUENCE_LENGTH:]

                    # --- Make prediction when sequence is full ---
                    if len(self.sequence) == self.MAX_SEQUENCE_LENGTH:
                        input_data = np.expand_dims(self.sequence, axis=0)
                        pred = self.model(input_data, training=False).numpy()[0]
                        top_idx    = np.argmax(pred)
                        confidence = pred[top_idx]

                        if confidence >= self.threshold:
                            predicted_action = self.actions[top_idx]

                            # --- Consecutive stability check ---
                            if predicted_action == self._consec_word:
                                self._consec_count += 1
                            else:
                                # Different word — restart count
                                self._consec_word  = predicted_action
                                self._consec_count = 1
                                # If user switches to a fully different sign,
                                # cancel the current candidate
                                if self._candidate_word is not None:
                                    self._candidate_word = None
                                    self._idle_since     = None

                            if self._consec_count >= self.CONSECUTIVE_NEEDED:
                                # Candidate locked
                                if self._candidate_word != predicted_action:
                                    # Newly locked — start idle timer now
                                    self._candidate_word = predicted_action
                                    self._idle_since     = time.monotonic()
                                self.prediction_ready.emit(
                                    f"🟡 {predicted_action}  (confirmed — pause to commit)",
                                    float(confidence)
                                )
                            else:
                                # Still building up consecutive count
                                self.prediction_ready.emit(
                                    f"{predicted_action}  [{self._consec_count}/{self.CONSECUTIVE_NEEDED}]",
                                    float(confidence)
                                )
                        else:
                            # Confidence too low — only cancel candidate if no longer visible
                            if self._candidate_word is None:
                                self._consec_word  = None
                                self._consec_count = 0
                            self.prediction_ready.emit("Low confidence...", float(confidence))
                    else:
                        self.prediction_ready.emit(
                            f"Collecting frames... ({len(self.sequence)}/{self.MAX_SEQUENCE_LENGTH})",
                            0.0
                        )

                else:
                    # --- IDLE branch ---
                    now = time.monotonic()
                    if self._idle_since is None:
                        self._idle_since = now

                    idle_secs = now - self._idle_since

                    if self._candidate_word and idle_secs >= self.IDLE_TO_COMMIT:
                        # ✅ COMMIT the word
                        self.word_committed.emit(self._candidate_word)
                        self.prediction_ready.emit(
                            f"✅ Committed: {self._candidate_word}", 0.0
                        )
                        # Reset everything
                        self._candidate_word = None
                        self._consec_word    = None
                        self._consec_count   = 0
                        self.sequence        = []
                        self._idle_since     = None
                    elif idle_secs > 1.5 and self._candidate_word is None:
                        # Long idle with no candidate — just reset sequence
                        self.sequence = []
                        self.prediction_ready.emit("Waiting for sign...", 0.0)
                    elif self._candidate_word:
                        # Counting down to commit — show stable countdown
                        remaining = max(0.0, self.IDLE_TO_COMMIT - idle_secs)
                        self.prediction_ready.emit(
                            f"🟡 {self._candidate_word}  — committing in {remaining:.1f}s",
                            0.0
                        )
                    else:
                        self.prediction_ready.emit("Waiting for sign...", 0.0)

                # Emit frame to UI
                self._emit_frame(image)

                # Cap to TARGET_FPS — prevents flooding the Qt UI thread
                elapsed = time.monotonic() - t0
                sleep_time = FRAME_INTERVAL - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        self.cap.release()
        self.status_update.emit("Camera stopped")

    def stop(self):
        """Stop the recognition loop"""
        self.running = False
        self.wait()
