import cv2
import numpy as np
import os
import time
import sys
import subprocess # <-- NEW: For running external script
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal # <-- NEW: QThread, pyqtSignal

# --- Import from our local files ---
from ui_definitions import (
    create_sidebar, create_home_widget, 
    create_setup_widget, create_collection_widget,
    create_training_widget, create_recognition_widget, # <-- NEW
    create_manage_data_widget # <-- NEW
)
from mediapipe_logic import (
    mediapipe_detection, draw_styled_landmarks, ProcessingThread,
    mp_holistic, DATA_PATH, OUTPUT_PATH, RECORD_SECONDS
)
from recognition_logic import RecognitionWorker # <-- NEW
from video_recorder import VideoRecorder # <-- NEW: Professional video recorder

# --- APPLICATION STATES ---
STATE_HOME = 0
STATE_SETUP = 1
STATE_WAITING_FOR_BATCH = 2
STATE_BATCH_COUNTDOWN = 3
STATE_PAUSE_COUNTDOWN = 4
STATE_RECORDING = 5
STATE_PROCESSING = 6
STATE_SESSION_DONE = 7
STATE_TRAINING = 8 # <-- NEW
STATE_RECOGNITION = 9 # <-- NEW
STATE_MANAGE_DATA = 10 # <-- NEW

# --- NEW: Training Thread ---
class TrainingThread(QThread):
    """
    Runs the model training script in a separate process
    and emits its stdout line by line.
    """
    log_update = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.process = None

    def run(self):
        print("Starting training thread...")
        try:
            # We use sys.executable to ensure we use the same python interpreter
            # (e.g., from the virtual environment)
            self.process = subprocess.Popen(
                [sys.executable, 'train_model.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1 # Line-buffered
            )

            # Read stdout line by line in real-time
            for line in iter(self.process.stdout.readline, ''):
                if not line:
                    break
                self.log_update.emit(line.strip())

            self.process.stdout.close()
            self.process.wait()
            print("Training process finished.")

        except Exception as e:
            print(f"Error starting training process: {e}")
            self.log_update.emit(f"\n--- ERROR ---")
            self.log_update.emit(f"Failed to start training script: {e}")
            self.log_update.emit("Make sure 'train_model.py' is in the same folder.")
            self.log_update.emit("Ensure all requirements are installed (tensorflow, sklearn, etc.)")

    def stop(self):
        if self.process and self.process.poll() is None:
            print("Terminating training process...")
            self.process.terminate()
            self.process.wait()
            print("Training process terminated.")


class CollectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Project ANWAYA: Data Collector")
        self.setGeometry(100, 100, 1280, 720)
        self.setObjectName("MainWindow")

        # --- App State ---
        self.app_state = STATE_HOME
        self.action_name = ""
        self.num_videos = 0
        self.current_video_num = 0
        self.start_num = 0
        self.batch_size = 10
        self.countdown_timer = None
        self.countdown_val = 0
        self.record_start_time = 0
        self.current_record_seconds = RECORD_SECONDS

        # --- MediaPipe/CV ---
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.cap = None
        self.video_writer = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_frame)
        self.processing_thread = None
        self.training_thread = None # <-- NEW
        self.recognition_worker = None # <-- NEW
        
        # --- Professional Video Recorder ---
        self.video_recorder = VideoRecorder()
        self.video_recorder.recording_started.connect(self.on_recording_started)
        self.video_recorder.recording_stopped.connect(self.on_recording_stopped)
        self.video_recorder.error_occurred.connect(self.on_recorder_error)

        # --- Init ---
        self.initUI()
        self.load_existing_actions()
        
    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Create UI from definitions
        self.sidebar = create_sidebar(self)
        self.main_layout.addWidget(self.sidebar)

        self.main_content = QWidget()
        self.main_content.setObjectName("MainContent")
        self.main_content_layout = QVBoxLayout(self.main_content)
        self.main_layout.addWidget(self.main_content, 1)

        self.stacked_widget = QStackedWidget()
        self.main_content_layout.addWidget(self.stacked_widget)

        self.home_widget = create_home_widget(self)
        self.setup_widget = create_setup_widget(self)
        self.collection_widget = create_collection_widget(self)
        self.training_widget = create_training_widget(self) # <-- NEW
        self.recognition_widget = create_recognition_widget(self) # <-- NEW
        self.manage_data_widget = create_manage_data_widget(self) # <-- NEW
        
        self.stacked_widget.addWidget(self.home_widget)
        self.stacked_widget.addWidget(self.setup_widget)
        self.stacked_widget.addWidget(self.collection_widget)
        self.stacked_widget.addWidget(self.training_widget) # <-- NEW
        self.stacked_widget.addWidget(self.recognition_widget) # <-- NEW
        self.stacked_widget.addWidget(self.manage_data_widget) # <-- NEW

        # Connect signals
        self.rec_time_spin.valueChanged.connect(self.update_record_time)
        
        # Connect recognition signals
        self.recognition_back_btn.clicked.connect(self.go_to_home)
        self.recognition_start_btn.clicked.connect(self.start_recognition)
        self.recognition_stop_btn.clicked.connect(self.stop_recognition)
        
        # Connect manage data signals
        self.manage_data_back_btn.clicked.connect(self.go_to_home)
        self.manage_actions_list.itemClicked.connect(self.on_manage_action_selected)
        self.view_videos_btn.clicked.connect(self.view_action_videos)
        self.delete_action_btn.clicked.connect(self.delete_action_data)
        self.export_data_btn.clicked.connect(self.export_data_info)
        self.refresh_actions_btn.clicked.connect(self.refresh_actions_data)
        self.restart_app_btn.clicked.connect(self.restart_application)

        self.stacked_widget.setCurrentWidget(self.home_widget)

    # --- UI Logic ---
    
    def update_record_time(self, value):
        self.current_record_seconds = value
        print(f"Record time updated to: {self.current_record_seconds}s")
    
    def load_existing_actions(self):
        self.action_list.clear()
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        try:
            actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
            actions.sort()
            action_items = []
            for action in actions:
                vid_count = len([f for f in os.listdir(os.path.join(DATA_PATH, action)) if f.endswith(('.mp4', '.avi', '.mov', '.webm'))])
                action_items.append(f"{action} ({vid_count})")
            self.action_list.addItems(action_items)
        except Exception as e:
            print(f"Error loading actions: {e}")
            
    def filter_actions(self, text):
        for i in range(self.action_list.count()):
            item = self.action_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def on_action_clicked(self, item):
        action_name = item.text().split(' (')[0]
        self.action_name_input.setText(action_name)

    # --- Navigation ---
    
    def go_to_setup(self):
        self.stacked_widget.setCurrentWidget(self.setup_widget)
        self.set_state(STATE_SETUP)

    def go_to_recognition(self):
        """Navigate to recognition page"""
        self.stacked_widget.setCurrentWidget(self.recognition_widget)
        self.set_state(STATE_RECOGNITION)
    
    def go_to_manage_data(self):
        """Navigate to data management page"""
        self.stacked_widget.setCurrentWidget(self.manage_data_widget)
        self.set_state(STATE_MANAGE_DATA)
        self.load_data_statistics()

    def go_to_home(self):
        # Stop any running threads before going home
        if self.app_state == STATE_TRAINING:
            if self.training_thread and self.training_thread.isRunning():
                self.training_thread.stop()
        
        if self.app_state == STATE_RECOGNITION:
            if self.recognition_worker and self.recognition_worker.isRunning():
                self.stop_recognition()
        
        self.stacked_widget.setCurrentWidget(self.home_widget)
        self.set_state(STATE_HOME)

    # --- NEW: Training Navigation and Slots ---
    def go_to_training(self):
        self.set_state(STATE_TRAINING)
        self.stacked_widget.setCurrentWidget(self.training_widget)
        
        self.training_log_display.clear()
        self.training_log_display.append("--- Starting Model Training ---")
        self.training_log_display.append(f"Using Python: {sys.executable}")
        self.training_log_display.append("This may take several minutes...")
        
        self.train_model_button.setDisabled(True)
        self.training_back_button.setDisabled(True)

        # Start the thread
        self.training_thread = TrainingThread()
        self.training_thread.log_update.connect(self.append_training_log)
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.start()

    def append_training_log(self, text):
        """Appends text to the training log and auto-scrolls."""
        self.training_log_display.append(text)
        self.training_log_display.verticalScrollBar().setValue(
            self.training_log_display.verticalScrollBar().maximum()
        )

    def on_training_finished(self):
        """Called when the TrainingThread finishes."""
        self.append_training_log("\n--- TRAINING COMPLETE ---")
        self.train_model_button.setDisabled(False)
        self.training_back_button.setDisabled(False)
        self.set_state(STATE_HOME) # Or wherever you want to be
        QMessageBox.information(self, "Training Complete", "The model training process has finished.")

    # --- Recognition Control Methods ---
    
    def start_recognition(self):
        """Start real-time recognition"""
        self.recognition_worker = RecognitionWorker()
        self.recognition_worker.frame_ready.connect(self.update_recognition_frame)
        self.recognition_worker.prediction_ready.connect(self.update_prediction)
        self.recognition_worker.status_update.connect(self.update_recognition_status)
        self.recognition_worker.error_occurred.connect(self.show_recognition_error)
        
        self.recognition_start_btn.setEnabled(False)
        self.recognition_stop_btn.setEnabled(True)
        
        self.recognition_worker.start()
    
    def stop_recognition(self):
        """Stop real-time recognition"""
        if self.recognition_worker:
            self.recognition_worker.stop()
            self.recognition_worker = None
        
        self.recognition_start_btn.setEnabled(True)
        self.recognition_stop_btn.setEnabled(False)
        
        # Clear video display
        self.recognition_video_label.clear()
        self.recognition_video_label.setText("Camera Stopped")
        self.recognition_prediction_label.setText("Ready to recognize...")
        self.recognition_confidence_label.setText("")
    
    def update_recognition_frame(self, qt_image):
        """Update video frame display"""
        pixmap = QPixmap.fromImage(qt_image)
        self.recognition_video_label.setPixmap(
            pixmap.scaled(self.recognition_video_label.size(), 
                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
    
    def update_prediction(self, action, confidence):
        """Update prediction display"""
        if confidence > 0:
            self.recognition_prediction_label.setText(action)
            self.recognition_confidence_label.setText(f"{confidence*100:.0f}%")
        else:
            self.recognition_prediction_label.setText(action)
            self.recognition_confidence_label.setText("")
    
    def update_recognition_status(self, status):
        """Update status indicator"""
        self.recognition_status_text.setText(status)
        if "started" in status.lower():
            self.recognition_status_indicator.setStyleSheet("color: #4caf50; font-size: 20px;")
        else:
            self.recognition_status_indicator.setStyleSheet("color: #666; font-size: 20px;")
    
    def show_recognition_error(self, error_msg):
        """Show error message"""
        QMessageBox.critical(self, "Recognition Error", error_msg)
        self.stop_recognition()

    # --- END Recognition Control Methods ---


    # --- Session Logic ---

    def start_session(self):
        self.action_name = self.action_name_input.text().strip()
        self.num_videos = self.num_videos_input.value()
        
        if not self.action_name:
            QMessageBox.warning(self, "Error", "Please enter an action name.")
            return
            
        self.action_video_dir = os.path.join(DATA_PATH, self.action_name)
        self.action_landmark_dir = os.path.join(OUTPUT_PATH, self.action_name)
        os.makedirs(self.action_video_dir, exist_ok=True)
        os.makedirs(self.action_landmark_dir, exist_ok=True)
        
        self.start_num = 0
        while True:
            video_file_exists = any(os.path.exists(os.path.join(self.action_video_dir, f"{self.start_num}{ext}")) for ext in ['.mp4', '.webm', '.avi', '.mov'])
            if not video_file_exists: break
            self.start_num += 1
        print(f"Starting video number will be: {self.start_num}")
        self.current_video_num = self.start_num
        
        self.session_action_label.setText(f"Action: {self.action_name}")
        self.session_progress_bar.setRange(0, self.num_videos)
        self.session_progress_bar.setValue(0)
        self.stop_session_button.setDisabled(False)
        self.action_list.setDisabled(True)
        self.action_search.setDisabled(True)
        self.train_model_button.setDisabled(True) # Disable training during collection

        cam_index = self.cam_select.currentIndex()
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", f"Could not open webcam {cam_index}.")
            return
        
        # Initialize the video recorder with the camera
        self.video_recorder.initialize_recorder(self.cap)
            
        self.camera_timer.start(33)
        self.stacked_widget.setCurrentWidget(self.collection_widget)
        self.set_state(STATE_WAITING_FOR_BATCH)

    def stop_session(self):
        self.camera_timer.stop()
        
        # Stop video recorder if recording
        if self.video_recorder.is_recording():
            self.video_recorder.stop_recording()
        
        if self.cap: self.cap.release(); self.cap = None
        if self.video_writer: self.video_writer.release(); self.video_writer = None
        if self.processing_thread: self.processing_thread.quit(); self.processing_thread.wait()
            
        self.load_existing_actions()
        self.stacked_widget.setCurrentWidget(self.home_widget)
        
        self.session_action_label.setText("Action: N/A")
        self.session_progress_bar.setValue(0)
        self.stop_session_button.setDisabled(True)
        self.action_list.setDisabled(False)
        self.action_search.setDisabled(False)
        self.train_model_button.setDisabled(False) # Re-enable training
        
        self.set_state(STATE_HOME)
        print("Session stopped.")

    def set_state(self, new_state):
        self.app_state = new_state
        print(f"New state: {new_state}")
        
        # Disable sidebar buttons based on state
        is_idle = new_state in [STATE_HOME, STATE_SETUP, STATE_RECOGNITION]
        self.action_list.setEnabled(is_idle)
        self.action_search.setEnabled(is_idle)
        self.train_model_button.setEnabled(is_idle and new_state != STATE_RECOGNITION)
        self.stop_session_button.setEnabled(not is_idle and new_state != STATE_TRAINING)
        self.cam_select.setEnabled(is_idle and new_state != STATE_RECOGNITION)
        self.rec_time_spin.setEnabled(is_idle and new_state != STATE_RECOGNITION)
        
        if self.app_state > STATE_SETUP and self.app_state != STATE_TRAINING:
            self.center_text_label.setVisible(False)
            self.recording_label.setVisible(False)
            self.start_batch_button.setVisible(False)

        if new_state == STATE_WAITING_FOR_BATCH:
            batch_num = (self.current_video_num - self.start_num) // self.batch_size + 1
            self.center_text_label.setText(f"Batch {batch_num}. Ready for video {self.current_video_num}.")
            self.center_text_label.setObjectName("LabelBigOverlay")
            self.center_text_label.setVisible(True)
            self.start_batch_button.setText("START BATCH (S)")
            self.start_batch_button.setVisible(True)
            self.status_text_label.setText(f"Ready for batch. Press 'S' to start.")
            
        elif new_state == STATE_BATCH_COUNTDOWN:
            self.status_text_label.setText("Get ready...")
            self.center_text_label.setObjectName("LabelCountdown")
            self.start_countdown(5, STATE_PAUSE_COUNTDOWN)
            
        elif new_state == STATE_PAUSE_COUNTDOWN:
            self.status_text_label.setText(f"Starting video {self.current_video_num}...")
            self.center_text_label.setObjectName("LabelCountdown")
            self.start_countdown(2, STATE_RECORDING)
            
        elif new_state == STATE_RECORDING:
            self.recording_label.setVisible(True)
            self.status_text_label.setText(f"Recording video {self.current_video_num}...")
            self.record_start_time = time.time()
            self.start_professional_recording()

        elif new_state == STATE_PROCESSING:
            self.center_text_label.setText("Processing...")
            self.center_text_label.setObjectName("LabelProcessing")
            self.center_text_label.setVisible(True)
            self.status_text_label.setText("Processing landmarks...")
            self.start_processing()
            
        elif new_state == STATE_SESSION_DONE:
            self.status_text_label.setText("Session Complete!")
            QMessageBox.information(self, "Success", "Data collection and processing complete!")
            self.stop_session()

    def start_countdown(self, seconds, next_state):
        self.countdown_val = seconds
        self.countdown_next_state = next_state
        self.center_text_label.setText(str(self.countdown_val))
        self.center_text_label.setVisible(True)
        
        if self.countdown_timer: self.countdown_timer.stop()
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)

    def update_countdown(self):
        self.countdown_val -= 1
        if self.countdown_val <= 0:
            self.countdown_timer.stop()
            self.set_state(self.countdown_next_state)
        else:
            self.center_text_label.setText(str(self.countdown_val))

    def start_batch_countdown(self):
        if self.app_state == STATE_WAITING_FOR_BATCH:
            self.set_state(STATE_BATCH_COUNTDOWN)
    
    # --- Professional Video Recording Methods ---
    
    def start_professional_recording(self):
        """Start recording using the VideoRecorder system"""
        if not self.cap:
            print("[MainWindow] ERROR: Camera not initialized")
            return
        
        # Prepare file path
        video_name = str(self.current_video_num)
        self.video_save_path = os.path.join(self.action_video_dir, f"{video_name}.mp4")
        self.landmark_save_folder = os.path.join(self.action_landmark_dir, video_name)
        os.makedirs(self.landmark_save_folder, exist_ok=True)
        
        # Initialize and start recorder
        if not self.video_recorder.is_ready():
            self.video_recorder.initialize_recorder(self.cap)
        
        # Start recording with duration limit
        success = self.video_recorder.start_recording(
            self.video_save_path, 
            duration_seconds=self.current_record_seconds
        )
        
        if not success:
            print(f"[MainWindow] Failed to start recording")
            QMessageBox.warning(self, "Recording Error", "Could not start video recording")
            self.set_state(STATE_WAITING_FOR_BATCH)
    
    def on_recording_started(self):
        """Callback when recording starts"""
        print(f"[MainWindow] Recording started callback")
    
    def on_recording_stopped(self, file_path):
        """Callback when recording stops"""
        print(f"[MainWindow] Recording stopped: {file_path}")
        # Note: Don't change state here, it's managed by update_frame
    
    def on_recorder_error(self, error_msg):
        """Callback for recorder errors"""
        print(f"[MainWindow] Recorder error: {error_msg}")
        QMessageBox.warning(self, "Recording Error", error_msg)
            
    def init_video_writer(self):
        if not self.cap: return
        video_name = str(self.current_video_num)
        self.video_save_path = os.path.join(self.action_video_dir, f"{video_name}.mp4")
        self.landmark_save_folder = os.path.join(self.action_landmark_dir, video_name)
        os.makedirs(self.landmark_save_folder, exist_ok=True)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate FPS based on timer interval (33ms = ~30.3 FPS)
        # Writing every frame with this FPS ensures video duration matches recording time
        timer_interval_ms = 33  # milliseconds
        fps = 1000.0 / timer_interval_ms  # ~30.3 FPS
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.video_save_path, fourcc, fps, (width, height))
        print(f"VideoWriter initialized for {self.video_save_path} at {fps:.1f} FPS")

    # --- Main Camera Loop ---
    
    def update_frame(self):
        if not self.cap or not self.cap.isOpened(): return
        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, self.holistic)
        draw_styled_landmarks(image, results)
        
        if self.app_state == STATE_RECORDING:
            elapsed = time.time() - self.record_start_time
            cv2.putText(image, f"0:0{int(elapsed)+1}", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Check if recording duration completed
            if elapsed >= self.current_record_seconds:
                # Stop the professional recorder
                if self.video_recorder.is_recording():
                    saved_path = self.video_recorder.stop_recording()
                    print(f"[MainWindow] Video recording completed: {saved_path}")
                
                # Transition to processing
                self.set_state(STATE_PROCESSING)
                
        qt_image = self.convert_cv_to_pixmap(image)
        self.video_feed_label.setPixmap(qt_image)
        
    def convert_cv_to_pixmap(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_feed_label.width(), self.video_feed_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    # --- Processing ---

    def start_processing(self):
        print("Starting processing thread...")
        self.processing_thread = ProcessingThread(self.video_save_path, self.landmark_save_folder, self.holistic)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def on_processing_finished(self, output_folder):
        print(f"Processing finished for {output_folder}")
        self.current_video_num += 1
        progress = self.current_video_num - self.start_num
        self.session_progress_bar.setValue(progress)
        
        if progress >= self.num_videos:
            self.set_state(STATE_SESSION_DONE)
        else:
            if progress % self.batch_size == 0:
                self.set_state(STATE_WAITING_FOR_BATCH)
            else:
                self.set_state(STATE_PAUSE_COUNTDOWN)

    # --- Event Handlers ---

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Q:
            if self.app_state > STATE_SETUP and self.app_state != STATE_TRAINING:
                self.stop_session()
            elif self.app_state < STATE_SETUP:
                self.close()
        if key == Qt.Key_S:
            if self.app_state == STATE_WAITING_FOR_BATCH:
                self.start_batch_countdown()

    # --- Data Management Methods ---
    
    def load_data_statistics(self):
        """Load and display data statistics"""
        try:
            if not os.path.exists(DATA_PATH):
                self.data_stats_label.setText("No data collected yet.")
                self.manage_actions_list.clear()
                return
            
            actions = [d for d in os.listdir(DATA_PATH) 
                      if os.path.isdir(os.path.join(DATA_PATH, d))]
            
            total_videos = 0
            total_processed = 0
            action_data = []
            
            for action in actions:
                action_video_dir = os.path.join(DATA_PATH, action)
                action_processed_dir = os.path.join(OUTPUT_PATH, action)
                
                video_count = len([f for f in os.listdir(action_video_dir) 
                                  if f.endswith(('.mp4', '.avi', '.mov', '.webm'))])
                
                processed_count = 0
                if os.path.exists(action_processed_dir):
                    processed_count = len([d for d in os.listdir(action_processed_dir) 
                                          if os.path.isdir(os.path.join(action_processed_dir, d))])
                
                total_videos += video_count
                total_processed += processed_count
                action_data.append((action, video_count, processed_count))
            
            # Update statistics
            stats_text = f"""
📊 Dataset Statistics:
━━━━━━━━━━━━━━━━━━━━
Total Actions: {len(actions)}
Total Videos: {total_videos}
Total Processed: {total_processed}
━━━━━━━━━━━━━━━━━━━━
            """.strip()
            self.data_stats_label.setText(stats_text)
            
            # Update actions list
            self.manage_actions_list.clear()
            for action, vid_count, proc_count in sorted(action_data):
                item_text = f"{action}  |  📹 {vid_count} videos  |  ✓ {proc_count} processed"
                self.manage_actions_list.addItem(item_text)
                
        except Exception as e:
            self.data_stats_label.setText(f"Error loading statistics: {str(e)}")
    
    def on_manage_action_selected(self, item):
        """Handle action selection in manage data page"""
        action_name = item.text().split('  |  ')[0]
        
        action_video_dir = os.path.join(DATA_PATH, action_name)
        action_processed_dir = os.path.join(OUTPUT_PATH, action_name)
        
        video_files = [f for f in os.listdir(action_video_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.webm'))]
        
        processed_folders = []
        if os.path.exists(action_processed_dir):
            processed_folders = [d for d in os.listdir(action_processed_dir) 
                                if os.path.isdir(os.path.join(action_processed_dir, d))]
        
        details_text = f"""
Action: {action_name}
━━━━━━━━━━━━━━━━━━━━
Videos: {len(video_files)}
Processed: {len(processed_folders)}
Location: {action_video_dir}
        """.strip()
        
        self.action_details_label.setText(details_text)
        self.view_videos_btn.setEnabled(True)
        self.delete_action_btn.setEnabled(True)
        self.selected_action = action_name
    
    def view_action_videos(self):
        """Open file explorer to view action videos"""
        if hasattr(self, 'selected_action'):
            action_video_dir = os.path.join(DATA_PATH, self.selected_action)
            if os.path.exists(action_video_dir):
                os.startfile(action_video_dir)
    
    def delete_action_data(self):
        """Delete all data for selected action"""
        if not hasattr(self, 'selected_action'):
            return
        
        reply = QMessageBox.question(
            self, 'Confirm Deletion',
            f'Are you sure you want to delete all data for action "{self.selected_action}"?\n\n'
            'This will delete:\n'
            '- All video files\n'
            '- All processed landmarks\n\n'
            'This action cannot be undone!',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                import shutil
                
                # Delete video folder
                action_video_dir = os.path.join(DATA_PATH, self.selected_action)
                if os.path.exists(action_video_dir):
                    shutil.rmtree(action_video_dir)
                
                # Delete processed folder
                action_processed_dir = os.path.join(OUTPUT_PATH, self.selected_action)
                if os.path.exists(action_processed_dir):
                    shutil.rmtree(action_processed_dir)
                
                QMessageBox.information(self, "Success", 
                                      f'All data for "{self.selected_action}" has been deleted.')
                
                # Reload statistics
                self.load_data_statistics()
                self.action_details_label.setText("Select an action to view details")
                self.view_videos_btn.setEnabled(False)
                self.delete_action_btn.setEnabled(False)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete data: {str(e)}")
    
    def export_data_info(self):
        """Export dataset information to text file"""
        try:
            if not os.path.exists(DATA_PATH):
                QMessageBox.warning(self, "No Data", "No data to export.")
                return
            
            import datetime
            
            actions = [d for d in os.listdir(DATA_PATH) 
                      if os.path.isdir(os.path.join(DATA_PATH, d))]
            
            export_text = f"""
ISL Dataset Information
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

Total Actions: {len(actions)}

{'='*50}
ACTION DETAILS:
{'='*50}

"""
            
            for action in sorted(actions):
                action_video_dir = os.path.join(DATA_PATH, action)
                action_processed_dir = os.path.join(OUTPUT_PATH, action)
                
                video_files = [f for f in os.listdir(action_video_dir) 
                              if f.endswith(('.mp4', '.avi', '.mov', '.webm'))]
                
                processed_count = 0
                if os.path.exists(action_processed_dir):
                    processed_count = len([d for d in os.listdir(action_processed_dir) 
                                          if os.path.isdir(os.path.join(action_processed_dir, d))])
                
                export_text += f"""
Action: {action}
  - Videos: {len(video_files)}
  - Processed: {processed_count}
  - Path: {action_video_dir}

"""
            
            export_path = "dataset_info.txt"
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(export_text)
            
            QMessageBox.information(self, "Export Complete", 
                                  f'Dataset information exported to:\n{os.path.abspath(export_path)}')
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {str(e)}")
    
    def refresh_actions_data(self):
        """Refresh the actions list and statistics"""
        self.load_data_statistics()
        self.action_details_label.setText("Select an action to view details")
        self.view_videos_btn.setEnabled(False)
        self.delete_action_btn.setEnabled(False)
        
        # Also refresh the sidebar action list
        self.load_existing_actions()
        
        QMessageBox.information(self, "Refreshed", "Data has been refreshed successfully!")
    
    def restart_application(self):
        """Restart the entire application"""
        reply = QMessageBox.question(
            self, 'Restart Application',
            'Are you sure you want to restart the application?\n\n'
            'All unsaved progress will be lost.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Stop all running processes
            self.stop_session()
            if self.training_thread and self.training_thread.isRunning():
                self.training_thread.stop()
            if self.recognition_worker and self.recognition_worker.isRunning():
                self.recognition_worker.stop()
            if self.holistic:
                self.holistic.close()
            
            # Restart the application
            import sys
            import subprocess
            
            # Get the python executable and script path
            python = sys.executable
            script = os.path.abspath(sys.argv[0])
            
            # Close current application
            self.close()
            
            # Start new instance
            subprocess.Popen([python, script])
            
            # Exit current process
            sys.exit()
    
    # --- END Data Management Methods ---

    def closeEvent(self, event):
        # Clean up all threads
        self.stop_session()
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
        if self.recognition_worker and self.recognition_worker.isRunning():
            self.recognition_worker.stop()
        if self.holistic:
            self.holistic.close()
        event.accept()