import cv2
import numpy as np
import os
import mediapipe as mp
import time
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QLineEdit, QSpinBox, QListWidget, 
    QStackedWidget, QFrame, QProgressBar, QSizePolicy, QComboBox,
    QGridLayout, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal

# --- 1. SCRIPT PARAMETERS (From your script) ---
DATA_PATH = os.path.join('ISL_Data')     # Path for raw videos
OUTPUT_PATH = os.path.join('ISL_Processed') # Path for processed landmarks
SEQUENCE_LENGTH = 30
RECORD_SECONDS = 3
KEYPOINT_SIZE = 1662

# --- 2. MEDIAPIPE SETUP (From your script) ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """
    Performs landmark detection on an image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Color conversion
    image.flags.writeable = False                  # Make image non-writeable
    results = model.process(image)                 # Make detection
    image.flags.writeable = True                   # Make image writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Color conversion back
    return image, results

def draw_styled_landmarks(image, results):
    """
    Draws the landmarks on the image with the clean style.
    """
    # 1. Pose connections (Green)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)) 
    
    # 2. Left hand connections (Blue)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1)) 
    
    # 3. Right hand connections (Red)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1))

def extract_keypoints(results):
    """
    Extracts the keypoints from the MediaPipe results and flattens them into a single array.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# --- 3. PROCESSING THREAD (To prevent UI freezing) ---

class ProcessingThread(QThread):
    """
    Runs the landmark processing in a separate thread.
    """
    finished = pyqtSignal(str) # Signal to emit when done

    def __init__(self, video_path, output_folder, holistic_model):
        super().__init__()
        self.video_path = video_path
        self.output_folder = output_folder
        self.holistic_model = holistic_model

    def run(self):
        """
        This is the function that will be executed in the new thread.
        """
        print(f"  [Thread] Processing {self.video_path}...")
        
        # --- YOUR process_and_save_landmarks FUNCTION ---
        # This function is run in the background.
        
        cap_proc = cv2.VideoCapture(self.video_path)
        all_video_keypoints_raw = []

        while(cap_proc.isOpened()):
            ret, frame = cap_proc.read()
            if not ret:
                break 

            # NOTE: We can't pass the main 'holistic' model to a new thread.
            # We must create a NEW model instance *within* this thread.
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as thread_holistic:
                image, results = mediapipe_detection(frame, thread_holistic)
                keypoints = extract_keypoints(results)
                all_video_keypoints_raw.append(keypoints)

        cap_proc.release()
        
        if not all_video_keypoints_raw: 
            print(f"  [Thread] Warning: No keypoints from {self.video_path}. Creating {SEQUENCE_LENGTH} empty files.")
            for j in range(SEQUENCE_LENGTH):
                npy_path = os.path.join(self.output_folder, f"{j}.npy")
                np.save(npy_path, np.zeros(KEYPOINT_SIZE))
            self.finished.emit(self.output_folder) # Emit signal
            return 

        num_frames = len(all_video_keypoints_raw)
        indices = np.linspace(
            0,             
            num_frames - 1,
            SEQUENCE_LENGTH, 
            dtype=int        
        )
        
        for j, frame_index in enumerate(indices):
            keypoints_to_save = all_video_keypoints_raw[frame_index]
            npy_path = os.path.join(self.output_folder, f"{j}.npy")
            np.save(npy_path, keypoints_to_save)
        
        print(f"  [Thread] Successfully saved {SEQUENCE_LENGTH} landmark files to {self.output_folder}/")
        # --- END OF YOUR FUNCTION ---

        self.finished.emit(self.output_folder) # Emit signal


# --- 4. STYLING (QSS - Qt StyleSheet) ---
APP_STYLESHEET = """
    /* Main Window */
    QMainWindow, #MainWindow {
        background-color: #1E1E1E; /* Dark Charcoal */
    }

    /* All Widgets */
    QWidget {
        font-family: 'Roboto', 'Inter', 'Segoe UI', sans-serif;
        color: #FFFFFF; /* White text */
        font-size: 14px;
    }

    /* Sidebar */
    #Sidebar {
        background-color: #2D2D2D; /* Lighter Gray */
        border-right: 1px solid #444;
    }
    
    #Sidebar QLabel {
        font-size: 16px;
        font-weight: bold;
    }
    
    #SidebarTitle {
        font-size: 12px;
        font-weight: bold;
        color: #AFAFAF;
        margin-top: 5px;
        margin-bottom: 0px;
    }

    /* Main Content Area */
    #MainContent {
        background-color: #1E1E1E;
    }

    /* Labels */
    QLabel {
        color: #FFFFFF;
    }
    
    #LabelHelper {
        color: #AFAFAF; /* Light Gray */
        font-size: 12px;
    }
    
    #LabelBigOverlay {
        font-size: 48px;
        font-weight: bold;
        color: rgba(255, 255, 255, 0.9);
    }
    
    #LabelRecording {
        font-size: 24px;
        font-weight: bold;
        color: #D93025; /* Red */
        background-color: rgba(0, 0, 0, 0.5);
        padding: 5px;
        border-radius: 5px;
    }
    
    #LabelCountdown {
        font-size: 72px;
        font-weight: bold;
        color: #FFC300; /* Amber */
    }
    
    #LabelProcessing {
        font-size: 48px;
        font-weight: bold;
        color: #0078D4; /* Blue */
    }
    
    /* Buttons */
    QPushButton {
        background-color: #0078D4; /* Primary Accent Blue */
        color: #FFFFFF;
        border: none;
        padding: 10px 15px;
        border-radius: 4px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #005A9E; /* Darker Blue */
    }
    QPushButton:pressed {
        background-color: #004578;
    }
    
    #ButtonStop {
        background-color: #D93025; /* Red */
    }
    #ButtonStop:hover {
        background-color: #A52A2A;
    }
    
    #ButtonGray {
        background-color: #555;
    }
    #ButtonGray:hover {
        background-color: #777;
    }
    
    #ButtonGreen {
        background-color: #34A853; /* Green */
    }
    #ButtonGreen:hover {
        background-color: #2E8B57;
    }

    /* Text Inputs / Spin Boxes / Combo Boxes */
    QLineEdit, QSpinBox, QComboBox {
        background-color: #2D2D2D;
        color: #FFFFFF;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 8px;
        font-size: 14px;
    }
    QComboBox QAbstractItemView {
        background-color: #2D2D2D;
        border: 1px solid #555;
        selection-background-color: #0078D4;
    }

    /* List Widget */
    QListWidget {
        background-color: #2D2D2D;
        border: 1px solid #555;
        border-radius: 4px;
    }
    QListWidget::item {
        padding: 8px;
    }
    QListWidget::item:hover {
        background-color: #444;
    }
    QListWidget::item:selected {
        background-color: #0078D4;
    }

    /* Progress Bar */
    QProgressBar {
        border: 1px solid #555;
        border-radius: 4px;
        text-align: center;
        color: #FFFFFF;
    }
    QProgressBar::chunk {
        background-color: #0078D4;
        border-radius: 4px;
    }
"""


# --- 5. APPLICATION STATES ---
STATE_SETUP = 0
STATE_WAITING_FOR_BATCH = 1
STATE_BATCH_COUNTDOWN = 2
STATE_PAUSE_COUNTDOWN = 3
STATE_RECORDING = 4
STATE_PROCESSING = 5
STATE_SESSION_DONE = 6


# --- 6. MAIN APPLICATION WINDOW ---

class CollectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Project Vaani: Data Collector")
        self.setGeometry(100, 100, 1280, 720)
        self.setObjectName("MainWindow")

        # --- App State ---
        self.app_state = STATE_SETUP
        self.action_name = ""
        self.num_videos = 0
        self.current_video_num = 0
        self.start_num = 0 # Starting video number
        self.batch_size = 10
        self.countdown_timer = None
        self.countdown_val = 0
        self.record_start_time = 0

        # --- MediaPipe/CV ---
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.cap = None
        self.video_writer = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_frame)
        self.processing_thread = None

        # --- Init ---
        self.initUI()
        self.load_existing_actions()
        self.setStyleSheet(APP_STYLESHEET)

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # --- 1. Left Sidebar ---
        self.sidebar = self.create_sidebar()
        self.main_layout.addWidget(self.sidebar)

        # --- 2. Right Main Content Area ---
        self.main_content = QWidget()
        self.main_content.setObjectName("MainContent")
        self.main_content_layout = QVBoxLayout(self.main_content)
        self.main_layout.addWidget(self.main_content, 1) # '1' makes it stretch

        # Stacked widget to switch between "Setup" and "Collection"
        self.stacked_widget = QStackedWidget()
        self.main_content_layout.addWidget(self.stacked_widget)

        # --- Page 1: Setup Screen ---
        self.setup_widget = self.create_setup_widget()
        self.stacked_widget.addWidget(self.setup_widget)

        # --- Page 2: Collection Screen ---
        self.collection_widget = self.create_collection_widget()
        self.stacked_widget.addWidget(self.collection_widget)

    def create_sidebar(self):
        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(250)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(10)

        title = QLabel("PROJECT VAANI")
        title.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title)
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        sidebar_layout.addWidget(line)

        # --- Action List ---
        sidebar_layout.addWidget(QLabel("ACTIONS", objectName="SidebarTitle"))
        self.action_search = QLineEdit()
        self.action_search.setPlaceholderText("Search actions...")
        self.action_search.textChanged.connect(self.filter_actions)
        sidebar_layout.addWidget(self.action_search)

        self.action_list = QListWidget()
        self.action_list.itemClicked.connect(self.on_action_clicked)
        sidebar_layout.addWidget(self.action_list, 1) # Stretch

        # --- Current Session ---
        sidebar_layout.addWidget(QLabel("SESSION", objectName="SidebarTitle"))
        self.session_action_label = QLabel("Action: N/A")
        sidebar_layout.addWidget(self.session_action_label)
        
        self.session_progress_bar = QProgressBar()
        self.session_progress_bar.setValue(0)
        sidebar_layout.addWidget(self.session_progress_bar)

        # --- Settings ---
        sidebar_layout.addWidget(QLabel("SETTINGS", objectName="SidebarTitle"))
        
        # Camera
        cam_layout = QHBoxLayout()
        cam_layout.addWidget(QLabel("Camera:"))
        self.cam_select = QComboBox()
        self.cam_select.addItems(["Webcam 0", "Webcam 1"]) # Add more if needed
        cam_layout.addWidget(self.cam_select)
        sidebar_layout.addLayout(cam_layout)

        # Record Time
        rec_layout = QHBoxLayout()
        rec_layout.addWidget(QLabel("Rec. Time (s):"))
        self.rec_time_spin = QSpinBox()
        self.rec_time_spin.setRange(1, 10)
        self.rec_time_spin.setValue(RECORD_SECONDS)
        self.rec_time_spin.valueChanged.connect(lambda x: globals().update(RECORD_SECONDS=x))
        rec_layout.addWidget(self.rec_time_spin)
        sidebar_layout.addLayout(rec_layout)

        # --- Controls ---
        sidebar_layout.addStretch()
        self.stop_session_button = QPushButton("STOP SESSION")
        self.stop_session_button.setObjectName("ButtonStop")
        self.stop_session_button.clicked.connect(self.stop_session)
        self.stop_session_button.setDisabled(True)
        sidebar_layout.addWidget(self.stop_session_button)

        return sidebar

    def create_setup_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)

        title = QLabel("Setup New Collection Session")
        title.setFont(QFont('Roboto', 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addStretch(1)

        # Action Name
        layout.addWidget(QLabel("Action Name (Marathi)"))
        self.action_name_input = QLineEdit()
        self.action_name_input.setPlaceholderText("e.g., आभार")
        self.action_name_input.setFont(QFont('Roboto', 16))
        layout.addWidget(self.action_name_input)
        layout.addWidget(QLabel("(Select from list or type new)", objectName="LabelHelper"))

        # Number of Videos
        layout.addWidget(QLabel("Number of Videos to Record"))
        self.num_videos_input = QSpinBox()
        self.num_videos_input.setRange(1, 1000)
        self.num_videos_input.setValue(50)
        self.num_videos_input.setFont(QFont('Roboto', 16))
        layout.addWidget(self.num_videos_input)

        layout.addStretch(1)

        # Start Button
        self.start_session_button = QPushButton("START SESSION")
        self.start_session_button.setFont(QFont('Roboto', 18, QFont.Bold))
        self.start_session_button.setMinimumHeight(50)
        self.start_session_button.clicked.connect(self.start_session)
        layout.addWidget(self.start_session_button)
        
        layout.addStretch(2)

        return widget

    def create_collection_widget(self):
        widget = QWidget()
        
        # Use a QGridLayout to overlay widgets
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # --- 1. Video Feed (Bottom layer) ---
        self.video_feed_label = QLabel()
        self.video_feed_label.setAlignment(Qt.AlignCenter)
        self.video_feed_label.setStyleSheet("background-color: #000;")
        layout.addWidget(self.video_feed_label, 0, 0)
        
        # --- 2. Overlay Container (Top layer) ---
        # This widget holds all the text overlays
        overlay_widget = QWidget()
        overlay_widget.setAttribute(Qt.WA_TransparentForMouseEvents) # Let clicks pass through
        overlay_layout = QVBoxLayout(overlay_widget)
        overlay_layout.setContentsMargins(20, 20, 20, 20)
        
        # Top-left "RECORDING" text
        top_layout = QHBoxLayout()
        self.recording_label = QLabel("🔴 RECORDING")
        self.recording_label.setObjectName("LabelRecording")
        self.recording_label.setVisible(False)
        top_layout.addWidget(self.recording_label, 0, Qt.AlignTop | Qt.AlignLeft)
        overlay_layout.addLayout(top_layout)
        
        overlay_layout.addStretch()
        
        # Center text (Countdown, "Press S", "Processing")
        self.center_text_label = QLabel("PRESS 'S' TO START")
        self.center_text_label.setObjectName("LabelBigOverlay")
        self.center_text_label.setAlignment(Qt.AlignCenter)
        self.center_text_label.setVisible(False)
        overlay_layout.addWidget(self.center_text_label)
        
        overlay_layout.addStretch()
        
        # --- 3. Bottom Status Bar (Separate from overlay) ---
        status_bar_widget = QWidget()
        status_bar_widget.setFixedHeight(60)
        status_bar_widget.setStyleSheet("background-color: rgba(0, 0, 0, 0.5);")
        status_layout = QHBoxLayout(status_bar_widget)
        
        self.status_text_label = QLabel("Status: Ready")
        status_layout.addWidget(self.status_text_label, 1)
        
        self.start_batch_button = QPushButton("START BATCH (S)")
        self.start_batch_button.setObjectName("ButtonGreen")
        self.start_batch_button.clicked.connect(self.start_batch_countdown)
        status_layout.addWidget(self.start_batch_button)
        
        self.quit_button = QPushButton("QUIT (Q)")
        self.quit_button.setObjectName("ButtonGray")
        self.quit_button.clicked.connect(self.stop_session)
        status_layout.addWidget(self.quit_button)
        
        # Add the overlay and status bar to the grid
        layout.addWidget(overlay_widget, 0, 0) # Overlays on top of video
        layout.addWidget(status_bar_widget, 0, 0, Qt.AlignBottom) # Status bar at bottom
        
        return widget

    # --- UI Logic ---
    
    def load_existing_actions(self):
        self.action_list.clear()
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        
        try:
            actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
            actions.sort()
            
            # Get video counts
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
        action_name = item.text().split(' (')[0] # Get name without count
        self.action_name_input.setText(action_name)

    # --- State Machine & Session Logic ---

    def start_session(self):
        self.action_name = self.action_name_input.text().strip()
        self.num_videos = self.num_videos_input.value()
        
        if not self.action_name:
            QMessageBox.warning(self, "Error", "Please enter an action name.")
            return
            
        # --- Create directories (from your script) ---
        self.action_video_dir = os.path.join(DATA_PATH, self.action_name)
        self.action_landmark_dir = os.path.join(OUTPUT_PATH, self.action_name)
        os.makedirs(self.action_video_dir, exist_ok=True)
        os.makedirs(self.action_landmark_dir, exist_ok=True)
        
        # --- Find starting number (from your script) ---
        self.start_num = 0
        while True:
            video_file_exists = any(
                os.path.exists(os.path.join(self.action_video_dir, f"{self.start_num}{ext}")) 
                for ext in ['.mp4', '.webm', '.avi', '.mov']
            )
            if not video_file_exists:
                break
            self.start_num += 1
        print(f"Starting video number will be: {self.start_num}")
        self.current_video_num = self.start_num
        
        # --- Update Sidebar ---
        self.session_action_label.setText(f"Action: {self.action_name}")
        self.session_progress_bar.setRange(0, self.num_videos)
        self.session_progress_bar.setValue(0)
        self.stop_session_button.setDisabled(False)
        self.action_list.setDisabled(True)
        self.action_search.setDisabled(True)

        # --- Start Camera ---
        cam_index = self.cam_select.currentIndex()
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", f"Could not open webcam {cam_index}.")
            return
            
        self.camera_timer.start(33) # ~30 FPS

        # --- Switch to Collection Screen ---
        self.stacked_widget.setCurrentWidget(self.collection_widget)
        self.set_state(STATE_WAITING_FOR_BATCH)

    def stop_session(self):
        self.camera_timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.processing_thread:
            self.processing_thread.quit()
            self.processing_thread.wait()
            
        self.load_existing_actions()
        self.stacked_widget.setCurrentWidget(self.setup_widget)
        
        # Reset sidebar
        self.session_action_label.setText("Action: N/A")
        self.session_progress_bar.setValue(0)
        self.stop_session_button.setDisabled(True)
        self.action_list.setDisabled(False)
        self.action_search.setDisabled(False)
        
        self.set_state(STATE_SETUP)
        print("Session stopped.")

    def set_state(self, new_state):
        self.app_state = new_state
        print(f"New state: {new_state}")
        
        # --- Manage UI visibility based on state ---
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
            self.start_countdown(5, STATE_PAUSE_COUNTDOWN) # 5 sec countdown, then go to pause
            
        elif new_state == STATE_PAUSE_COUNTDOWN:
            self.status_text_label.setText(f"Starting video {self.current_video_num}...")
            self.center_text_label.setObjectName("LabelCountdown")
            self.start_countdown(2, STATE_RECORDING) # 2 sec countdown, then go to recording
            
        elif new_state == STATE_RECORDING:
            self.recording_label.setVisible(True)
            self.status_text_label.setText(f"Recording video {self.current_video_num}...")
            self.record_start_time = time.time()
            self.init_video_writer()

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
        
        # Use a QTimer for the countdown
        if self.countdown_timer:
            self.countdown_timer.stop()
        
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
            
    def init_video_writer(self):
        if not self.cap:
            return
            
        video_name = str(self.current_video_num)
        self.video_save_path = os.path.join(self.action_video_dir, f"{video_name}.mp4")
        self.landmark_save_folder = os.path.join(self.action_landmark_dir, video_name)
        os.makedirs(self.landmark_save_folder, exist_ok=True)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20.0 # Hard-coded from your script
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.video_save_path, fourcc, fps, (width, height))
        print(f"VideoWriter initialized for {self.video_save_path}")

    # --- Main Camera Loop ---
    
    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to grab frame from webcam.")
            return

        # --- Apply logic from your script ---
        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, self.holistic)
        draw_styled_landmarks(image, results)
        
        # --- State-specific actions ---
        if self.app_state == STATE_RECORDING:
            elapsed = time.time() - self.record_start_time
            
            # Show timer on frame
            cv2.putText(image, f"0:0{int(elapsed)+1}", (150, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            if elapsed < RECORD_SECONDS:
                if self.video_writer:
                    self.video_writer.write(frame) # Write the *flipped* frame
            else:
                # Time is up!
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                    print(f"Video saved: {self.video_save_path}")
                
                self.set_state(STATE_PROCESSING)
                
        # --- Convert frame to QPixmap and display it ---
        qt_image = self.convert_cv_to_pixmap(image)
        self.video_feed_label.setPixmap(qt_image)
        
    def convert_cv_to_pixmap(self, cv_img):
        """Converts an OpenCV image (NumPy array) to a QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_feed_label.width(), self.video_feed_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    # --- Processing ---

    def start_processing(self):
        print("Starting processing thread...")
        self.processing_thread = ProcessingThread(
            self.video_save_path, 
            self.landmark_save_folder,
            self.holistic # This is just passed, thread will create its own
        )
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
            # Check if it's time for a new batch
            if progress % self.batch_size == 0:
                self.set_state(STATE_WAITING_FOR_BATCH)
            else:
                self.set_state(STATE_PAUSE_COUNTDOWN)

    # --- Event Handlers ---

    def keyPressEvent(self, event):
        """Handle global key presses."""
        if event.key() == Qt.Key_Q:
            if self.app_state != STATE_SETUP:
                self.stop_session()
            else:
                self.close()
                
        if event.key() == Qt.Key_S:
            if self.app_state == STATE_WAITING_FOR_BATCH:
                self.start_batch_countdown()

    def closeEvent(self, event):
        """Ensure everything is cleaned up on exit."""
        self.stop_session()
        if self.holistic:
            self.holistic.close()
        event.accept()

# --- 7. RUN APPLICATION ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = CollectionApp()
    main_window.show()
    sys.exit(app.exec_())