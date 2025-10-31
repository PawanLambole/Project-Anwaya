from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QLineEdit, QSpinBox, QListWidget, QFrame, QProgressBar, 
    QSizePolicy, QComboBox, QGridLayout, QSpacerItem, QTextEdit
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

def create_sidebar(main_window):
    """
    Creates the sidebar widget.
    'main_window' is the instance of CollectionApp
    """
    sidebar = QWidget()
    sidebar.setObjectName("Sidebar")
    sidebar.setFixedWidth(250)
    sidebar_layout = QVBoxLayout(sidebar)
    sidebar_layout.setContentsMargins(10, 10, 10, 10)
    sidebar_layout.setSpacing(10)

    title = QLabel("PROJECT ANWAYA")
    title.setAlignment(Qt.AlignCenter)
    sidebar_layout.addWidget(title)
    
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    sidebar_layout.addWidget(line)

    sidebar_layout.addWidget(QLabel("ACTIONS", objectName="SidebarTitle"))
    main_window.action_search = QLineEdit()
    main_window.action_search.setPlaceholderText("Search actions...")
    main_window.action_search.textChanged.connect(main_window.filter_actions)
    sidebar_layout.addWidget(main_window.action_search)

    main_window.action_list = QListWidget()
    main_window.action_list.itemClicked.connect(main_window.on_action_clicked)
    sidebar_layout.addWidget(main_window.action_list, 1)

    sidebar_layout.addWidget(QLabel("SESSION", objectName="SidebarTitle"))
    main_window.session_action_label = QLabel("Action: N/A")
    sidebar_layout.addWidget(main_window.session_action_label)
    
    main_window.session_progress_bar = QProgressBar()
    main_window.session_progress_bar.setValue(0)
    sidebar_layout.addWidget(main_window.session_progress_bar)

    sidebar_layout.addWidget(QLabel("SETTINGS", objectName="SidebarTitle"))
    cam_layout = QHBoxLayout()
    cam_layout.addWidget(QLabel("Camera:"))
    main_window.cam_select = QComboBox()
    main_window.cam_select.addItems(["Webcam 0", "Webcam 1"])
    cam_layout.addWidget(main_window.cam_select)
    sidebar_layout.addLayout(cam_layout)

    rec_layout = QHBoxLayout()
    rec_layout.addWidget(QLabel("Rec. Time (s):"))
    main_window.rec_time_spin = QSpinBox()
    main_window.rec_time_spin.setRange(1, 10)
    from mediapipe_logic import RECORD_SECONDS
    main_window.rec_time_spin.setValue(RECORD_SECONDS)
    rec_layout.addWidget(main_window.rec_time_spin)
    sidebar_layout.addLayout(rec_layout)

    sidebar_layout.addStretch()
    
    # --- NEW: Train Model Button ---
    main_window.train_model_button = QPushButton("Train New Model")
    main_window.train_model_button.setObjectName("ButtonTrain")
    main_window.train_model_button.clicked.connect(main_window.go_to_training)
    sidebar_layout.addWidget(main_window.train_model_button)
    # --- END NEW ---

    main_window.stop_session_button = QPushButton("STOP SESSION")
    main_window.stop_session_button.setObjectName("ButtonStop")
    main_window.stop_session_button.clicked.connect(main_window.stop_session)
    main_window.stop_session_button.setDisabled(True)
    sidebar_layout.addWidget(main_window.stop_session_button)

    return sidebar

def create_home_widget(main_window):
    """This is your Homepage UI."""
    widget = QWidget()
    widget.setObjectName("Homepage")
    
    layout = QVBoxLayout(widget)
    layout.setAlignment(Qt.AlignCenter)
    layout.setSpacing(10)
    
    layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    title = QLabel("Project ANWAYA")
    title.setObjectName("LabelHomeTitle")
    title.setAlignment(Qt.AlignCenter)
    layout.addWidget(title)

    tagline = QLabel("Indian Sign Language to Marathi Translation")
    tagline.setObjectName("LabelHomeTagline")
    tagline.setAlignment(Qt.AlignCenter)
    layout.addWidget(tagline)

    button_layout = QVBoxLayout()
    button_layout.setObjectName("HomeButtonLayout")
    button_layout.setAlignment(Qt.AlignCenter)
    button_layout.setSpacing(15)
    
    main_window.start_collection_button = QPushButton("Start New Collection")
    main_window.start_collection_button.clicked.connect(main_window.go_to_setup)
    button_layout.addWidget(main_window.start_collection_button)

    main_window.recognize_btn = QPushButton("🤟 Real-Time Recognition")
    main_window.recognize_btn.setObjectName("recognizeButton")
    main_window.recognize_btn.clicked.connect(main_window.go_to_recognition)
    button_layout.addWidget(main_window.recognize_btn)

    main_window.manage_data_button = QPushButton("Manage Data (Coming Soon)")
    main_window.manage_data_button.setObjectName("ButtonDisabled")
    main_window.manage_data_button.setDisabled(True)
    button_layout.addWidget(main_window.manage_data_button)

    main_window.quit_home_button = QPushButton("Quit Application")
    main_window.quit_home_button.setObjectName("ButtonGray")
    main_window.quit_home_button.clicked.connect(main_window.close)
    button_layout.addWidget(main_window.quit_home_button)
    
    layout.addLayout(button_layout)
    layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
    
    footer = QLabel("B.Tech Final Year Project")
    footer.setObjectName("LabelHelper")
    footer.setAlignment(Qt.AlignCenter)
    layout.addWidget(footer)
    return widget

def create_setup_widget(main_window):
    widget = QWidget()
    widget.setObjectName("SetupWidget")
    layout = QVBoxLayout(widget)
    layout.setAlignment(Qt.AlignCenter)
    layout.setSpacing(15)

    title = QLabel("Setup New Collection Session")
    title.setFont(QFont('Roboto', 24, QFont.Bold))
    title.setAlignment(Qt.AlignCenter)
    layout.addWidget(title)
    layout.addStretch(1)

    layout.addWidget(QLabel("Action Name (Marathi)"))
    main_window.action_name_input = QLineEdit()
    main_window.action_name_input.setPlaceholderText("e.g., आभार")
    main_window.action_name_input.setFont(QFont('Roboto', 16))
    layout.addWidget(main_window.action_name_input)
    layout.addWidget(QLabel("(Select from list or type new)", objectName="LabelHelper"))

    layout.addWidget(QLabel("Number of Videos to Record"))
    main_window.num_videos_input = QSpinBox()
    main_window.num_videos_input.setRange(1, 1000)
    main_window.num_videos_input.setValue(50)
    main_window.num_videos_input.setFont(QFont('Roboto', 16))
    layout.addWidget(main_window.num_videos_input)
    layout.addStretch(1)

    setup_button_layout = QHBoxLayout()
    main_window.back_button = QPushButton("Back to Home")
    main_window.back_button.setObjectName("ButtonGray")
    main_window.back_button.clicked.connect(main_window.go_to_home)
    setup_button_layout.addWidget(main_window.back_button)

    main_window.start_session_button = QPushButton("START SESSION")
    main_window.start_session_button.setFont(QFont('Roboto', 18, QFont.Bold))
    main_window.start_session_button.setMinimumHeight(50)
    main_window.start_session_button.clicked.connect(main_window.start_session)
    setup_button_layout.addWidget(main_window.start_session_button, 1)
    
    layout.addLayout(setup_button_layout)
    layout.addStretch(2)
    return widget

def create_recognition_widget(main_window):
    """
    Creates the Recognition (Real-time Translation) Page UI
    """
    widget = QWidget()
    widget.setObjectName("RecognitionPage")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(15)

    # Header with back button
    header_layout = QHBoxLayout()
    main_window.recognition_back_btn = QPushButton("← Back to Home")
    main_window.recognition_back_btn.setObjectName("ButtonGray")
    header_layout.addWidget(main_window.recognition_back_btn)
    
    header_title = QLabel("Real-Time ISL Recognition")
    header_title.setObjectName("RecognitionHeaderTitle")
    header_title.setAlignment(Qt.AlignCenter)
    header_layout.addWidget(header_title, 1)
    
    header_layout.addWidget(QLabel())  # Spacer for symmetry
    header_layout.itemAt(2).widget().setFixedWidth(main_window.recognition_back_btn.sizeHint().width())
    
    layout.addLayout(header_layout)

    # Video display area
    main_window.recognition_video_label = QLabel()
    main_window.recognition_video_label.setObjectName("videoLabel")
    main_window.recognition_video_label.setAlignment(Qt.AlignCenter)
    main_window.recognition_video_label.setMinimumSize(640, 480)
    main_window.recognition_video_label.setStyleSheet("""
        QLabel#videoLabel {
            background-color: #1a1a1a;
            border: 2px solid #0078D4;
            border-radius: 10px;
        }
    """)
    main_window.recognition_video_label.setText("Camera Stopped")
    layout.addWidget(main_window.recognition_video_label, 1)

    # Prediction display banner
    prediction_banner = QWidget()
    prediction_banner.setObjectName("predictionBanner")
    prediction_banner.setFixedHeight(60)
    prediction_layout = QHBoxLayout(prediction_banner)
    prediction_layout.setContentsMargins(20, 0, 20, 0)
    
    main_window.recognition_prediction_label = QLabel("Ready to recognize...")
    main_window.recognition_prediction_label.setObjectName("predictionLabel")
    main_window.recognition_prediction_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    
    main_window.recognition_confidence_label = QLabel("")
    main_window.recognition_confidence_label.setObjectName("confidenceLabel")
    main_window.recognition_confidence_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    
    prediction_layout.addWidget(main_window.recognition_prediction_label)
    prediction_layout.addWidget(main_window.recognition_confidence_label)
    
    layout.addWidget(prediction_banner)

    # Control buttons
    control_layout = QHBoxLayout()
    control_layout.addStretch()
    
    main_window.recognition_start_btn = QPushButton("Start Camera")
    main_window.recognition_start_btn.setObjectName("startRecognitionButton")
    main_window.recognition_start_btn.setMinimumHeight(50)
    main_window.recognition_start_btn.setMinimumWidth(200)
    control_layout.addWidget(main_window.recognition_start_btn)
    
    main_window.recognition_stop_btn = QPushButton("Stop Camera")
    main_window.recognition_stop_btn.setObjectName("stopRecognitionButton")
    main_window.recognition_stop_btn.setMinimumHeight(50)
    main_window.recognition_stop_btn.setMinimumWidth(200)
    main_window.recognition_stop_btn.setEnabled(False)
    control_layout.addWidget(main_window.recognition_stop_btn)
    
    control_layout.addStretch()
    layout.addLayout(control_layout)

    # Status indicator
    status_layout = QHBoxLayout()
    
    main_window.recognition_status_indicator = QLabel("●")
    main_window.recognition_status_indicator.setObjectName("statusIndicator")
    main_window.recognition_status_indicator.setStyleSheet("color: #666; font-size: 20px;")
    
    main_window.recognition_status_text = QLabel("Camera stopped")
    main_window.recognition_status_text.setObjectName("statusText")
    main_window.recognition_status_text.setStyleSheet("color: #666; font-size: 14px;")
    
    status_layout.addStretch()
    status_layout.addWidget(main_window.recognition_status_indicator)
    status_layout.addWidget(main_window.recognition_status_text)
    status_layout.addStretch()
    
    layout.addLayout(status_layout)
    
    return widget

def create_collection_widget(main_window):
    widget = QWidget()
    layout = QGridLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    
    main_window.video_feed_label = QLabel()
    main_window.video_feed_label.setAlignment(Qt.AlignCenter)
    main_window.video_feed_label.setStyleSheet("background-color: #000;")
    layout.addWidget(main_window.video_feed_label, 0, 0)
    
    overlay_widget = QWidget()
    overlay_widget.setAttribute(Qt.WA_TransparentForMouseEvents)
    overlay_layout = QVBoxLayout(overlay_widget)
    overlay_layout.setContentsMargins(20, 20, 20, 20)
    
    top_layout = QHBoxLayout()
    main_window.recording_label = QLabel("🔴 RECORDING")
    main_window.recording_label.setObjectName("LabelRecording")
    main_window.recording_label.setVisible(False)
    top_layout.addWidget(main_window.recording_label, 0, Qt.AlignTop | Qt.AlignLeft)
    overlay_layout.addLayout(top_layout)
    
    overlay_layout.addStretch()
    main_window.center_text_label = QLabel("PRESS 'S' TO START")
    main_window.center_text_label.setObjectName("LabelBigOverlay")
    main_window.center_text_label.setAlignment(Qt.AlignCenter)
    main_window.center_text_label.setVisible(False)
    overlay_layout.addWidget(main_window.center_text_label)
    overlay_layout.addStretch()
    
    status_bar_widget = QWidget()
    status_bar_widget.setFixedHeight(60)
    status_bar_widget.setStyleSheet("background-color: rgba(0, 0, 0, 0.5);")
    status_layout = QHBoxLayout(status_bar_widget)
    
    main_window.status_text_label = QLabel("Status: Ready")
    status_layout.addWidget(main_window.status_text_label, 1)
    
    main_window.start_batch_button = QPushButton("START BATCH (S)")
    main_window.start_batch_button.setObjectName("ButtonGreen")
    main_window.start_batch_button.clicked.connect(main_window.start_batch_countdown)
    status_layout.addWidget(main_window.start_batch_button)
    
    main_window.quit_button = QPushButton("STOP (Q)")
    main_window.quit_button.setObjectName("ButtonStop")
    main_window.quit_button.clicked.connect(main_window.stop_session)
    status_layout.addWidget(main_window.quit_button)
    
    layout.addWidget(overlay_widget, 0, 0)
    layout.addWidget(status_bar_widget, 0, 0, Qt.AlignBottom)
    return widget

# --- NEW: Training Page UI ---
def create_training_widget(main_window):
    """
    Creates the model training page.
    """
    widget = QWidget()
    widget.setObjectName("TrainingPage")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(15)

    title = QLabel("Model Training in Progress...")
    title.setObjectName("TrainingTitle")
    title.setAlignment(Qt.AlignCenter)
    layout.addWidget(title)

    # Log display
    main_window.training_log_display = QTextEdit()
    main_window.training_log_display.setObjectName("TrainingLog")
    main_window.training_log_display.setReadOnly(True)
    layout.addWidget(main_window.training_log_display, 1) # Stretch

    # Button layout
    button_layout = QHBoxLayout()
    button_layout.addStretch()
    main_window.training_back_button = QPushButton("Back to Home")
    main_window.training_back_button.setObjectName("ButtonGray")
    main_window.training_back_button.clicked.connect(main_window.go_to_home)
    main_window.training_back_button.setDisabled(True) # Disabled during training
    button_layout.addWidget(main_window.training_back_button)
    button_layout.addStretch()
    
    layout.addLayout(button_layout)
    return widget