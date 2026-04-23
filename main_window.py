import cv2

import numpy as np
import os
import time
import sys
import json
import subprocess # <-- NEW: For running external script
import re # <-- NEW: For parsing training logs
import ctypes
import platform

# Windows-specific imports (conditional)
if platform.system() == "Windows":
    try:
        from ctypes.wintypes import MSG
        import win32con
        import win32api
        import win32gui
        HAS_WIN32 = True
    except ImportError:
        print("Warning: Windows-specific modules (pywin32) not available. Some features may be limited.")
        HAS_WIN32 = False
else:
    HAS_WIN32 = False

from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget, QMessageBox, QSplitter, QLabel, QShortcut)
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QPoint
import google.generativeai as genai

# --- Import from our local files ---
from ui_definitions import (
    create_sidebar, create_home_widget, 
    create_setup_widget, create_collection_widget,
    create_training_widget, create_recognition_widget,
    create_manage_data_widget, create_top_menubar,
    create_batch_process_widget
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
STATE_TRAINING = 8
STATE_RECOGNITION = 9
STATE_MANAGE_DATA = 10
STATE_BATCH_PROCESS = 11

# --- Gemini API Configuration ---
GEMINI_API_KEY = ""
api_key_file = "api_key.txt"

if os.path.exists(api_key_file):
    try:
        with open(api_key_file, "r", encoding="utf-8") as f:
            GEMINI_API_KEY = f.read().strip()
        if not GEMINI_API_KEY:
            print("WARNING: api_key.txt exists but is empty. Gemini features will be disabled.")
    except Exception as e:
        print(f"ERROR reading API key from {api_key_file}: {e}")
else:
    print("NOTE: api_key.txt not found. Gemini grammar correction is disabled.")
    print("      To enable it, create api_key.txt with your Google Generative AI API key")
    print("      Get a free key at: https://aistudio.google.com/apikey")

# --- NEW: Training Thread ---
class TrainingThread(QThread):
    """
    Runs the model training script in a separate process
    and emits its stdout line by line.
    """
    log_update = pyqtSignal(str)
    load_progress_update = pyqtSignal(int, int)
    epoch_update = pyqtSignal(int, int)
    
    def __init__(self, dataset_name):
        super().__init__()
        self.process = None
        self.dataset_name = dataset_name

    def run(self):
        print("Starting training thread...")
        try:
            # We use sys.executable to ensure we use the same python interpreter
            # (e.g., from the virtual environment)
            self.process = subprocess.Popen(
                [sys.executable, '-u', 'train_model.py', '--dataset', self.dataset_name],
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
                stripped_line = line.strip()
                self.log_update.emit(stripped_line)
                
                # Parse loading progress: [LOAD_PROGRESS] 5/10
                load_match = re.search(r'\[LOAD_PROGRESS\]\s+(\d+)/(\d+)', stripped_line)
                if load_match:
                    current, total = int(load_match.group(1)), int(load_match.group(2))
                    self.load_progress_update.emit(current, total)
                
                # Parse epoch progress: Epoch 1/150
                epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', stripped_line)
                if epoch_match:
                    current, total = int(epoch_match.group(1)), int(epoch_match.group(2))
                    self.epoch_update.emit(current, total)

            self.process.stdout.close()
            self.process.wait()
            print("Training process finished.")

        except Exception as e:
            print(f"Error starting training process: {e}")
            self.log_update.emit(f"\n--- ERROR ---")
            self.log_update.emit(f"Failed to start training script: {e}")
            self.log_update.emit("Make sure 'train_model.py' is in the same folder.")
            self.log_update.emit("Ensure all requirements are installed (tensorflow, sklearn, etc.)")

    def request_graceful_stop(self):
        # Create the stop_training.flag file
        model_dir = os.path.join('model', self.dataset_name)
        os.makedirs(model_dir, exist_ok=True)
        stop_flag_path = os.path.join(model_dir, 'stop_training.flag')
        
        try:
            with open(stop_flag_path, 'w') as f:
                f.write('STOP')
            print(f"Graceful stop requested. Flag created at {stop_flag_path}")
            self.log_update.emit(f"\n[INFO] Graceful stop requested...")
        except Exception as e:
            print(f"Failed to create stop flag: {e}")

    def stop(self):
        if self.process and self.process.poll() is None:
            print("Terminating training process...")
            self.process.terminate()
            self.process.wait()
            print("Training process terminated.")


# --- NEW: Gemini Sentence Correction Thread ---
class GeminiCorrectionThread(QThread):
    """
    Calls Gemini API to correct Marathi sentence grammar in the background
    without blocking the UI thread.
    """
    correction_ready = pyqtSignal(str, str)  # (original_sentence, corrected_sentence)
    correction_error = pyqtSignal(str)       # error message

    def __init__(self, sentence, api_key):
        super().__init__()
        self.sentence = sentence
        self.api_key = api_key

    def run(self):
        # Check if API key is available
        if not self.api_key:
            self.correction_error.emit("Gemini API key not configured. Grammar correction disabled.")
            return
        
        try:
            # Configure Gemini API
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')

            # Create prompt for Marathi grammar correction
            prompt = f"""You are an advanced Marathi language translator. Your task is to take translated sign language sequences and turn them into natural, grammatically correct Marathi sentences.

RULES:
1. If the input is a sequence of words that is grammatically incorrect or missing verbs/subjects (e.g., "निळा रंग आवडतो" or "मी जेवण"), you MUST rephrase it into a complete, natural Marathi sentence (e.g., "मला निळा रंग आवडतो" or "मी जेवण करत आहे").
2. However, if the input is ALREADY a universally understood stand-alone greeting or short phrase (e.g., "शुभ प्रभात", "धन्यवाद", "नमस्कार", "माफ करा"), you MUST return it EXACTLY as it is without expanding it into a formal sentence. Do not say "तुमचा प्रभात शुभ असो" for "शुभ प्रभात".
3. Return ONLY the final corrected Marathi text.

Input: {self.sentence}

Corrected Marathi text:"""

            # Call Gemini API
            response = model.generate_content(prompt)
            corrected_sentence = response.text.strip()

            # Remove any trailing punctuation from the corrected sentence since we'll add it back
            if corrected_sentence.endswith('.'):
                corrected_sentence = corrected_sentence[:-1]

            # Emit the result
            self.correction_ready.emit(self.sentence, corrected_sentence)

        except Exception as e:
            error_msg = str(e)
            if "API key not valid" in error_msg or "invalid_api_key" in error_msg:
                self.correction_error.emit("Invalid API key. Please check api_key.txt")
            elif "429" in error_msg or "rate_limit" in error_msg:
                self.correction_error.emit("API rate limit exceeded. Using original sentence.")
            elif "service unavailable" in error_msg.lower() or "500" in error_msg:
                self.correction_error.emit("Gemini service unavailable. Using original sentence.")
            else:
                print(f"Gemini API error: {error_msg}")
                self.correction_error.emit(f"Grammar correction failed: {error_msg}")


class CollectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anvaya")
        # Geometry is set responsively in showEvent; use a sane placeholder
        self.setGeometry(100, 100, 1280, 720)
        self.setObjectName("MainWindow")
        
        # Start frameless, but we'll add standard window styles back in showEvent
        # so Windows 11 snap features work natively.
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

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
        self.active_dataset = "Default"
        self.is_dark_theme = False
        
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
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # --- Top Menu Bar ---
        self.top_menu_bar = create_top_menubar(self)
        self.main_layout.addWidget(self.top_menu_bar)

        # Main content area with a resizable QSplitter
        self.content_splitter = QSplitter(Qt.Horizontal)
        # Give the splitter an ID prefix to target it in stylesheets easily if needed
        self.content_splitter.setObjectName("ContentSplitter")
        self.main_layout.addWidget(self.content_splitter)

        # Create UI from definitions
        self.sidebar = create_sidebar(self)
        
        # Connect the hamburger toggle button (which is in the top_menu_bar)
        if hasattr(self, 'sidebar_toggle_btn'):
            try:
                self.sidebar_toggle_btn.clicked.disconnect()
            except TypeError:
                pass
            self.sidebar_toggle_btn.clicked.connect(self.toggle_sidebar)
        
        self.main_content = QWidget()
        self.main_content.setObjectName("MainContent")
        self.main_content_layout = QVBoxLayout(self.main_content)
        
        # Add widgets to the splitter
        self.content_splitter.addWidget(self.sidebar)
        self.content_splitter.addWidget(self.main_content)

        # Initial splitter proportion: sidebar 22%, main content 78%
        # Will be recalculated after showEvent centers/resizes the window
        self.content_splitter.setSizes([280, 1000])

        self.stacked_widget = QStackedWidget()
        self.main_content_layout.addWidget(self.stacked_widget)

        self.home_widget = create_home_widget(self)
        self.setup_widget = create_setup_widget(self)
        self.collection_widget = create_collection_widget(self)
        self.training_widget = create_training_widget(self)
        self.recognition_widget = create_recognition_widget(self)
        self.manage_data_widget = create_manage_data_widget(self)
        self.batch_process_widget = create_batch_process_widget(self)
        
        self.stacked_widget.addWidget(self.home_widget)
        self.stacked_widget.addWidget(self.setup_widget)
        self.stacked_widget.addWidget(self.collection_widget)
        self.stacked_widget.addWidget(self.training_widget)
        self.stacked_widget.addWidget(self.recognition_widget)
        self.stacked_widget.addWidget(self.manage_data_widget)
        self.stacked_widget.addWidget(self.batch_process_widget)

        # Connect signals
        self.rec_time_spin.valueChanged.connect(self.update_record_time)
        self.content_splitter.splitterMoved.connect(self.on_splitter_moved)
        
        # Connect recognition signals
        self.recognition_back_btn.clicked.connect(self.go_to_home)
        self.recognition_start_btn.clicked.connect(self.start_recognition)
        self.recognition_stop_btn.clicked.connect(self.stop_recognition)
        self.sentence_backspace_btn.clicked.connect(self.sentence_backspace)
        self.sentence_clear_btn.clicked.connect(self.sentence_clear)
        
        # Connect manage data signals
        self.manage_data_back_btn.clicked.connect(self.go_to_home)
        self.manage_actions_list.itemClicked.connect(self.on_manage_action_selected)
        self.view_videos_btn.clicked.connect(self.view_action_videos)
        self.delete_action_btn.clicked.connect(self.delete_action_data)
        self.export_data_btn.clicked.connect(self.export_data_info)
        self.refresh_actions_btn.clicked.connect(self.refresh_actions_data)
        self.restart_app_btn.clicked.connect(self.restart_application)

        # Connect batch process signals
        self.batch_back_btn.clicked.connect(self.go_to_home)
        self.batch_scan_btn.clicked.connect(self.scan_batch_folders)
        self.batch_start_btn.clicked.connect(self.start_batch_processing)
        self.batch_add_btn.clicked.connect(self.batch_add_folders)
        self.batch_remove_btn.clicked.connect(self.batch_remove_folders)
        
        # Connect setup screen dataset dropdown to action list refresh
        self.dataset_name_input.currentTextChanged.connect(self.on_setup_dataset_changed)
        # Connect action name input to restore saved settings when action is selected
        self.action_name_input.action_confirmed.connect(self.on_setup_action_changed)

        self.stacked_widget.setCurrentWidget(self.home_widget)

    # --- UI Logic ---
    
    def toggle_theme(self):
        from PyQt5.QtWidgets import QApplication
        self.is_dark_theme = not self.is_dark_theme
        
        stylesheet_file = "style_dark.qss" if self.is_dark_theme else "style.qss"
        try:
            with open(stylesheet_file, "r") as f:
                stylesheet = f.read()
                QApplication.instance().setStyleSheet(stylesheet)
                
            if self.is_dark_theme:
                self.theme_toggle_btn.setText("☀️ Switch to Light Theme" if getattr(self, 'sidebar_content', None) and self.sidebar_content.isVisible() else "☀️")
            else:
                self.theme_toggle_btn.setText("🌙 Switch to Dark Theme" if getattr(self, 'sidebar_content', None) and self.sidebar_content.isVisible() else "🌙")
        except Exception as e:
            print(f"Failed to load {stylesheet_file}: {e}")

    def reload_stylesheets(self):
        """Hot-reload for the CSS stylesheets."""
        from PyQt5.QtWidgets import QApplication
        stylesheet_file = "style_dark.qss" if self.is_dark_theme else "style.qss"
        try:
            with open(stylesheet_file, "r") as f:
                stylesheet = f.read()
                QApplication.instance().setStyleSheet(stylesheet)
            print(f"[DEV] Reloaded {stylesheet_file} successfully.")
        except Exception as e:
            print(f"[DEV] Failed to reload {stylesheet_file}: {e}")

    # --- Native Event Hook for Snap Assit ---
    def showEvent(self, event):
        super().showEvent(event)
        # Add basic window flags purely to fool the OS into providing Snap Assist
        try:
            hwnd = int(self.winId())
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
            style |= win32con.WS_THICKFRAME | win32con.WS_CAPTION | win32con.WS_MAXIMIZEBOX | win32con.WS_MINIMIZEBOX | win32con.WS_SYSMENU
            win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
            win32gui.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE |
                                  win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED)
        except Exception as e:
            print(f"showEvent Windows hook failed: {e}")

        # Responsive startup sizing — only runs once on first show
        if not getattr(self, '_startup_centered', False):
            self._startup_centered = True
            from PyQt5.QtWidgets import QApplication
            screen = QApplication.primaryScreen().availableGeometry()
            # Use 90% of available screen, bounded to a min of 900×600
            w = max(900, int(screen.width() * 0.90))
            h = max(600, int(screen.height() * 0.90))
            x = screen.x() + (screen.width() - w) // 2
            y = screen.y() + (screen.height() - h) // 2
            self.setGeometry(x, y, w, h)
            # Set splitter proportionally based on actual window width
            sidebar_w = max(220, int(w * 0.22))
            self.content_splitter.setSizes([sidebar_w, w - sidebar_w])

    def nativeEvent(self, eventType, message):
        try:
            msg = MSG.from_address(message.__int__())
            if msg.message == win32con.WM_NCCALCSIZE:
                # Remove the caption that we just added so it's a true frameless window
                if msg.wParam:
                    return True, 0
            
            elif msg.message == win32con.WM_NCHITTEST:
                # Get current mouse coords from the OS
                x = win32api.LOWORD(msg.lParam)
                if x & 0x8000: x -= 0x10000
                    
                y = win32api.HIWORD(msg.lParam)
                if y & 0x8000: y -= 0x10000
                
                # Convert coords relative to our application
                local_pos = self.mapFromGlobal(QPoint(x, y))
                hx = local_pos.x()
                hy = local_pos.y()
                
                bw = 8 # Border snapping width
                
                left = (hx < bw)
                right = (hx > self.width() - bw)
                top = (hy < bw)
                bottom = (hy > self.height() - bw)
                
                if top and left: return True, win32con.HTTOPLEFT
                elif top and right: return True, win32con.HTTOPRIGHT
                elif bottom and left: return True, win32con.HTBOTTOMLEFT
                elif bottom and right: return True, win32con.HTBOTTOMRIGHT
                elif top: return True, win32con.HTTOP
                elif bottom: return True, win32con.HTBOTTOM
                elif left: return True, win32con.HTLEFT
                elif right: return True, win32con.HTRIGHT
                
                # Titlebar drag region (custom title bar area -> TopMenuBar)
                # Treat the top 50 pixels specifically as dragging
                if hy < 50:
                    # Check if the mouse is hovering over a button in the top menu bar
                    global_pos = QPoint(x, y)
                    local_pos = self.top_menu_bar.mapFromGlobal(global_pos)
                    child = self.top_menu_bar.childAt(local_pos)
                    
                    if child and isinstance(child, QPushButton):
                        return True, win32con.HTCLIENT
                    
                    # Otherwise, treat the rest of the top 50px as the drag caption
                    return True, win32con.HTCAPTION
                        
                return True, win32con.HTCLIENT
        except Exception as e:
            pass # Failsafe against incorrect hook resolution
            
        return super().nativeEvent(eventType, message)

    def toggle_sidebar(self):
        print("Toggling sidebar...")
        is_expanded = self.sidebar_content.isVisible()
        total = sum(self.content_splitter.sizes())

        if is_expanded:
            # Save the current sidebar width so we can restore it exactly later
            self._saved_sidebar_width = self.content_splitter.sizes()[0]
            self.sidebar_content.setVisible(False)
            collapsed_w = 60
            self.content_splitter.setSizes([collapsed_w, max(0, total - collapsed_w)])
            if self.is_dark_theme:
                self.theme_toggle_btn.setText("☀️")
            else:
                self.theme_toggle_btn.setText("🌙")
            self.train_model_button.setText("📈")
            self.stop_session_button.setText("⏹️")
        else:
            self.sidebar_content.setVisible(True)
            # Restore to saved width, or default proportionally (22% of window width)
            default_sidebar_w = max(220, int(self.width() * 0.22))
            restore_w = getattr(self, '_saved_sidebar_width', default_sidebar_w)
            self.content_splitter.setSizes([restore_w, max(0, total - restore_w)])
            if self.is_dark_theme:
                self.theme_toggle_btn.setText("☀️ Switch to Light Theme")
            else:
                self.theme_toggle_btn.setText("🌙 Switch to Dark Theme")
            self.train_model_button.setText("📈 Train New Model")
            self.stop_session_button.setText("⏹️ STOP SESSION")
            
    def on_splitter_moved(self, pos, index):
        if not self.sidebar_content.isVisible():
            return
        sidebar_width = self.sidebar.width()
        if sidebar_width > 220:
            self.rec_time_label.setText("Recording Time (s):")
        else:
            self.rec_time_label.setText("Rec. Time (s):")

    def update_record_time(self, value):
        self.current_record_seconds = value
        print(f"Record time updated to: {self.current_record_seconds}s")
    
    def load_existing_actions(self):
        self.action_list.clear()
        self.dataset_name_input.clear()
        self.model_select.blockSignals(True)
        self.model_select.clear()
        
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
        try:
            datasets = [d for d in os.listdir(OUTPUT_PATH) if os.path.isdir(os.path.join(OUTPUT_PATH, d))]
            datasets.sort()
            
            if not datasets:
                datasets = ["Default"]
            
            self.dataset_name_input.addItems(datasets)
            self.model_select.addItems(datasets)
            self.training_dataset_dropdown.clear()
            self.training_dataset_dropdown.addItems(datasets)
            
            if hasattr(self, 'connected_model_input'):
                self.connected_model_input.clear()
                self.connected_model_input.addItem("None")
                self.connected_model_input.addItems(datasets)
            
            if self.active_dataset in datasets:
                self.model_select.setCurrentText(self.active_dataset)
                self.dataset_name_input.setCurrentText(self.active_dataset)
            else:
                self.active_dataset = datasets[0]
                self.model_select.setCurrentText(self.active_dataset)
                self.dataset_name_input.setCurrentText(self.active_dataset)
                
            dataset_path = os.path.join(OUTPUT_PATH, self.active_dataset)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
                
            actions = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
            actions.sort()
            action_items = []
            for action in actions:
                vid_count = len([d for d in os.listdir(os.path.join(dataset_path, action)) if os.path.isdir(os.path.join(dataset_path, action, d))])
                action_items.append(f"{action} ({vid_count})")
            self.action_list.addItems(action_items)

            # Populate the action name combo with existing action names
            if hasattr(self, 'action_name_input'):
                self.action_name_input.set_action_list(actions)
        except Exception as e:
            print(f"Error loading actions: {e}")
        finally:
            self.model_select.blockSignals(False)
            
    def on_model_changed(self, index):
        self.active_dataset = self.model_select.currentText()
        self.load_existing_actions()

    def on_setup_dataset_changed(self, text):
        """Refresh the action name dropdown when the setup screen's dataset dropdown changes.
        This is independent of the sidebar model so the user can set up
        a collection for any dataset, not just the one active in the sidebar."""
        dataset_name = text.strip()
        if not dataset_name:
            return
        dataset_path = os.path.join(OUTPUT_PATH, dataset_name)
        if not os.path.exists(dataset_path):
            # New dataset — no existing actions yet
            if hasattr(self, 'action_name_input'):
                self.action_name_input.set_action_list([])
            return
        try:
            actions = sorted([d for d in os.listdir(dataset_path)
                              if os.path.isdir(os.path.join(dataset_path, d))])
            if hasattr(self, 'action_name_input'):
                self.action_name_input.set_action_list(actions)
        except Exception as e:
            print(f"[on_setup_dataset_changed] Error: {e}")

    def on_setup_action_changed(self, action_name):
        """When an action is selected in the setup screen, pre-populate the
        Connect-to-Model and Terminator dropdowns with the already-saved settings."""
        action_name = action_name.strip()
        if not action_name:
            return
        dataset_name = self.dataset_name_input.currentText().strip()
        if not dataset_name:
            return
        
        model_dir = os.path.join('model', dataset_name)
        
        # --- Restore connected model ---
        action_configs = {}
        config_path = os.path.join(model_dir, 'action_configs.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    action_configs = json.load(f)
            except Exception:
                pass
        connected_model = action_configs.get(action_name, "None")
        if hasattr(self, 'connected_model_input'):
            idx = self.connected_model_input.findText(connected_model)
            if idx >= 0:
                self.connected_model_input.setCurrentIndex(idx)
            else:
                self.connected_model_input.setCurrentIndex(0)  # default "None"
        
        # --- Restore terminator flag ---
        termination_actions = []
        term_path = os.path.join(model_dir, 'termination_actions.json')
        if os.path.exists(term_path):
            try:
                with open(term_path, 'r', encoding='utf-8') as f:
                    termination_actions = json.load(f)
            except Exception:
                pass
        is_termination = action_name in termination_actions
        if hasattr(self, 'termination_input'):
            self.termination_input.setCurrentText("Yes" if is_termination else "No")
            
        # --- Auto-detect next video number dynamically ---
        next_num = 0
        existing_nums = []
        for base_dir in ['ISL_Data', 'ISL_Processed']:
            target_dir = os.path.join(base_dir, dataset_name, action_name)
            if os.path.exists(target_dir):
                for item in os.listdir(target_dir):
                    name, _ = os.path.splitext(item)
                    if name.isdigit():
                        existing_nums.append(int(name))
                        
        if existing_nums:
            next_num = max(existing_nums) + 1
            
        if hasattr(self, 'start_video_num_input'):
            self.start_video_num_input.setValue(next_num)

    def filter_actions(self, text):
        for i in range(self.action_list.count()):
            item = self.action_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def on_action_clicked(self, item):
        action_name = item.text().split(' (')[0]
        self.action_name_input.lineEdit().setText(action_name)
        # Also fire action_confirmed so saved settings (connected model / terminator)
        # are immediately restored whenever the user clicks an action in the list.
        self.on_setup_action_changed(action_name)

    # --- Navigation ---
    
    def go_to_setup(self):
        self.stacked_widget.setCurrentWidget(self.setup_widget)
        self.set_state(STATE_SETUP)

    def go_to_recognition(self):
        """Navigate to recognition page"""
        self.stacked_widget.setCurrentWidget(self.recognition_widget)
        self.set_state(STATE_RECOGNITION)
    
    def go_to_batch_process(self):
        self.stacked_widget.setCurrentWidget(self.batch_process_widget)
        self.set_state(STATE_BATCH_PROCESS)
        # Populate dataset dropdown
        datasets = []
        if os.path.exists(DATA_PATH):
            datasets = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
            
        datasets = sorted(datasets)
        if not datasets:
            datasets = ["Default"]
        self.batch_dataset_input.clear()
        self.batch_dataset_input.addItems(datasets)
        self.batch_dataset_input.setCurrentText(self.active_dataset)
        self.scan_batch_folders()
        
    def scan_batch_folders(self):
        self.batch_folder_list.clear()
        dataset_name = self.batch_dataset_input.currentText().strip()
        if not dataset_name:
            return
            
        data_path = os.path.join(DATA_PATH, dataset_name)
        if not os.path.exists(data_path):
            return
            
        # Add folders containing MP4 videos
        for d in os.listdir(data_path):
            folder_path = os.path.join(data_path, d)
            if os.path.isdir(folder_path):
                # Count videos (case-insensitive)
                videos = [v for v in os.listdir(folder_path) if v.lower().endswith('.mp4')]
                self.batch_folder_list.addItem(f"{d} ({len(videos)} videos)")
                    
    def batch_add_folders(self):
        selected_items = self.batch_folder_list.selectedItems()
        for item in selected_items:
            # Prevent duplicates
            is_dup = False
            for i in range(self.batch_selected_list.count()):
                if self.batch_selected_list.item(i).text() == item.text():
                    is_dup = True
                    break
            if not is_dup:
                self.batch_selected_list.addItem(item.text())
            # Remove from left list if you only want it on one side at a time:
            self.batch_folder_list.takeItem(self.batch_folder_list.row(item))

    def batch_remove_folders(self):
        selected_items = self.batch_selected_list.selectedItems()
        for item in selected_items:
            self.batch_folder_list.addItem(item.text())
            self.batch_selected_list.takeItem(self.batch_selected_list.row(item))

    def start_batch_processing(self):
        if self.batch_selected_list.count() == 0:
            QMessageBox.warning(self, "No Folders Selected", "Please select at least one folder to process.")
            return
            
        dataset_name = self.batch_dataset_input.currentText().strip()
        tasks = []
        for i in range(self.batch_selected_list.count()):
            item = self.batch_selected_list.item(i)
            action_name = item.text().split(' (')[0]
            action_dir = os.path.join(DATA_PATH, dataset_name, action_name)
            videos = [v for v in os.listdir(action_dir) if v.lower().endswith('.mp4')]
            for video in videos:
                video_base = video.split('.')[0]
                tasks.append({
                    'video_path': os.path.join(action_dir, video),
                    'output_folder': os.path.join(OUTPUT_PATH, dataset_name, action_name, video_base)
                })
                
        if not tasks:
            return
            
        self.batch_start_btn.setDisabled(True)
        self.batch_scan_btn.setDisabled(True)
        self.batch_overall_progress.setRange(0, len(tasks))
        self.batch_overall_progress.setValue(0)
        
        # Clear old progress UI
        while self.batch_progress_area.count():
            item = self.batch_progress_area.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.batch_tasks = tasks
        self.batch_completed = 0
        self.active_threads = []
        if not hasattr(self, 'finished_threads'):
            self.finished_threads = []
        self.finished_threads.clear() # clear previous run's finished threads safely
        self.max_threads = self.batch_thread_spinner.value()
        
        self._spawn_next_batch_threads()
        
    def _spawn_next_batch_threads(self):
        while len(self.active_threads) < self.max_threads and self.batch_tasks:
            task = self.batch_tasks.pop(0)
            os.makedirs(task['output_folder'], exist_ok=True)
            
            # Create thread
            thread = ProcessingThread(task['video_path'], task['output_folder'], None)
            thread.finished.connect(self._on_batch_thread_completed)
            # Add a progress label
            lbl = QLabel(f"Processing: {os.path.basename(task['video_path'])}...")
            self.batch_progress_area.addWidget(lbl)
            
            # Store references
            thread._lbl = lbl
            self.active_threads.append(thread)
            thread.start()
            
    def _on_batch_thread_completed(self, output_folder):
        thread = self.sender()
        if thread in self.active_threads:
            self.active_threads.remove(thread)
            
            # Save reference to prevent GC from killing thread eagerly
            if not hasattr(self, 'finished_threads'):
                self.finished_threads = []
            self.finished_threads.append(thread)
            
            if hasattr(thread, '_lbl'):
                thread._lbl.setText(f"Completed: {os.path.basename(thread.video_path)}")
                thread._lbl.setStyleSheet("color: #34A853;")
            
            # thread.deleteLater() # Removed: Let python GC handle cleanup when we clear finished_threads list
            
        self.batch_completed += 1
        self.batch_overall_progress.setValue(self.batch_completed)
        
        if self.batch_tasks:
            self._spawn_next_batch_threads()
        elif not self.active_threads:
            self.batch_start_btn.setDisabled(False)
            self.batch_scan_btn.setDisabled(False)
            QMessageBox.information(self, "Batch Complete", "All batch processing tasks have completed successfully!")
            self.load_existing_actions()

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
        self.training_log_display.append("--- Prepare Model Training ---")
        self.training_log_display.append("Select a dataset and click Start Training.")
        
        self.train_model_button.setDisabled(False) # Make sure it's enabled if we come from somewhere else
        self.training_back_button.setDisabled(False)
        self.start_training_btn.setDisabled(False)
        
    def start_training_process(self):
        dataset_name = self.training_dataset_dropdown.currentText()
        if not dataset_name:
            QMessageBox.warning(self, "Error", "No dataset selected.")
            return

        self.training_log_display.append(f"\n--- Starting Model Training for Dataset: {dataset_name} ---")
        self.training_log_display.append(f"Using Python: {sys.executable}")
        self.training_log_display.append("This may take several minutes...")
        
        self.train_model_button.setDisabled(True)
        self.training_back_button.setDisabled(True)
        self.start_training_btn.setDisabled(True)
        self.stop_training_btn.setDisabled(False)
        
        self.loading_progress_bar.setValue(0)
        self.training_progress_bar.setValue(0)
        self.training_chart_label.setText("Training in progress...")
        self.training_chart_label.clear()

        # Start the thread with the dataset name
        self.training_thread = TrainingThread(dataset_name)
        self.training_thread.log_update.connect(self.append_training_log)
        self.training_thread.load_progress_update.connect(self.update_loading_progress)
        self.training_thread.epoch_update.connect(self.update_epoch_progress)
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.start()

    def stop_training_process(self):
        if self.training_thread and self.training_thread.isRunning():
            self.stop_training_btn.setDisabled(True)
            self.append_training_log("\n[UI] Requesting graceful stop. The model will save after the current epoch...")
            self.training_thread.request_graceful_stop()

    def update_loading_progress(self, current, total):
        self.loading_progress_bar.setRange(0, total)
        self.loading_progress_bar.setValue(current)

    def update_epoch_progress(self, current, total):
        self.training_progress_bar.setRange(0, total)
        self.training_progress_bar.setValue(current)

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
        self.start_training_btn.setDisabled(False)
        self.stop_training_btn.setDisabled(True)
        
        # Load the chart if it exists
        dataset_name = self.training_dataset_dropdown.currentText()
        chart_path = os.path.join('model report', dataset_name, 'training_history.png')
        if os.path.exists(chart_path):
            pixmap = QPixmap(chart_path)
            self.training_chart_label.setPixmap(pixmap.scaled(self.training_chart_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.training_chart_label.setText("Chart not available.")
            
        QMessageBox.information(self, "Training Complete", "The model training process has finished.")

    # --- Recognition Control Methods ---
    
    def start_recognition(self):
        """Start real-time recognition"""
        # Reset sentence when starting a new session
        self._sentence_words = []
        self.sentence_label.setText("(sentence will appear here)")

        # Use stored OS index from dropdown userData (avoids virtual cam confusion)
        cam_index = self.cam_select.currentData()
        if cam_index is None:
            cam_index = self.cam_select.currentIndex()
        self.recognition_worker = RecognitionWorker(cam_index, self.active_dataset)
        self.recognition_worker.frame_ready.connect(self.update_recognition_frame)
        self.recognition_worker.prediction_ready.connect(self.update_prediction)
        self.recognition_worker.word_committed.connect(self.on_word_committed)
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
    
    def on_word_committed(self, word, is_termination=False):
        """Called when recognition_logic commits a confirmed word."""
        if not hasattr(self, '_sentence_words'):
            self._sentence_words = []
        self._sentence_words.append(word)
        self._refresh_sentence_display()

        if is_termination:
            current_sentence = self.sentence_label.text()
            if current_sentence and current_sentence != "(sentence will appear here)":
                completed_text = current_sentence + "."
                current_panel_text = self.completed_sentences_text.toPlainText()
                if current_panel_text:
                    self.completed_sentences_text.append(completed_text)
                else:
                    self.completed_sentences_text.setText(completed_text)
                
                # Trigger Gemini correction only. Replace the final sequence.
                self._start_gemini_correction_live(current_sentence, True)

                # Clear the bottom sentence builder for the next sentence
                self.sentence_clear()

    def _refresh_sentence_display(self):
        if not hasattr(self, '_sentence_words') or not self._sentence_words:
            self.sentence_label.setText("(sentence will appear here)")
        else:
            display_text = ""
            for i, w in enumerate(self._sentence_words):
                if i == 0:
                    display_text += w
                else:
                    prev_w = self._sentence_words[i-1]
                    # If this word is 1 char and prev word is 1 char, don't add space
                    if len(w) == 1 and len(prev_w) == 1 and w.isalpha() and prev_w.isalpha():
                        display_text += w
                    else:
                        display_text += " " + w
            self.sentence_label.setText(display_text)

    def sentence_backspace(self):
        """Remove the last committed word."""
        if hasattr(self, '_sentence_words') and self._sentence_words:
            self._sentence_words.pop()
            self._refresh_sentence_display()
        if hasattr(self, 'recognition_worker') and self.recognition_worker:
            self.recognition_worker.reset_prediction_state()

    def sentence_clear(self):
        """Clear the entire sentence."""
        self._sentence_words = []
        self._refresh_sentence_display()
        if hasattr(self, 'recognition_worker') and self.recognition_worker:
            self.recognition_worker.reset_prediction_state()

    def clear_all_sentences(self):
        """Clear both the active sentence and the completed sentences history block."""
        self.sentence_clear()
        if hasattr(self, 'completed_sentences_text'):
            self.completed_sentences_text.clear()

    def _start_gemini_correction_live(self, sentence, is_termination=False):
        """Start Gemini API correction for the current sentence being built."""
        if not hasattr(self, '_correction_threads'):
            self._correction_threads = []

        # Cancel any pending correction threads for live updates (to avoid conflicts)
        if not is_termination and hasattr(self, '_live_correction_thread'):
            if self._live_correction_thread and self._live_correction_thread.isRunning():
                # Don't start a new correction if one is already running
                return

        # Create and start correction thread
        correction_thread = GeminiCorrectionThread(sentence, GEMINI_API_KEY)

        if is_termination:
            # For termination, update the completed sentences panel
            correction_thread.correction_ready.connect(self._on_correction_ready_completed)
        else:
            # For live updates, update the sentence label in real-time
            correction_thread.correction_ready.connect(self._on_correction_ready_live)
            self._live_correction_thread = correction_thread

        correction_thread.correction_error.connect(self._on_correction_error)
        correction_thread.finished.connect(lambda: self._cleanup_correction_thread(correction_thread))

        self._correction_threads.append(correction_thread)
        correction_thread.start()
        print(f"Started Gemini correction for: {sentence}")

    def _on_correction_ready_live(self, original_sentence, corrected_sentence):
        """Handle when Gemini correction is ready for live sentence updates."""
        print(f"Live Correction - Original: {original_sentence}")
        print(f"Live Correction - Corrected: {corrected_sentence}")

        # Update the live sentence display with corrected text
        current_display = self.sentence_label.text()
        if current_display == original_sentence:
            self.sentence_label.setText(corrected_sentence)
            print("✓ Live sentence corrected and updated in UI")

    def _on_correction_ready_completed(self, original_sentence, corrected_sentence):
        """Handle when Gemini correction is ready for completed sentences."""
        print(f"Completed Correction - Original: {original_sentence}")
        print(f"Completed Correction - Corrected: {corrected_sentence}")

        # Get current text from completed sentences
        current_text = self.completed_sentences_text.toPlainText()

        # Replace the original sentence with the corrected one
        # The original is stored with a period, so we search for "original."
        original_with_period = original_sentence + "."
        corrected_with_period = corrected_sentence + "."

        if original_with_period in current_text:
            updated_text = current_text.replace(original_with_period, corrected_with_period, 1)
            self.completed_sentences_text.setText(updated_text)
            print("✓ Completed sentence corrected and updated in UI")
        else:
            print("⚠ Original sentence not found in completed text")

    def _start_gemini_correction(self, sentence):
        """Start Gemini API correction for the given sentence."""
        if not hasattr(self, '_correction_threads'):
            self._correction_threads = []

        # Create and start correction thread
        correction_thread = GeminiCorrectionThread(sentence, GEMINI_API_KEY)
        correction_thread.correction_ready.connect(self._on_correction_ready)
        correction_thread.correction_error.connect(self._on_correction_error)
        correction_thread.finished.connect(lambda: self._cleanup_correction_thread(correction_thread))

        self._correction_threads.append(correction_thread)
        correction_thread.start()
        print(f"Started Gemini correction for: {sentence}")

    def _on_correction_ready(self, original_sentence, corrected_sentence):
        """Handle when Gemini correction is ready."""
        print(f"Original: {original_sentence}")
        print(f"Corrected: {corrected_sentence}")

        # Get current text from completed sentences
        current_text = self.completed_sentences_text.toPlainText()

        # Replace the original sentence with the corrected one
        # The original is stored with a period, so we search for "original."
        original_with_period = original_sentence + "."
        corrected_with_period = corrected_sentence + "."

        if original_with_period in current_text:
            updated_text = current_text.replace(original_with_period, corrected_with_period, 1)
            self.completed_sentences_text.setText(updated_text)
            print("✓ Sentence corrected and updated in UI")
        else:
            print("⚠ Original sentence not found in completed text")

    def _on_correction_error(self, error_message):
        """Handle Gemini correction errors."""
        print(f"Gemini correction error: {error_message}")
        # Optionally show a non-intrusive notification to the user
        # For now, just log it

    def _cleanup_correction_thread(self, thread):
        """Clean up finished correction thread."""
        if hasattr(self, '_correction_threads') and thread in self._correction_threads:
            self._correction_threads.remove(thread)

    def update_recognition_frame(self, qt_image):
        """Update video frame display - scale to fit the label"""
        pixmap = QPixmap.fromImage(qt_image)
        label_w = self.recognition_video_label.width()
        label_h = self.recognition_video_label.height()
        if label_w > 0 and label_h > 0:
            pixmap = pixmap.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.recognition_video_label.setPixmap(pixmap)
    
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

    def save_connection_only(self):
        dataset_name = self.dataset_name_input.currentText().strip()
        if not dataset_name:
            dataset_name = "Default"
            
        self.action_name = self.action_name_input.currentText().strip()
        if not self.action_name:
            QMessageBox.warning(self, "Error", "Please enter an action name.")
            return
            
        import json
        config_path = os.path.join('model', dataset_name, 'action_configs.json')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        action_configs = {}
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                try:
                    action_configs = json.load(f)
                except json.JSONDecodeError:
                    pass
        
        connected_model = "None"
        if hasattr(self, 'connected_model_input'):
            connected_model = self.connected_model_input.currentText().strip()
            
        if connected_model and connected_model != "None":
            action_configs[self.action_name] = connected_model
        else:
            action_configs.pop(self.action_name, None)
            
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(action_configs, f, indent=4, ensure_ascii=False)
            
        # --- NEW: Save termination config ---
        term_config_path = os.path.join('model', dataset_name, 'termination_actions.json')
        termination_actions = []
        if os.path.exists(term_config_path):
            with open(term_config_path, 'r', encoding='utf-8') as f:
                try:
                    termination_actions = json.load(f)
                except json.JSONDecodeError:
                    pass
        
        is_termination = False
        if hasattr(self, 'termination_input'):
            is_termination = (self.termination_input.currentText() == "Yes")
            
        if is_termination:
            if self.action_name not in termination_actions:
                termination_actions.append(self.action_name)
        else:
            if self.action_name in termination_actions:
                termination_actions.remove(self.action_name)
                
        with open(term_config_path, 'w', encoding='utf-8') as f:
            json.dump(termination_actions, f, indent=4, ensure_ascii=False)
        # --- END NEW ---
            
        QMessageBox.information(self, "Success", f"Connection saved:\nAction '{self.action_name}' -> Model '{connected_model}'\nTerminator: {is_termination}")

    def start_session(self):
        dataset_name = self.dataset_name_input.currentText().strip()
        if not dataset_name:
            dataset_name = "Default"
            
        self.action_name = self.action_name_input.currentText().strip()
        self.num_videos = self.num_videos_input.value()
        
        if not self.action_name:
            QMessageBox.warning(self, "Error", "Please enter an action name.")
            return
            
        self.action_video_dir = os.path.join(DATA_PATH, dataset_name, self.action_name)
        self.action_landmark_dir = os.path.join(OUTPUT_PATH, dataset_name, self.action_name)
        os.makedirs(self.action_video_dir, exist_ok=True)
        os.makedirs(self.action_landmark_dir, exist_ok=True)
        
        # --- NEW: Save connected model config ---
        import json
        config_path = os.path.join('model', dataset_name, 'action_configs.json')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        action_configs = {}
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                try:
                    action_configs = json.load(f)
                except json.JSONDecodeError:
                    pass
        
        connected_model = "None"
        if hasattr(self, 'connected_model_input'):
            connected_model = self.connected_model_input.currentText().strip()
            
        if connected_model and connected_model != "None":
            action_configs[self.action_name] = connected_model
        else:
            action_configs.pop(self.action_name, None)
            
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(action_configs, f, indent=4, ensure_ascii=False)
            
        # --- NEW: Save termination config ---
        term_config_path = os.path.join('model', dataset_name, 'termination_actions.json')
        termination_actions = []
        if os.path.exists(term_config_path):
            with open(term_config_path, 'r', encoding='utf-8') as f:
                try:
                    termination_actions = json.load(f)
                except json.JSONDecodeError:
                    pass
        
        is_termination = False
        if hasattr(self, 'termination_input'):
            is_termination = (self.termination_input.currentText() == "Yes")
            
        if is_termination:
            if self.action_name not in termination_actions:
                termination_actions.append(self.action_name)
        else:
            if self.action_name in termination_actions:
                termination_actions.remove(self.action_name)
                
        with open(term_config_path, 'w', encoding='utf-8') as f:
            json.dump(termination_actions, f, indent=4, ensure_ascii=False)
        # --- END NEW ---
        
        # Use the exact value from the UI, which automatically generates the correct sequence number
        self.start_num = self.start_video_num_input.value()
        print(f"Starting video number: {self.start_num}")
        self.current_video_num = self.start_num
        
        self.session_action_label.setText(f"Action: {self.action_name}")
        self.session_progress_bar.setRange(0, self.num_videos)
        self.session_progress_bar.setValue(0)
        self.stop_session_button.setDisabled(False)
        self.action_list.setDisabled(True)
        self.action_search.setDisabled(True)
        self.train_model_button.setDisabled(True) # Disable training during collection

        # Use stored OS index from dropdown userData (avoids virtual cam confusion)
        cam_index = self.cam_select.currentData()
        if cam_index is None:
            cam_index = self.cam_select.currentIndex()
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
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
            self.start_batch_button.setText("▶  START BATCH  (S)")
            self.start_batch_button.setVisible(True)
            self.status_text_label.setText(f"⬤  Batch {batch_num} — Ready. Press S to start.")
            
        elif new_state == STATE_BATCH_COUNTDOWN:
            self.status_text_label.setText("⬤  Get ready...")
            self.center_text_label.setObjectName("LabelCountdown")
            self.start_countdown(5, STATE_PAUSE_COUNTDOWN)
            
        elif new_state == STATE_PAUSE_COUNTDOWN:
            self.status_text_label.setText(f"⬤  Starting video {self.current_video_num}...")
            self.center_text_label.setObjectName("LabelCountdown")
            self.start_countdown(2, STATE_RECORDING)
            
        elif new_state == STATE_RECORDING:
            self.recording_label.setVisible(True)
            self.status_text_label.setText(f"🔴  Recording video {self.current_video_num}...")
            self.record_start_time = time.time()
            self.start_professional_recording()

        elif new_state == STATE_PROCESSING:
            self.center_text_label.setText("Processing...")
            self.center_text_label.setObjectName("LabelProcessing")
            self.center_text_label.setVisible(True)
            self.status_text_label.setText("⚙  Processing landmarks...")
            self.start_processing()
            
        elif new_state == STATE_SESSION_DONE:
            self.status_text_label.setText("✅  Session Complete!")
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
            remaining = max(0, self.current_record_seconds - elapsed)
            cv2.putText(image, f"Recording: {remaining:.1f}s left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
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
        qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_format)
        # Scale to 85% of the label — large but not edge-to-edge
        label_w = self.video_feed_label.width()
        label_h = self.video_feed_label.height()
        if label_w > 0 and label_h > 0:
            target_w = int(label_w * 0.85)
            target_h = int(label_h * 0.85)
            pixmap = pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pixmap

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
        if key == Qt.Key_F5:
            self.reload_stylesheets()
            
        if key == Qt.Key_Backspace:
            self.sentence_backspace()
            
        if key == Qt.Key_Delete:
            self.sentence_clear()
            
        if key == Qt.Key_Up:
            if self.rec_time_spin.isEnabled():
                self.rec_time_spin.stepUp()
        elif key == Qt.Key_Down:
            if self.rec_time_spin.isEnabled():
                self.rec_time_spin.stepDown()
                
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
            dataset_path = os.path.join(DATA_PATH, self.active_dataset)
            if not os.path.exists(dataset_path):
                self.data_stats_label.setText(f"No data collected yet for dataset: {self.active_dataset}.")
                self.manage_actions_list.clear()
                return
            
            actions = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d))]
            
            total_videos = 0
            total_processed = 0
            action_data = []
            
            for action in actions:
                action_video_dir = os.path.join(dataset_path, action)
                action_processed_dir = os.path.join(OUTPUT_PATH, self.active_dataset, action)
                
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
        
        action_video_dir = os.path.join(DATA_PATH, self.active_dataset, action_name)
        action_processed_dir = os.path.join(OUTPUT_PATH, self.active_dataset, action_name)
        
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
            action_video_dir = os.path.join(DATA_PATH, self.active_dataset, self.selected_action)
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
                action_video_dir = os.path.join(DATA_PATH, self.active_dataset, self.selected_action)
                if os.path.exists(action_video_dir):
                    shutil.rmtree(action_video_dir)
                
                # Delete processed folder
                action_processed_dir = os.path.join(OUTPUT_PATH, self.active_dataset, self.selected_action)
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
            dataset_path = os.path.join(DATA_PATH, self.active_dataset)
            if not os.path.exists(dataset_path):
                QMessageBox.warning(self, "No Data", f"No data to export for dataset {self.active_dataset}.")
                return
            
            import datetime
            
            actions = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d))]
            
            export_text = f"""
ISL Dataset Information - {self.active_dataset}
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

Total Actions: {len(actions)}

{'='*50}
ACTION DETAILS:
{'='*50}

"""
            
            for action in sorted(actions):
                action_video_dir = os.path.join(dataset_path, action)
                action_processed_dir = os.path.join(OUTPUT_PATH, self.active_dataset, action)
                
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

    def show_about_us(self):
        QMessageBox.information(
            self, "About Us",
            "Project ANWAYA\n\n"
            "This application is a Personalised Customisable Sign Language to Text Converter. "
            "Developed as a final year college project (B.Tech). Our goal is to bridge the "
            "communication gap for the deaf and mute community by converting gestures to text in real-time."
        )

    def show_how_to_use(self):
        QMessageBox.information(
            self, "How to Use",
            "1. Setup Collection: Create a dataset and record custom sign language gestures.\n"
            "2. Train Model: Navigate to the training page and train the system on your dataset.\n"
            "3. Real-Time Recognition: Start the camera on the recognition page to translate gestures into text instantly.\n"
            "4. Manage Data: View, edit, and delete recorded gestures from your dataset.\n\n"
            "Need more help? Check out our complete documentation."
        )

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