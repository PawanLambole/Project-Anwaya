from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QLineEdit, QSpinBox, QListWidget, QFrame, QProgressBar, 
    QSizePolicy, QComboBox, QGridLayout, QSpacerItem, QTextEdit,
    QCompleter, QScrollArea, QListView
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QTimer, QStringListModel, pyqtSignal


def _ui_font(point_size: int, weight: int = QFont.Normal) -> QFont:
    font = QFont()
    font.setPointSize(point_size)
    font.setWeight(weight)
    return font

_COMBO_VIEW_STYLE = """
QListView {
    background-color: #FFFFFF;
    color: #111827;
    border: 1px solid #D1D5DB;
    border-radius: 4px;
    outline: none;
}
QListView::item {
    min-height: 36px;
    padding: 4px 10px;
}
QListView::item:hover {
    background-color: #FFF7ED;
    color: #EA580C;
}
QListView::item:selected {
    background-color: #EA580C;
    color: #FFFFFF;
}
"""

def _make_combo(items=None, editable=False) -> QComboBox:
    """Create a QComboBox with an explicitly-styled QListView popup.
    This forces Qt to use its own rendering engine instead of Windows
    native controls, preventing the black-background issue."""
    cb = QComboBox()
    view = QListView()
    view.setStyleSheet(_COMBO_VIEW_STYLE)
    cb.setView(view)
    if editable:
        cb.setEditable(True)
    if items:
        cb.addItems(items)
    return cb


class ActionComboBox(QComboBox):
    """
    An editable QComboBox for Marathi action names.
    The built-in QCompleter fires on every keystroke, but with IME/transliteration
    the actual Marathi character arrives only after the IME commits. This subclass
    overrides inputMethodEvent so that after IME commits text, a short QTimer fires
    and shows the matching suggestions in the dropdown popup.
    """
    # Emitted when the user finishes entering / selecting an action name
    action_confirmed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        view = QListView()
        view.setStyleSheet(_COMBO_VIEW_STYLE)
        self.setView(view)
        self.setInsertPolicy(QComboBox.NoInsert)
        # Our completer
        self._completer = QCompleter(self)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchContains)
        self._completer.setCompletionMode(QCompleter.PopupCompletion)
        self.setCompleter(self._completer)
        # When completer selects an item, emit action_confirmed
        self._completer.activated.connect(self._on_action_confirmed)
        # Make the popup readable — large font + comfortable row height
        popup = self._completer.popup()
        popup.setFont(_ui_font(16))
        popup.setStyleSheet("""
            QListView {
                font-size: 18px;
                padding: 4px;
            }
            QListView::item {
                padding: 10px 14px;
                min-height: 44px;
            }
            QListView::item:selected {
                background-color: #EA580C;
                color: white;
            }
        """)
        # Timer to show popup after IME commits
        self._ime_timer = QTimer(self)
        self._ime_timer.setSingleShot(True)
        self._ime_timer.setInterval(150)   # 150 ms — enough for IME to finish
        self._ime_timer.timeout.connect(self._show_suggestions)
        # Also hook textEdited (for normal keyboard) to show suggestions
        self.lineEdit().textEdited.connect(self._on_text_edited)
        # Emit action_confirmed when user presses Enter or clicks dropdown item
        self.lineEdit().editingFinished.connect(lambda: self._on_action_confirmed(self.lineEdit().text()))
        self.activated[str].connect(self._on_action_confirmed)

    def set_action_list(self, actions):
        """Populate both the dropdown items and the completer model."""
        current = self.lineEdit().text()
        self.blockSignals(True)
        self.clear()
        self.addItems(actions)
        self.setCurrentIndex(-1)
        self.lineEdit().setText(current)
        self.blockSignals(False)
        self._completer.setModel(QStringListModel(actions, self._completer))

    def inputMethodEvent(self, event):
        """Called by Qt when the IME commits/updates text (e.g. Marathi transliteration)."""
        super().inputMethodEvent(event)
        # After the base class processes the IME event, schedule a popup
        self._ime_timer.start()

    def _on_text_edited(self, text):
        """Called on normal (non-IME) edits — show suggestions immediately."""
        self._ime_timer.stop()
        self._show_suggestions()

    def _on_action_confirmed(self, text):
        """Emit action_confirmed whenever the user commits an action name."""
        action = text.strip() if text else self.lineEdit().text().strip()
        if action:
            self.action_confirmed.emit(action)

    def _show_suggestions(self):
        """Force the completer to update and show the popup."""
        text = self.lineEdit().text().strip()
        if text:
            self._completer.setCompletionPrefix(text)
            if self._completer.completionCount() > 0:
                self._completer.complete()

# Lazy-import so the app still starts if camera_utils has an error
try:
    from camera_utils import enumerate_cameras, get_default_camera_index
except Exception:
    enumerate_cameras = None
    get_default_camera_index = None

class CustomTitleBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setObjectName("TopMenuBar")
        self.setFixedHeight(50)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 0, 10, 0)
        
        # Left side: Navigation buttons
        self.nav_layout = QHBoxLayout()
        self.nav_layout.setSpacing(10)
        
        self.parent.sidebar_toggle_btn = QPushButton("☰")
        self.parent.sidebar_toggle_btn.setObjectName("SidebarToggleBtn")
        self.parent.sidebar_toggle_btn.setFixedSize(40, 40)
        self.parent.sidebar_toggle_btn.setToolTip("Toggle Sidebar")
        # In CustomTitleBar __init__, toggle_sidebar might not be fully linked yet,
        # so we will connect it later or check if it exists:
        if hasattr(self.parent, 'toggle_sidebar'):
            self.parent.sidebar_toggle_btn.clicked.connect(self.parent.toggle_sidebar)
            
        self.nav_layout.addWidget(self.parent.sidebar_toggle_btn)
        
        self.parent.home_menu_btn = QPushButton("Home")
        self.parent.home_menu_btn.setObjectName("TopMenuTitleBtn")
        self.parent.home_menu_btn.clicked.connect(self.parent.go_to_home)
        self.nav_layout.addWidget(self.parent.home_menu_btn)
        
        self.parent.about_us_btn = QPushButton("About Us")
        self.parent.about_us_btn.setObjectName("MenuBarButton")
        self.parent.about_us_btn.clicked.connect(self.parent.show_about_us)
        self.nav_layout.addWidget(self.parent.about_us_btn)
        
        self.parent.how_to_use_btn = QPushButton("How to Use")
        self.parent.how_to_use_btn.setObjectName("MenuBarButton")
        self.parent.how_to_use_btn.clicked.connect(self.parent.show_how_to_use)
        self.nav_layout.addWidget(self.parent.how_to_use_btn)
        
        self.layout.addLayout(self.nav_layout)
        self.layout.addStretch(1)
        
        # Center: Project Name
        self.title_label = QLabel("Anvaya")
        self.title_label.setObjectName("TitleBarProjectName")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #EA580C; font-size: 20px; font-weight: bold;")
        self.layout.addWidget(self.title_label)
        
        self.layout.addStretch(1)
        
        # Right side: Window controls
        self.controls_layout = QHBoxLayout()
        self.controls_layout.setSpacing(5)
        
        self.minimize_btn = QPushButton("\uE921")
        self.minimize_btn.setObjectName("TitleBarControlBtn")
        self.minimize_btn.setFixedSize(30, 30)
        self.minimize_btn.clicked.connect(self.parent.showMinimized)
        self.controls_layout.addWidget(self.minimize_btn)
        
        self.maximize_btn = QPushButton("\uE922")
        self.maximize_btn.setObjectName("TitleBarControlBtn")
        self.maximize_btn.setFixedSize(30, 30)
        self.maximize_btn.clicked.connect(self.toggle_maximize)
        self.controls_layout.addWidget(self.maximize_btn)
        
        self.close_btn = QPushButton("\uE8BB")
        self.close_btn.setObjectName("TitleBarCloseBtn")
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.clicked.connect(self.parent.close)
        self.controls_layout.addWidget(self.close_btn)
        
        self.layout.addLayout(self.controls_layout)
        
        self.start_pos = None

    def toggle_maximize(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
            self.maximize_btn.setText("\uE922")
        else:
            self.parent.showMaximized()
            self.maximize_btn.setText("\uE923")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.globalPos() - self.parent.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.start_pos:
            self.parent.move(event.globalPos() - self.start_pos)
            event.accept()
            
    def mouseReleaseEvent(self, event):
        self.start_pos = None


def create_top_menubar(main_window):
    return CustomTitleBar(main_window)

def create_sidebar(main_window):
    """
    Creates the sidebar widget.
    'main_window' is the instance of CollectionApp
    """
    sidebar = QWidget()
    sidebar.setObjectName("Sidebar")
    sidebar.setMinimumWidth(60)
    sidebar.setMaximumWidth(600)
    sidebar_layout = QVBoxLayout(sidebar)
    sidebar_layout.setContentsMargins(10, 20, 10, 20)
    sidebar_layout.setSpacing(10)

    # --- Main Content Widget (Hidable) ---
    main_window.sidebar_content = QWidget()
    main_window.sidebar_content.setObjectName("SidebarContent")
    content_layout = QVBoxLayout(main_window.sidebar_content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(10)

    content_layout.addWidget(QLabel("ACTIONS", objectName="SidebarTitle"))
    main_window.action_search = QLineEdit()
    main_window.action_search.setPlaceholderText("Search actions...")
    main_window.action_search.textChanged.connect(main_window.filter_actions)
    content_layout.addWidget(main_window.action_search)

    main_window.action_list = QListWidget()
    main_window.action_list.setMinimumHeight(150) # Ensure it has a decent minimum size
    main_window.action_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    main_window.action_list.itemClicked.connect(main_window.on_action_clicked)
    content_layout.addWidget(main_window.action_list, 1)

    content_layout.addWidget(QLabel("ACTIVE MODEL", objectName="SidebarTitle"))
    main_window.model_select = QComboBox()
    main_window.model_select.currentIndexChanged.connect(main_window.on_model_changed)
    content_layout.addWidget(main_window.model_select)

    content_layout.addWidget(QLabel("SESSION", objectName="SidebarTitle"))
    main_window.session_action_label = QLabel("Action: N/A")
    content_layout.addWidget(main_window.session_action_label)
    
    main_window.session_progress_bar = QProgressBar()
    main_window.session_progress_bar.setValue(0)
    content_layout.addWidget(main_window.session_progress_bar)

    content_layout.addWidget(QLabel("SETTINGS", objectName="SidebarTitle"))
    cam_layout = QHBoxLayout()
    cam_layout.addWidget(QLabel("Camera:"))
    main_window.cam_select = QComboBox()

    # Populate with real OS device names + store the true OS index as item data
    main_window._camera_list = []  # [{'index': int, 'name': str, ...}]
    _default_dropdown_idx = 0

    if enumerate_cameras is not None:
        try:
            cams = enumerate_cameras(max_test=8)
        except Exception:
            cams = []
    else:
        cams = []

    if cams:
        main_window._camera_list = cams
        _best_os_idx = get_default_camera_index(cams) if get_default_camera_index else cams[0]["index"]
        for _i, _cam in enumerate(cams):
            _tag = " [virtual]" if _cam["is_virtual"] else (" [built-in]" if _cam["is_builtin"] else "")
            main_window.cam_select.addItem(f"{_cam['name']}{_tag}", userData=_cam["index"])
            if _cam["index"] == _best_os_idx:
                _default_dropdown_idx = _i
    else:
        # Fallback if enumeration fails: generic entries with index as userData
        for _i in range(3):
            main_window.cam_select.addItem(f"Camera {_i}", userData=_i)

    main_window.cam_select.setCurrentIndex(_default_dropdown_idx)
    cam_layout.addWidget(main_window.cam_select)
    content_layout.addLayout(cam_layout)

    rec_layout = QHBoxLayout()
    main_window.rec_time_label = QLabel("Rec. Time (s):")
    rec_layout.addWidget(main_window.rec_time_label)
    main_window.rec_time_spin = QSpinBox()
    main_window.rec_time_spin.setRange(1, 10)
    from mediapipe_logic import RECORD_SECONDS
    main_window.rec_time_spin.setValue(RECORD_SECONDS)
    rec_layout.addWidget(main_window.rec_time_spin)
    content_layout.addLayout(rec_layout)

    # Wrap sidebar_content in a scroll area so it never gets clipped
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(main_window.sidebar_content)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    sidebar_layout.addWidget(scroll, 1)

    # --- Buttons (always visible at the bottom, no addStretch competition) ---
    main_window.theme_toggle_btn = QPushButton("🌙 Switch to Dark Theme")
    main_window.theme_toggle_btn.setObjectName("ButtonGray")
    main_window.theme_toggle_btn.clicked.connect(main_window.toggle_theme)
    sidebar_layout.addWidget(main_window.theme_toggle_btn)


    main_window.train_model_button = QPushButton("📈 Train New Model")
    main_window.train_model_button.setObjectName("ButtonTrain")
    main_window.train_model_button.clicked.connect(main_window.go_to_training)
    sidebar_layout.addWidget(main_window.train_model_button)

    main_window.stop_session_button = QPushButton("⏹️ STOP SESSION")
    main_window.stop_session_button.setObjectName("ButtonStop")
    main_window.stop_session_button.clicked.connect(main_window.stop_session)
    main_window.stop_session_button.setDisabled(True)
    sidebar_layout.addWidget(main_window.stop_session_button)

    return sidebar

class HomeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Homepage")
        from PyQt5.QtGui import QPixmap
        import os
        bg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bg.png")
        if os.path.exists(bg_path):
            self.bg_image = QPixmap(bg_path)
        else:
            self.bg_image = None

    def paintEvent(self, event):
        # Let stylesheet properties apply first
        super().paintEvent(event)
        
        if self.bg_image and not self.bg_image.isNull():
            from PyQt5.QtGui import QPainter
            from PyQt5.QtCore import Qt
            painter = QPainter(self)
            painter.setOpacity(0.15)  # Lower opacity (15%)
            
            scaled = self.bg_image.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            painter.end()


def create_home_widget(main_window):
    """This is your Homepage UI."""
    widget = HomeWidget()
    
    layout = QVBoxLayout(widget)
    layout.setAlignment(Qt.AlignCenter)
    layout.setSpacing(10)
    
    layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    title_layout = QHBoxLayout()
    
    proj_label = QLabel("Project ")
    proj_label.setObjectName("LabelHomeTitleNormal")
    proj_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    title_layout.addWidget(proj_label)
    
    anv_label = QLabel("Anvaya")
    anv_label.setObjectName("LabelHomeTitleSamarkan")
    anv_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    title_layout.addWidget(anv_label)
    
    layout.addLayout(title_layout)

    tagline = QLabel("Personalised Customisable Sign Language to Text Converter")
    tagline.setObjectName("LabelHomeTagline")
    tagline.setAlignment(Qt.AlignCenter)
    layout.addWidget(tagline)

    button_container = QWidget()
    button_container.setObjectName("HomeButtonLayout")
    button_layout = QGridLayout(button_container)
    button_layout.setAlignment(Qt.AlignCenter)
    button_layout.setSpacing(15)
    
    main_window.start_collection_button = QPushButton("Start New Collection")
    main_window.start_collection_button.setObjectName("HomeBtnSaffron")
    main_window.start_collection_button.clicked.connect(main_window.go_to_setup)
    button_layout.addWidget(main_window.start_collection_button, 0, 0)

    main_window.recognize_btn = QPushButton("Real-Time Recognition")
    main_window.recognize_btn.setObjectName("HomeBtnGreen")
    main_window.recognize_btn.clicked.connect(main_window.go_to_recognition)
    button_layout.addWidget(main_window.recognize_btn, 0, 1)

    main_window.manage_data_button = QPushButton("Manage Data")
    main_window.manage_data_button.setObjectName("HomeBtnPurple")
    main_window.manage_data_button.clicked.connect(main_window.go_to_manage_data)
    button_layout.addWidget(main_window.manage_data_button, 1, 0)
    
    main_window.batch_processor_btn = QPushButton("Batch Processor")
    main_window.batch_processor_btn.setObjectName("HomeBtnNavy")
    main_window.batch_processor_btn.clicked.connect(main_window.go_to_batch_process)
    button_layout.addWidget(main_window.batch_processor_btn, 1, 1)
    
    layout.addWidget(button_container)
    layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
    
    # Bottom Layout for Footer and Quit Button
    bottom_layout = QHBoxLayout()
    
    # Left Spacer
    bottom_layout.addStretch(1)
    
    footer = QLabel("B.Tech Final Year Project")
    footer.setObjectName("LabelHelper")
    footer.setAlignment(Qt.AlignCenter)
    bottom_layout.addWidget(footer)
    
    # Right Stretch to push Quit button to the edge
    bottom_layout.addStretch(1)
    
    main_window.quit_home_button = QPushButton("Quit Application")
    main_window.quit_home_button.setObjectName("HomeBtnNavy")
    main_window.quit_home_button.clicked.connect(main_window.close)
    bottom_layout.addWidget(main_window.quit_home_button)
    
    layout.addLayout(bottom_layout)
    return widget

def create_setup_widget(main_window):
    widget = QWidget()
    widget.setObjectName("SetupWidget")
    
    main_layout = QVBoxLayout(widget)
    main_layout.setContentsMargins(0, 0, 0, 0)
    
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setStyleSheet("background-color: transparent;")
    
    content_widget = QWidget()
    content_widget.setObjectName("SetupContentWidget")
    
    layout = QVBoxLayout(content_widget)
    layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
    layout.setSpacing(15)
    layout.setContentsMargins(40, 30, 40, 30)

    title = QLabel("Setup New Collection Session")
    title.setFont(_ui_font(20, QFont.Bold))
    title.setAlignment(Qt.AlignCenter)
    layout.addWidget(title)
    
    main_layout.addWidget(scroll)
    scroll.setWidget(content_widget)

    lbl_dataset = QLabel("Dataset Name")
    lbl_dataset.setObjectName("SetupLabel")
    layout.addWidget(lbl_dataset)
    main_window.dataset_name_input = _make_combo(editable=True)
    main_window.dataset_name_input.setPlaceholderText("e.g., Default")
    main_window.dataset_name_input.setFont(_ui_font(16))
    layout.addWidget(main_window.dataset_name_input)
    layout.addWidget(QLabel("(Select from list or type new)", objectName="LabelHelper"))

    lbl_action = QLabel("Action Name (Marathi)")
    lbl_action.setObjectName("SetupLabel")
    layout.addWidget(lbl_action)
    main_window.action_name_input = ActionComboBox()
    main_window.action_name_input.setPlaceholderText("e.g., आभार")
    main_window.action_name_input.lineEdit().setPlaceholderText("e.g., आभार")
    main_window.action_name_input.setFont(_ui_font(16))
    layout.addWidget(main_window.action_name_input)
    layout.addWidget(QLabel("(Type action name — suggestions appear after typing)", objectName="LabelHelper"))

    # ── Horizontal Layout for Video Count & Start Index ──
    vid_layout = QHBoxLayout()
    
    # Left: Number of Videos
    vid_left = QVBoxLayout()
    lbl_num = QLabel("Number of Videos to Record")
    lbl_num.setObjectName("SetupLabel")
    vid_left.addWidget(lbl_num)
    main_window.num_videos_input = QSpinBox()
    main_window.num_videos_input.setRange(1, 1000)
    main_window.num_videos_input.setValue(50)
    main_window.num_videos_input.setFont(_ui_font(16))
    vid_left.addWidget(main_window.num_videos_input)
    vid_left.addWidget(QLabel("", objectName="LabelHelper")) # Spacer for alignment
    vid_layout.addLayout(vid_left)
    
    # Right: Start From
    vid_right = QVBoxLayout()
    lbl_start = QLabel("Start From (Video Number)")
    lbl_start.setObjectName("SetupLabel")
    vid_right.addWidget(lbl_start)
    main_window.start_video_num_input = QSpinBox()
    main_window.start_video_num_input.setRange(0, 9999)
    main_window.start_video_num_input.setValue(0)
    main_window.start_video_num_input.setFont(_ui_font(16))
    vid_right.addWidget(main_window.start_video_num_input)
    vid_right.addWidget(QLabel("(Updates automatically on action select)", objectName="LabelHelper"))
    vid_layout.addLayout(vid_right)
    
    layout.addLayout(vid_layout)

    # ── Horizontal Layout for Connection & Termination ──
    options_layout = QHBoxLayout()
    
    # Left: Connected Model
    opt_left = QVBoxLayout()
    lbl_connected_model = QLabel("Connect Action to Model (Optional)")
    lbl_connected_model.setObjectName("SetupLabel")
    opt_left.addWidget(lbl_connected_model)
    main_window.connected_model_input = _make_combo(items=["None"])
    main_window.connected_model_input.setFont(_ui_font(16))
    opt_left.addWidget(main_window.connected_model_input)
    opt_left.addWidget(QLabel("(Switches to this model when action recognized)", objectName="LabelHelper"))
    options_layout.addLayout(opt_left)
    
    # Right: Terminator
    opt_right = QVBoxLayout()
    lbl_termination = QLabel("Is this a Terminator? (Optional)")
    lbl_termination.setObjectName("SetupLabel")
    opt_right.addWidget(lbl_termination)
    main_window.termination_input = _make_combo(items=["No", "Yes"])
    main_window.termination_input.setFont(_ui_font(16))
    opt_right.addWidget(main_window.termination_input)
    opt_right.addWidget(QLabel("(If Yes, ends sentence with '.')", objectName="LabelHelper"))
    options_layout.addLayout(opt_right)
    
    layout.addLayout(options_layout)

    setup_button_layout = QHBoxLayout()
    main_window.back_button = QPushButton("Back to Home")
    main_window.back_button.setObjectName("ButtonGray")
    main_window.back_button.clicked.connect(main_window.go_to_home)
    setup_button_layout.addWidget(main_window.back_button)

    main_window.save_connection_btn = QPushButton("Save Connection Only")
    main_window.save_connection_btn.setFont(_ui_font(14, QFont.Bold))
    main_window.save_connection_btn.setMinimumHeight(50)
    main_window.save_connection_btn.setObjectName("ButtonPurple")
    main_window.save_connection_btn.clicked.connect(main_window.save_connection_only)
    setup_button_layout.addWidget(main_window.save_connection_btn, 1)

    main_window.start_session_button = QPushButton("START SESSION")
    main_window.start_session_button.setFont(_ui_font(18, QFont.Bold))
    main_window.start_session_button.setMinimumHeight(50)
    main_window.start_session_button.clicked.connect(main_window.start_session)
    setup_button_layout.addWidget(main_window.start_session_button, 1)
    
    layout.addLayout(setup_button_layout)
    return widget

def create_recognition_widget(main_window):
    """
    Creates the Recognition (Real-time Translation) Page UI.
    Fully responsive — video fills all available space via stretch factor.
    """
    widget = QWidget()
    widget.setObjectName("RecognitionPage")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(16, 10, 16, 10)
    layout.setSpacing(6)

    # ── Header ──────────────────────────────────────────────────────────────
    header_layout = QHBoxLayout()
    main_window.recognition_back_btn = QPushButton("← Back to Home")
    main_window.recognition_back_btn.setObjectName("ButtonGray")
    header_layout.addWidget(main_window.recognition_back_btn)

    header_title = QLabel("Real-Time ISL Recognition")
    header_title.setObjectName("RecognitionHeaderTitle")
    header_title.setAlignment(Qt.AlignCenter)
    header_layout.addWidget(header_title, 1)

    # Invisible spacer to balance the back button on the right
    spacer_lbl = QLabel()
    spacer_lbl.setFixedWidth(main_window.recognition_back_btn.sizeHint().width())
    header_layout.addWidget(spacer_lbl)

    layout.addLayout(header_layout)

    # ── Main Video and Completed Sentences Layout ───────────────────────────
    video_and_sentences_layout = QHBoxLayout()
    video_and_sentences_layout.setSpacing(10)

    # ── Video (fills all remaining space) ───────────────────────────────────
    main_window.recognition_video_label = QLabel()
    main_window.recognition_video_label.setObjectName("videoLabel")
    main_window.recognition_video_label.setAlignment(Qt.AlignCenter)
    # QSizePolicy.Ignored: the pixmap size never feeds back into the layout,
    # preventing the infinite-zoom loop when the sidebar is toggled.
    main_window.recognition_video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    main_window.recognition_video_label.setStyleSheet("QLabel#videoLabel { background-color: transparent; }")
    main_window.recognition_video_label.setText("Camera Stopped")
    video_and_sentences_layout.addWidget(main_window.recognition_video_label, 4)   # stretch=4 gives video more space

    # ── Completed Sentences Panel ───────────────────────────────────────────
    main_window.completed_sentences_text = QTextEdit()
    main_window.completed_sentences_text.setObjectName("CompletedSentences")
    main_window.completed_sentences_text.setReadOnly(True)
    main_window.completed_sentences_text.setStyleSheet("""
        QTextEdit#CompletedSentences {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 10px;
            font-size: 18px;
            color: #1E293B;
        }
    """)
    main_window.completed_sentences_text.setPlaceholderText("Completed sentences will appear here...")
    video_and_sentences_layout.addWidget(main_window.completed_sentences_text, 1) # stretch=1 for text area

    layout.addLayout(video_and_sentences_layout, 1) # The QHBoxLayout takes all spare height

    # ── Prediction banner (compact, fixed height) ───────────────────────────
    prediction_banner = QWidget()
    prediction_banner.setObjectName("predictionBanner")
    prediction_banner.setFixedHeight(40)
    prediction_banner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    prediction_layout = QHBoxLayout(prediction_banner)
    prediction_layout.setContentsMargins(14, 0, 14, 0)
    prediction_layout.setSpacing(8)

    main_window.recognition_prediction_label = QLabel("Ready to recognize...")
    main_window.recognition_prediction_label.setObjectName("predictionLabel")
    main_window.recognition_prediction_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    main_window.recognition_confidence_label = QLabel("")
    main_window.recognition_confidence_label.setObjectName("confidenceLabel")
    main_window.recognition_confidence_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    prediction_layout.addWidget(main_window.recognition_prediction_label, 1)
    prediction_layout.addWidget(main_window.recognition_confidence_label)
    layout.addWidget(prediction_banner)

    # ── Sentence panel (compact, fixed height) ──────────────────────────────
    sentence_panel = QWidget()
    sentence_panel.setObjectName("SentencePanel")
    sentence_panel.setFixedHeight(56)
    sentence_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    sentence_panel.setStyleSheet("""
        QWidget#SentencePanel {
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            background-color: transparent;
        }
    """)
    sentence_panel_layout = QHBoxLayout(sentence_panel)
    sentence_panel_layout.setContentsMargins(12, 4, 8, 4)
    sentence_panel_layout.setSpacing(6)

    sentence_icon = QLabel("📝")
    sentence_icon.setFixedWidth(26)
    sentence_panel_layout.addWidget(sentence_icon)

    main_window.sentence_label = QLabel("")
    main_window.sentence_label.setObjectName("SentenceLabel")
    main_window.sentence_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    main_window.sentence_label.setWordWrap(False)
    main_window.sentence_label.setStyleSheet("font-size: 17px; font-weight: bold; color: #1E293B;")
    main_window.sentence_label.setText("(sentence will appear here)")
    sentence_panel_layout.addWidget(main_window.sentence_label, 1)

    main_window.sentence_backspace_btn = QPushButton("⌫")
    main_window.sentence_backspace_btn.setObjectName("SentenceBackspaceBtn")
    main_window.sentence_backspace_btn.setFixedSize(38, 38)
    main_window.sentence_backspace_btn.setToolTip("Remove last word")
    main_window.sentence_backspace_btn.setStyleSheet("""
        QPushButton#SentenceBackspaceBtn {
            background-color: #F1F5F9;
            color: #475569;
            font-size: 16px;
            border-radius: 6px;
            border: 1px solid #CBD5E1;
        }
        QPushButton#SentenceBackspaceBtn:hover { background-color: #E2E8F0; }
    """)
    sentence_panel_layout.addWidget(main_window.sentence_backspace_btn)

    main_window.sentence_clear_btn = QPushButton("✕ Clear")
    main_window.sentence_clear_btn.setObjectName("SentenceClearBtn")
    main_window.sentence_clear_btn.setFixedHeight(38)
    main_window.sentence_clear_btn.setStyleSheet("""
        QPushButton#SentenceClearBtn {
            background-color: #FEE2E2;
            color: #DC2626;
            font-size: 13px;
            font-weight: bold;
            border-radius: 6px;
            border: 1px solid #FECACA;
            padding: 0 12px;
        }
        QPushButton#SentenceClearBtn:hover { background-color: #FECACA; }
    """)
    sentence_panel_layout.addWidget(main_window.sentence_clear_btn)
    layout.addWidget(sentence_panel)

    # ── Camera control buttons ───────────────────────────────────────────────
    control_layout = QHBoxLayout()
    control_layout.addStretch()

    main_window.recognition_start_btn = QPushButton("Start Camera")
    main_window.recognition_start_btn.setObjectName("startRecognitionButton")
    main_window.recognition_start_btn.setFixedHeight(40)
    main_window.recognition_start_btn.setMinimumWidth(150)
    control_layout.addWidget(main_window.recognition_start_btn)

    main_window.recognition_stop_btn = QPushButton("Stop Camera")
    main_window.recognition_stop_btn.setObjectName("stopRecognitionButton")
    main_window.recognition_stop_btn.setFixedHeight(40)
    main_window.recognition_stop_btn.setMinimumWidth(150)
    main_window.recognition_stop_btn.setEnabled(False)
    control_layout.addWidget(main_window.recognition_stop_btn)

    control_layout.addStretch()
    layout.addLayout(control_layout)

    # ── Status row ───────────────────────────────────────────────────────────
    status_layout = QHBoxLayout()
    status_layout.setContentsMargins(0, 0, 0, 0)

    main_window.recognition_status_indicator = QLabel("●")
    main_window.recognition_status_indicator.setObjectName("statusIndicator")
    main_window.recognition_status_indicator.setStyleSheet("color: #666; font-size: 14px;")

    main_window.recognition_status_text = QLabel("Camera stopped")
    main_window.recognition_status_text.setObjectName("statusText")
    main_window.recognition_status_text.setStyleSheet("color: #666; font-size: 13px;")

    status_layout.addStretch()
    status_layout.addWidget(main_window.recognition_status_indicator)
    status_layout.addWidget(main_window.recognition_status_text)
    status_layout.addStretch()

    layout.addLayout(status_layout)

    return widget


def create_manage_data_widget(main_window):
    """
    Creates the Data Management Page UI
    """
    widget = QWidget()
    widget.setObjectName("ManageDataPage")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(40, 40, 40, 40)
    layout.setSpacing(25)

    # Header with back button
    header_layout = QHBoxLayout()
    main_window.manage_data_back_btn = QPushButton("← Back to Home")
    main_window.manage_data_back_btn.setObjectName("ButtonGray")
    header_layout.addWidget(main_window.manage_data_back_btn)
    
    header_title = QLabel("Data Management")
    header_title.setObjectName("ManageDataHeaderTitle")
    header_title.setAlignment(Qt.AlignCenter)
    header_layout.addWidget(header_title, 1)
    
    header_layout.addWidget(QLabel())
    header_layout.itemAt(2).widget().setFixedWidth(main_window.manage_data_back_btn.sizeHint().width())
    
    layout.addLayout(header_layout)

    # Data statistics container
    stats_container = QWidget()
    stats_container.setObjectName("statsContainer")
    stats_layout = QVBoxLayout(stats_container)
    
    main_window.data_stats_label = QLabel("Loading statistics...")
    main_window.data_stats_label.setObjectName("dataStatsLabel")
    stats_layout.addWidget(main_window.data_stats_label)
    
    layout.addWidget(stats_container)

    # Action list with details
    actions_header_layout = QHBoxLayout()
    actions_label = QLabel("Actions in Dataset:")
    actions_label.setObjectName("SectionLabel")
    actions_header_layout.addWidget(actions_label)
    
    actions_header_layout.addStretch()
    
    main_window.refresh_actions_btn = QPushButton("🔄 Refresh")
    main_window.refresh_actions_btn.setObjectName("refreshActionsButton")
    main_window.refresh_actions_btn.setMaximumWidth(120)
    actions_header_layout.addWidget(main_window.refresh_actions_btn)
    
    layout.addLayout(actions_header_layout)
    
    main_window.manage_actions_list = QListWidget()
    main_window.manage_actions_list.setObjectName("manageActionsList")
    layout.addWidget(main_window.manage_actions_list, 1)

    # Action details panel
    details_container = QWidget()
    details_container.setObjectName("detailsContainer")
    details_layout = QVBoxLayout(details_container)
    
    main_window.action_details_label = QLabel("Select an action to view details")
    main_window.action_details_label.setObjectName("actionDetailsLabel")
    main_window.action_details_label.setWordWrap(True)
    details_layout.addWidget(main_window.action_details_label)
    
    layout.addWidget(details_container)

    # Action buttons
    button_layout = QHBoxLayout()
    
    main_window.view_videos_btn = QPushButton("📹 View Videos")
    main_window.view_videos_btn.setObjectName("viewVideosButton")
    main_window.view_videos_btn.setEnabled(False)
    button_layout.addWidget(main_window.view_videos_btn)
    
    main_window.delete_action_btn = QPushButton("🗑️ Delete Action")
    main_window.delete_action_btn.setObjectName("deleteActionButton")
    main_window.delete_action_btn.setEnabled(False)
    button_layout.addWidget(main_window.delete_action_btn)
    
    main_window.export_data_btn = QPushButton("💾 Export Data Info")
    main_window.export_data_btn.setObjectName("exportDataButton")
    button_layout.addWidget(main_window.export_data_btn)
    
    button_layout.addStretch()
    
    main_window.restart_app_btn = QPushButton("🔄 Restart Application")
    main_window.restart_app_btn.setObjectName("restartAppButton")
    main_window.restart_app_btn.setMinimumWidth(180)
    button_layout.addWidget(main_window.restart_app_btn)
    
    layout.addLayout(button_layout)
    
    return widget

def create_collection_widget(main_window):
    widget = QWidget()
    layout = QGridLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setRowStretch(0, 1)
    layout.setColumnStretch(0, 1)
    
    main_window.video_feed_label = QLabel()
    main_window.video_feed_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
    main_window.video_feed_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    main_window.video_feed_label.setStyleSheet("background-color: transparent;")
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
    status_bar_widget.setFixedHeight(72)
    status_bar_widget.setObjectName("CollectionStatusBar")
    status_layout = QHBoxLayout(status_bar_widget)
    status_layout.setContentsMargins(20, 8, 20, 8)
    status_layout.setSpacing(12)

    main_window.status_text_label = QLabel("⬤  Ready for batch")
    main_window.status_text_label.setObjectName("CollectionStatusLabel")
    status_layout.addWidget(main_window.status_text_label, 1)

    main_window.start_batch_button = QPushButton("▶  START BATCH  (S)")
    main_window.start_batch_button.setObjectName("CollectionStartBtn")
    main_window.start_batch_button.setMinimumHeight(46)
    main_window.start_batch_button.clicked.connect(main_window.start_batch_countdown)
    status_layout.addWidget(main_window.start_batch_button)

    main_window.quit_button = QPushButton("✕  STOP  (Q)")
    main_window.quit_button.setObjectName("CollectionStopBtn")
    main_window.quit_button.setMinimumHeight(46)
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
    layout.setContentsMargins(40, 40, 40, 40)
    layout.setSpacing(15)

    title = QLabel("Model Training")
    title.setObjectName("TrainingTitle")
    title.setAlignment(Qt.AlignCenter)
    layout.addWidget(title)
    
    # Dataset selection for training
    setup_layout = QHBoxLayout()
    setup_layout.addWidget(QLabel("Select Dataset to Train:"))
    main_window.training_dataset_dropdown = QComboBox()
    setup_layout.addWidget(main_window.training_dataset_dropdown, 1)
    
    main_window.start_training_btn = QPushButton("Start Training")
    main_window.start_training_btn.setObjectName("ButtonGreen")
    main_window.start_training_btn.clicked.connect(main_window.start_training_process)
    setup_layout.addWidget(main_window.start_training_btn)
    
    # --- NEW: Stop Training Button ---
    main_window.stop_training_btn = QPushButton("Stop Training")
    main_window.stop_training_btn.setObjectName("ButtonStop")
    main_window.stop_training_btn.clicked.connect(main_window.stop_training_process)
    main_window.stop_training_btn.setDisabled(True)
    setup_layout.addWidget(main_window.stop_training_btn)
    
    layout.addLayout(setup_layout)

    # --- NEW: Progress Bars Layout ---
    progress_layout = QVBoxLayout()
    
    # Dataset Loading Progress
    loading_layout = QHBoxLayout()
    loading_layout.addWidget(QLabel("Dataset Loading:"))
    main_window.loading_progress_bar = QProgressBar()
    main_window.loading_progress_bar.setValue(0)
    loading_layout.addWidget(main_window.loading_progress_bar, 1)
    progress_layout.addLayout(loading_layout)

    # Training Epoch Progress
    epoch_layout = QHBoxLayout()
    epoch_layout.addWidget(QLabel("Model Training (Epochs):"))
    main_window.training_progress_bar = QProgressBar()
    main_window.training_progress_bar.setValue(0)
    epoch_layout.addWidget(main_window.training_progress_bar, 1)
    progress_layout.addLayout(epoch_layout)

    layout.addLayout(progress_layout)

    content_layout = QHBoxLayout()
    
    # Log display
    main_window.training_log_display = QTextEdit()
    main_window.training_log_display.setObjectName("TrainingLog")
    main_window.training_log_display.setReadOnly(True)
    content_layout.addWidget(main_window.training_log_display, 1) # Stretch

    # --- NEW: Chart Display ---
    main_window.training_chart_label = QLabel("Chart will appear here after training.")
    main_window.training_chart_label.setAlignment(Qt.AlignCenter)
    main_window.training_chart_label.setMinimumSize(400, 300)
    main_window.training_chart_label.setStyleSheet("background-color: #f0f0f0; color: #333; border: 1px solid #ccc;")
    content_layout.addWidget(main_window.training_chart_label, 1)

    layout.addLayout(content_layout, 1)

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

def create_batch_process_widget(main_window):
    """
    Creates the Batch Data Processing Page UI
    """
    widget = QWidget()
    widget.setObjectName("BatchProcessPage")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(40, 40, 40, 40)
    layout.setSpacing(25)

    # Header
    header_layout = QHBoxLayout()
    main_window.batch_back_btn = QPushButton("← Back to Home")
    main_window.batch_back_btn.setObjectName("ButtonGray")
    header_layout.addWidget(main_window.batch_back_btn)

    header_title = QLabel("Parallel Batch Processor")
    header_title.setObjectName("BatchHeaderTitle")
    header_title.setStyleSheet("font-size: 26px; font-weight: bold; color: #EA580C;")
    header_title.setAlignment(Qt.AlignCenter)
    header_layout.addWidget(header_title, 1)

    spacer_lbl = QLabel()
    spacer_lbl.setFixedWidth(main_window.batch_back_btn.sizeHint().width())
    header_layout.addWidget(spacer_lbl)
    layout.addLayout(header_layout)

    # Settings Row
    settings_layout = QHBoxLayout()
    
    dataset_lbl = QLabel("Dataset:")
    dataset_lbl.setStyleSheet("font-size: 16px; font-weight: bold;")
    settings_layout.addWidget(dataset_lbl)
    
    main_window.batch_dataset_input = _make_combo(editable=True)
    main_window.batch_dataset_input.setFont(_ui_font(14))
    main_window.batch_dataset_input.setMinimumWidth(200)
    settings_layout.addWidget(main_window.batch_dataset_input)
    
    main_window.batch_scan_btn = QPushButton("Scan ISL_Data Folders")
    main_window.batch_scan_btn.setObjectName("ButtonBlue")
    settings_layout.addWidget(main_window.batch_scan_btn)
    settings_layout.addStretch()
    
    layout.addLayout(settings_layout)

    # Actions Lists (Dual List Layout)
    lists_layout = QHBoxLayout()
    
    # Left: Available
    left_layout = QVBoxLayout()
    list_lbl = QLabel("Available Action Folders:")
    list_lbl.setStyleSheet("font-size: 16px; font-weight: bold;")
    left_layout.addWidget(list_lbl)
    
    main_window.batch_folder_list = QListWidget()
    main_window.batch_folder_list.setSelectionMode(QListWidget.MultiSelection)
    main_window.batch_folder_list.setMinimumHeight(100)
    main_window.batch_folder_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    list_style = """
        QListWidget {
            background-color: #FFFFFF;
            color: #1E293B;
            border: 1px solid #CBD5E1;
            border-radius: 6px;
            font-size: 16px;
        }
        QListWidget::item { padding: 10px; border-bottom: 1px solid #F1F5F9; }
        QListWidget::item:selected { background-color: #EA580C; color: white; }
    """
    main_window.batch_folder_list.setStyleSheet(list_style)
    left_layout.addWidget(main_window.batch_folder_list, 1)
    lists_layout.addLayout(left_layout, 5)
    
    # Middle: Buttons
    btn_layout = QVBoxLayout()
    btn_layout.addStretch()
    main_window.batch_add_btn = QPushButton("Add >>")
    main_window.batch_add_btn.setObjectName("ButtonBlue")
    btn_layout.addWidget(main_window.batch_add_btn)
    
    main_window.batch_remove_btn = QPushButton("<< Remove")
    main_window.batch_remove_btn.setObjectName("ButtonGray")
    btn_layout.addWidget(main_window.batch_remove_btn)
    btn_layout.addStretch()
    lists_layout.addLayout(btn_layout, 1)
    
    # Right: Selected
    right_layout = QVBoxLayout()
    sel_lbl = QLabel("Selected for Processing:")
    sel_lbl.setStyleSheet("font-size: 16px; font-weight: bold;")
    right_layout.addWidget(sel_lbl)
    
    main_window.batch_selected_list = QListWidget()
    main_window.batch_selected_list.setSelectionMode(QListWidget.MultiSelection)
    main_window.batch_selected_list.setMinimumHeight(100)
    main_window.batch_selected_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    main_window.batch_selected_list.setStyleSheet(list_style)
    right_layout.addWidget(main_window.batch_selected_list, 1)
    lists_layout.addLayout(right_layout, 5)
    
    layout.addLayout(lists_layout, 3) # Big stretch factor so the lists take most of the screen

    # Progress Tracking Area
    prog_lbl = QLabel("Task Details (Live):")
    prog_lbl.setStyleSheet("font-size: 16px; font-weight: bold;")
    layout.addWidget(prog_lbl)
    
    main_window.batch_progress_area_widget = QWidget()
    main_window.batch_progress_area = QVBoxLayout(main_window.batch_progress_area_widget)
    main_window.batch_progress_area.setContentsMargins(10, 10, 10, 10)
    
    placeholder = QLabel("Task progress will appear here...")
    placeholder.setStyleSheet("color: #64748B; font-style: italic;")
    main_window.batch_progress_area.addWidget(placeholder)
    main_window.batch_progress_area.addStretch()
    
    # We will use a QScrollArea in case there are many concurrent bars
    progress_scroll = QScrollArea()
    progress_scroll.setWidgetResizable(True)
    progress_scroll.setWidget(main_window.batch_progress_area_widget)
    progress_scroll.setMinimumHeight(60)
    progress_scroll.setMaximumHeight(200) # prevent it from growing out of control
    layout.addWidget(progress_scroll, 1)

    # Overall Progress
    overall_progress_layout = QVBoxLayout()
    main_window.batch_overall_lbl = QLabel("Overall Progress:")
    main_window.batch_overall_lbl.setStyleSheet("font-weight: bold;")
    overall_progress_layout.addWidget(main_window.batch_overall_lbl)
    
    main_window.batch_overall_progress = QProgressBar()
    main_window.batch_overall_progress.setFixedHeight(20)
    main_window.batch_overall_progress.setTextVisible(True)
    overall_progress_layout.addWidget(main_window.batch_overall_progress)
    layout.addLayout(overall_progress_layout)

    # Start Button Layout
    bottom_layout = QHBoxLayout()
    
    threads_lbl = QLabel("Concurrent Threads:")
    threads_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
    bottom_layout.addWidget(threads_lbl)
    
    from PyQt5.QtWidgets import QSpinBox
    main_window.batch_thread_spinner = QSpinBox()
    main_window.batch_thread_spinner.setRange(1, 16)
    main_window.batch_thread_spinner.setValue(4)
    main_window.batch_thread_spinner.setFont(_ui_font(14))
    bottom_layout.addWidget(main_window.batch_thread_spinner)
    
    bottom_layout.addStretch()
    
    main_window.batch_start_btn = QPushButton("Start Parallel Processing")
    main_window.batch_start_btn.setObjectName("ButtonPurple")
    main_window.batch_start_btn.setMinimumHeight(50)
    main_window.batch_start_btn.setFont(_ui_font(16, QFont.Bold))
    bottom_layout.addWidget(main_window.batch_start_btn)
    
    layout.addLayout(bottom_layout)

    return widget