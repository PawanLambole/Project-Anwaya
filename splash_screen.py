from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDesktopWidget
from PyQt5.QtCore import Qt

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("SplashScreen")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(700, 450)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        title_layout = QHBoxLayout()
        title_layout.setAlignment(Qt.AlignCenter)
        
        proj_label = QLabel("Project ")
        proj_label.setObjectName("SplashTitleNormal")
        proj_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        title_layout.addWidget(proj_label)
        
        anv_label = QLabel("Anvaya")
        anv_label.setObjectName("SplashTitleSamarkan")
        anv_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_layout.addWidget(anv_label)
        
        layout.addLayout(title_layout)
        
        self.subtitle = QLabel("Personalised Customisable Sign Language to Text Converter")
        self.subtitle.setAlignment(Qt.AlignCenter)
        self.subtitle.setStyleSheet("color: #656D76; font-size: 16px;")
        layout.addWidget(self.subtitle)
        
        loading = QLabel("Loading Application...")
        loading.setObjectName("SplashLoading")
        loading.setAlignment(Qt.AlignCenter)
        layout.addWidget(loading)
        
        self.center_on_screen()

    def center_on_screen(self):
        """Centers the splash screen."""
        frame_geo = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        frame_geo.moveCenter(center_point)
        self.move(frame_geo.topLeft())