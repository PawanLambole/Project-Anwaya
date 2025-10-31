from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QDesktopWidget
from PyQt5.QtCore import Qt

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("SplashScreen")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(500, 300)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        title = QLabel("Project ANWAYA")
        title.setObjectName("SplashTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
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