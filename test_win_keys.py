import sys
import ctypes
from ctypes.wintypes import MSG
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QCursor
import win32con
import win32api
import win32gui

class FramelessSnapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 1. We start with a standard FramelessWindowHint
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setGeometry(100, 100, 800, 600)
        
        self.central_widget = QWidget()
        self.central_widget.setStyleSheet("background-color: #2E3440; color: white;")
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.label = QLabel("I am a frameless window!\nTry Win+Left, Win+Right, Win+Up, Win+Down", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 24px;")
        self.layout.addWidget(self.label)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        self.layout.addWidget(self.close_btn)

    def showEvent(self, event):
        super().showEvent(event)
        # 2. Once shown, we inject WS_THICKFRAME (resizable), WS_CAPTION (titlebar), WS_MAXIMIZEBOX, WS_MINIMIZEBOX
        # This fools Windows into thinking it's a standard window, so Snap Assist works!
        hwnd = int(self.winId())
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
        style |= win32con.WS_THICKFRAME | win32con.WS_CAPTION | win32con.WS_MAXIMIZEBOX | win32con.WS_MINIMIZEBOX | win32con.WS_SYSMENU
        win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
        
        # We need to tell Windows to re-evaluate the window frame
        win32gui.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 
                              win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | 
                              win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED)

    # 3. We intercept native Windows messages to hide the titlebar we just added and handle dragging/resizing natively
    def nativeEvent(self, eventType, message):
        msg = MSG.from_address(message.__int__())
        
        # Tell Windows we don't want the default non-client area painted (this hides the titlebar)
        if msg.message == win32con.WM_NCCALCSIZE:
            if msg.wParam:
                return True, 0
            
        # Handle hit testing (clicking) to allow resizing and dragging
        elif msg.message == win32con.WM_NCHITTEST:
            # Get global mouse position from the message
            x = win32api.LOWORD(msg.lParam)
            if x & 0x8000: x -= 0x10000 # handle signed 16-bit
                
            y = win32api.HIWORD(msg.lParam)
            if y & 0x8000: y -= 0x10000

            # Convert to local coordinates within the window
            local_pos = self.mapFromGlobal(QPoint(x, y))
            hx = local_pos.x()
            hy = local_pos.y()
            
            bw = 8 # Border width for resizing
            
            # Simple hit test for edges
            left = (hx < bw)
            right = (hx > self.width() - bw)
            top = (hy < bw)
            bottom = (hy > self.height() - bw)
            
            res = 0
            if top and left: res = win32con.HTTOPLEFT
            elif top and right: res = win32con.HTTOPRIGHT
            elif bottom and left: res = win32con.HTBOTTOMLEFT
            elif bottom and right: res = win32con.HTBOTTOMRIGHT
            elif top: res = win32con.HTTOP
            elif bottom: res = win32con.HTBOTTOM
            elif left: res = win32con.HTLEFT
            elif right: res = win32con.HTRIGHT
            
            if res != 0:
                return True, res

            # Important: define the dragging area (the "title bar")
            # If we click near the top (e.g. top 50 pixels), treat it as the caption for dragging and snapping
            if hy < 50:
                return True, win32con.HTCAPTION
                
            return True, win32con.HTCLIENT

        return super().nativeEvent(eventType, message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = FramelessSnapWindow()
    w.show()
    sys.exit(app.exec_())
