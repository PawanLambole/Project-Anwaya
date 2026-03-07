import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Import our new modules
from splash_screen import SplashScreen
from main_window import CollectionApp

def load_stylesheet():
    """Loads the external QSS stylesheet."""
    try:
        with open("style.qss", "r") as f:
            return f.read()
    except FileNotFoundError:
        print("Warning: style.qss not found. Using default styles.")
        return ""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    from PyQt5.QtGui import QFontDatabase
    import os
    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samarkan.ttf")
    if os.path.exists(font_path):
        QFontDatabase.addApplicationFont(font_path)
    
    # Load and apply the stylesheet
    stylesheet = load_stylesheet()
    app.setStyleSheet(stylesheet)
    
    # 1. Create and show the splash screen
    splash = SplashScreen()
    splash.show()
    
    # 2. Create the main window (it's hidden)
    main_window = CollectionApp()
    
    # 3. Set a timer to close the splash and show the main window
    QTimer.singleShot(3000, lambda: {
        print("Splash screen finished. Showing main window."),
        main_window.show(),
        splash.close()
    })
    
    sys.exit(app.exec_())