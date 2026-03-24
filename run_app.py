import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFontDatabase, QFont

# Import our new modules
from splash_screen import SplashScreen

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def configure_application_font(app: QApplication) -> None:
    """Sets a default app font that supports Devanagari, if available."""
    # Prefer fonts that include Devanagari coverage for the overall UI.
    # Decorative fonts (like Samarkan) should be applied selectively via QSS.
    font_candidates = [
        "Nirmala UI",
        "Mangal",
        "Noto Sans Devanagari",
    ]

    # Optionally register bundled fonts so QSS selectors can find them.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bundled_font_path = os.path.join(base_dir, "samarkan.ttf")
    if os.path.exists(bundled_font_path):
        QFontDatabase.addApplicationFont(bundled_font_path)

    available_families = set(QFontDatabase().families())
    for family in font_candidates:
        if family and family in available_families:
            current = app.font()
            current.setFamily(family)
            app.setFont(current)
            return

def load_stylesheet():
    """Loads the external QSS stylesheet."""
    try:
        with open("style.qss", "r") as f:
            return f.read()
    except FileNotFoundError:
        print("Warning: style.qss not found. Using default styles.")
        return ""

import traceback

def exception_hook(exc_type, exc_value, exc_traceback):
    print("\n--- UNCAUGHT EXCEPTION ---")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("--------------------------\n")
    sys.exit(1)

sys.excepthook = exception_hook

if __name__ == "__main__":
    app = QApplication(sys.argv)

    configure_application_font(app)
    
    # Load and apply the stylesheet
    stylesheet = load_stylesheet()
    app.setStyleSheet(stylesheet)
    
    # 1. Create and show the splash screen
    splash = SplashScreen()
    splash.show()
    app.processEvents()  # Force the splash screen to render before heavy initialization
    
    # IMPORT HERE so that heavy libraries (TensorFlow/MediaPipe) load while the splash screen is visible
    from main_window import CollectionApp
    
    # 2. Create the main window (it's hidden)
    main_window = CollectionApp()
    
    # 3. Set a timer to close the splash and show the main window
    def show_main():
        print("Splash screen finished. Showing main window.")
        main_window.show()
        splash.close()
        
    QTimer.singleShot(3000, show_main)
    
    sys.exit(app.exec_())