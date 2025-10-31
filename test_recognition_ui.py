"""
Test script to verify the recognition UI integration
Run this to check if all components are properly connected
"""

import sys
from PyQt5.QtWidgets import QApplication

# Try to import main components
try:
    from main_window import CollectionApp
    print("✓ Successfully imported CollectionApp")
except Exception as e:
    print(f"✗ Error importing CollectionApp: {e}")
    sys.exit(1)

try:
    from recognition_logic import RecognitionWorker
    print("✓ Successfully imported RecognitionWorker")
except Exception as e:
    print(f"✗ Error importing RecognitionWorker: {e}")
    sys.exit(1)

try:
    from ui_definitions import create_recognition_widget
    print("✓ Successfully imported create_recognition_widget")
except Exception as e:
    print(f"✗ Error importing create_recognition_widget: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("All imports successful!")
print("="*50)
print("\nStarting application...")
print("Note: Recognition will require 'isl_model.keras' and 'label_encoder.pkl'")
print("      to be present in the project directory.")
print("="*50)

# Create and run the application
app = QApplication(sys.argv)
window = CollectionApp()
window.show()
sys.exit(app.exec_())
