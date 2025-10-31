# Project ANWAYA

**Indian Sign Language to Marathi Translation System**

A PyQt5-based desktop application for collecting ISL gesture data, training LSTM models, and performing real-time sign language recognition.

## 🎯 Features

### 1. Data Collection
- Record ISL gesture videos with webcam
- MediaPipe integration for pose, hand, and face landmark detection
- Batch recording with configurable parameters
- Automatic landmark extraction and preprocessing
- Support for Marathi action names

### 2. Model Training
- LSTM neural network for sequence classification
- Automated training pipeline with progress tracking
- Early stopping and learning rate reduction
- Comprehensive training reports and visualizations
- Model performance metrics

### 3. Real-Time Recognition
- Live gesture recognition using webcam
- Visual feedback with MediaPipe landmarks
- Confidence-based predictions
- Support for Marathi text display
- Non-blocking thread execution

## 📋 Requirements

```
opencv-python
mediapipe
numpy
tensorflow
keras
scikit-learn
PyQt5
matplotlib
Pillow
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/PawanLambole/Project-Anwaya.git
cd Project-Anwaya
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Run the Application
```bash
python run_app.py
```

### Workflow

1. **Collect Data**
   - Click "Start New Collection"
   - Enter action name in Marathi (e.g., आभार, छान)
   - Set number of videos to record
   - Record gesture sequences

2. **Train Model**
   - Click "Train New Model" from sidebar
   - Wait for training to complete
   - Review training reports in `model report/` folder

3. **Real-Time Recognition**
   - Click "🤟 Real-Time Recognition"
   - Click "Start Camera"
   - Perform trained gestures
   - View predictions with confidence scores

## 📁 Project Structure

```
Project-Anwaya/
├── run_app.py                  # Application entry point
├── main_window.py              # Main application logic
├── ui_definitions.py           # UI components
├── splash_screen.py            # Startup splash screen
├── style.qss                   # Application styling
├── mediapipe_logic.py          # MediaPipe processing
├── recognition_logic.py        # Real-time recognition
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── model/                      # Trained models
│   ├── isl_model.keras        # LSTM model
│   └── label_encoder.pkl      # Label encoder
└── model report/               # Training reports
    ├── training_history.png
    ├── classification_report.txt
    └── training_summary.txt
```

## 🧠 Model Architecture

- **Input**: Sequences of 30 frames × 1662 features
  - Pose: 33 landmarks × 4 (x, y, z, visibility)
  - Face: 468 landmarks × 3 (x, y, z)
  - Left Hand: 21 landmarks × 3
  - Right Hand: 21 landmarks × 3

- **Architecture**:
  - LSTM (64 units, return_sequences=True)
  - Dropout (0.2)
  - LSTM (128 units)
  - Dropout (0.2)
  - Dense (64, ReLU)
  - Dropout (0.2)
  - Dense (32, ReLU)
  - Dense (num_actions, Softmax)

## 🎨 UI Theme

- Dark theme with blue (#0078D4) accent colors
- Support for Devanagari (Marathi) text
- Responsive layout with real-time video display
- Status indicators and progress bars

## 📊 Technical Details

- **Framework**: PyQt5
- **ML Framework**: TensorFlow/Keras
- **Computer Vision**: MediaPipe Holistic
- **Language**: Python 3.11+
- **Recognition FPS**: ~30 FPS
- **Confidence Threshold**: 50%

## 👥 Authors

B.Tech Final Year Project

## 📄 License

This project is part of an academic final year project.

## 🙏 Acknowledgments

- MediaPipe by Google for landmark detection
- TensorFlow/Keras for deep learning framework
- PyQt5 for the GUI framework

---

**Note**: Data folders (`ISL_Data/` and `ISL_Processed/`) are not included in the repository due to size constraints. You need to collect your own gesture data using the data collection feature.
