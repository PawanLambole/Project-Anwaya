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
opencv-python==4.10.0.84
numpy==1.26.4
mediapipe==0.10.14
PyQt5==5.15.11
tensorflow==2.16.1
scikit-learn==1.5.2
matplotlib==3.9.2
protobuf==4.25.3
psutil==5.9.8
google-generativeai==0.3.2
pywin32
```

## 🔑 Google Generative AI Setup (Optional but Recommended)

For **Marathi grammar correction** feature, you'll need a Google Generative AI API key:

1. Go to https://aistudio.google.com/apikey
2. Click "Get API Key"
3. Create a new API key in Google Cloud
4. Create a file named `api_key.txt` in the project root
5. Paste your API key in `api_key.txt`

**Note:** 
- The app works without this key, but grammar correction will be disabled
- Free tier includes limited API calls per month
- See `api_key_template.txt` for reference

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

## 🔧 Troubleshooting

### Application won't start (silent exit with no error)

1. **Verify your setup**:
   ```bash
   python verify_setup.py
   ```
   This will check all dependencies and configuration.

2. **Run with debug output**:
   - Windows: Use `run_app_debug.bat` instead of `start_app.bat`
   - This will show any error messages that normally get hidden

3. **Common issues**:
   - **Missing dependencies**: Run `pip install -r requirements.txt`
   - **Python version too old**: Requires Python 3.11+. Check with `python --version`
   - **Virtual environment not activated**: Run `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
   - **Missing pywin32**: On Windows, run `pip install pywin32` if you see import errors

4. **If using a non-Windows system**:
   - Some Windows-specific features are disabled automatically
   - Use `python run_app.py` directly (not the .bat file)

### Other issues

- **Camera not working**: The app will auto-detect your webcam. Check Device Manager to ensure your camera is connected.
- **GPU not detected**: The app will automatically fall back to CPU. GPU support requires CUDA/cuDNN installation.
- **Out of memory**: Reduce the number of frames or video resolution in settings.

### Grammar Correction Issues

- **"Gemini API key not configured"**: Create `api_key.txt` with your API key from https://aistudio.google.com/apikey
- **"Invalid API key"**: Check that your API key is correct and not expired. Go to https://console.cloud.google.com/ to verify.
- **"Rate limit exceeded"**: You've exceeded the free tier limit. Wait a bit or upgrade your plan at https://console.cloud.google.com/
- **"Grammar correction disabled"**: The app works fine without an API key. Sentences won't be auto-corrected, but recognition still works.
- **Grammar correction not working**: The feature is optional. If API key is missing or invalid, the app will use the original recognized text.

## 🙏 Acknowledgments

- MediaPipe by Google for landmark detection
- TensorFlow/Keras for deep learning framework
- PyQt5 for the GUI framework

---

**Note**: Data folders (`ISL_Data/` and `ISL_Processed/`) are not included in the repository due to size constraints. You need to collect your own gesture data using the data collection feature.
