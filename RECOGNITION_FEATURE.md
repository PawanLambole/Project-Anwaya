# Real-Time ISL Recognition - Feature Documentation

## Overview
The Real-Time ISL Recognition feature has been integrated into Project ANWAYA, allowing users to test trained models by recognizing Indian Sign Language gestures in real-time using their webcam.

## Files Added/Modified

### New Files:
1. **`recognition_logic.py`** - Contains the `RecognitionWorker` thread class that handles:
   - Loading trained model and label encoder
   - Real-time video capture and processing
   - MediaPipe landmark detection
   - LSTM model predictions
   - Frame-by-frame gesture recognition

2. **`test_recognition_ui.py`** - Test script to verify the integration

### Modified Files:
1. **`ui_definitions.py`**
   - Added `create_recognition_widget()` function
   - Added "Real-Time Recognition" button to home page
   - Created recognition page UI with video display, prediction banner, and controls

2. **`main_window.py`**
   - Added `STATE_RECOGNITION` state
   - Imported `RecognitionWorker` class
   - Added recognition navigation methods:
     - `go_to_recognition()`
     - `start_recognition()`
     - `stop_recognition()`
     - `update_recognition_frame()`
     - `update_prediction()`
     - `update_recognition_status()`
     - `show_recognition_error()`
   - Updated state management to handle recognition mode
   - Added cleanup for recognition worker in `closeEvent()`

3. **`style.qss`**
   - Added comprehensive styling for recognition page:
     - Video display area with blue border
     - Prediction banner with orange background
     - Start/Stop camera buttons (green/red)
     - Status indicators
     - Marathi text support

## Features

### 1. Video Display
- Real-time webcam feed with 640x480 minimum size
- MediaPipe landmarks overlay showing:
  - **Green** - Body pose connections
  - **Blue** - Left hand connections
  - **Red** - Right hand connections
- Mirrored view for natural interaction

### 2. Prediction Banner
- Displays current recognized action in Marathi
- Shows confidence percentage (e.g., "आभार: 98%")
- Orange background matching app theme
- Updates in real-time

### 3. Status Messages
- **"Collecting frames..."** - Initial startup (0-29 frames)
- **"Waiting..."** - Low confidence prediction
- **"[Action]: XX%"** - High confidence prediction with percentage

### 4. Controls
- **Start Camera** (Green button) - Begins recognition
- **Stop Camera** (Red button) - Stops recognition
- **Back to Home** - Returns to main menu
- Status indicator (● Green/Gray) shows camera state

## How It Works

### Recognition Pipeline:
1. **Frame Capture** - Captures video frame from webcam
2. **Mirror Flip** - Flips frame horizontally for natural view
3. **MediaPipe Processing** - Extracts pose, face, and hand landmarks
4. **Landmark Drawing** - Overlays colored connections on frame
5. **Keypoint Extraction** - Converts landmarks to 1662-feature vector
6. **Sequence Building** - Maintains sliding window of 30 frames
7. **Prediction** - When 30 frames collected:
   - Passes sequence through LSTM model
   - Gets confidence scores for each action
   - If confidence > 50%, displays prediction
8. **Display Update** - Shows frame and prediction in UI

### Technical Details:
- **Frame Rate**: ~30 FPS
- **Sequence Length**: 30 frames (same as training)
- **Feature Count**: 1662 keypoints per frame
  - Pose: 33 landmarks × 4 (x, y, z, visibility) = 132
  - Face: 468 landmarks × 3 (x, y, z) = 1404
  - Left Hand: 21 landmarks × 3 = 63
  - Right Hand: 21 landmarks × 3 = 63
- **Confidence Threshold**: 50%

## Usage Instructions

### Prerequisites:
1. Trained model file: `isl_model.keras`
2. Label encoder file: `label_encoder.pkl`
3. These are created by running `train_model.py`

### Steps:
1. Launch the application: `python run_app.py`
2. From the home page, click **"🤟 Real-Time Recognition"**
3. Click **"Start Camera"** to begin recognition
4. Perform ISL gestures in front of the camera
5. Watch predictions appear in the orange banner
6. Click **"Stop Camera"** to pause recognition
7. Click **"← Back to Home"** to return to main menu

## Error Handling

### Model Not Found Error:
```
Model or Label Encoder not found.
Please train the model first.
```
**Solution**: Run training first via "Train New Model" button

### Camera Error:
```
Could not open webcam
```
**Solution**: 
- Check camera permissions
- Ensure no other app is using the camera
- Try different camera index in sidebar settings

## Integration with Existing Features

### State Management:
- Recognition runs in separate thread (non-blocking)
- Sidebar disabled during recognition
- Training disabled during recognition
- Proper cleanup when switching pages

### Theme Consistency:
- Uses same color scheme as collection page
- Orange accent color (#0078D4) for predictions
- Dark background (#1E1E1E)
- Consistent button styles and spacing

## Performance Notes

- Recognition runs in separate QThread to prevent UI freezing
- Model prediction optimized with `verbose=0`
- Frame processing at ~30 FPS
- Low latency (~100ms) between gesture and prediction
- Efficient memory usage with sliding window approach

## Future Enhancements

Potential improvements:
1. Add recording of recognized sequences
2. Implement sentence formation from multiple gestures
3. Add confidence threshold adjustment slider
4. Support multiple camera selection
5. Add gesture history log
6. Export predictions to text file
7. Add audio feedback for predictions

## Troubleshooting

### Issue: Predictions are inaccurate
- Ensure good lighting conditions
- Stand in clear view of camera
- Perform gestures similar to training data
- Check if model is properly trained

### Issue: Video is laggy
- Close other applications using camera
- Reduce background processes
- Check CPU usage

### Issue: No predictions shown
- Wait for 30 frames to be collected
- Check confidence threshold (default 50%)
- Verify gesture is in trained actions

## Testing

Run the test script to verify integration:
```bash
python test_recognition_ui.py
```

This will:
- Verify all imports are successful
- Launch the application
- Show recognition page is accessible

## Credits

This feature integrates MediaPipe Holistic for landmark detection and TensorFlow/Keras LSTM model for gesture classification, maintaining consistency with the data collection and training pipeline.
