import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report # <-- NEW: Import
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau # <-- NEW: Import

# --- 1. SET YOUR PARAMETERS ---
PROCESSED_DATA_PATH = os.path.join('ISL_Processed')
# Ensure actions are sorted so the LabelEncoder is consistent
actions = np.array(sorted([d for d in os.listdir(PROCESSED_DATA_PATH) if os.path.isdir(os.path.join(PROCESSED_DATA_PATH, d))]))
num_actions = len(actions)

SEQUENCE_LENGTH = 30
NUM_FEATURES = 1662

# --- 2. LOAD, LABEL, AND PRE-PROCESS DATA (FIXED) ---
print("Loading data...")
sequences = []
labels = []

for action in actions:
    action_path = os.path.join(PROCESSED_DATA_PATH, action)
    
    video_folders = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
    
    for video_folder in video_folders:
        video_path = os.path.join(action_path, video_folder)
        
        window = []
        
        for frame_num in range(SEQUENCE_LENGTH):
            frame_file = f"{frame_num}.npy"
            frame_path = os.path.join(video_path, frame_file)
            
            if os.path.exists(frame_path):
                res = np.load(frame_path)
                window.append(res)
            else:
                print(f"Warning: Missing frame {frame_path}. Appending zeros.")
                window.append(np.zeros(NUM_FEATURES))
        
        sequences.append(window)
        labels.append(action)

print(f"Loaded {len(sequences)} total sequences.")

# --- 3. CONVERT LABELS AND SEQUENCES ---
label_encoder = LabelEncoder()
integer_encoded_labels = label_encoder.fit_transform(labels)
y = to_categorical(integer_encoded_labels, num_classes=num_actions)

X = np.array(sequences, dtype='float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

print(f"Data shape (X_train): {X_train.shape}")
print(f"Labels shape (y_train): {y_train.shape}")

# --- 4. BUILD THE IMPROVED LSTM MODEL ---

print("Building improved model...")
model = Sequential()
model.add(Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)))

# First LSTM layer - increased units
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(Dropout(0.3))

# Second LSTM layer - increased units
model.add(LSTM(256, return_sequences=True, activation='tanh'))
model.add(Dropout(0.3))

# Third LSTM layer - for better temporal feature extraction
model.add(LSTM(128, return_sequences=False, activation='tanh'))
model.add(Dropout(0.3))

# Dense layers with batch normalization concept
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(num_actions, activation='softmax'))

# Use Adam optimizer with custom learning rate
from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. TRAIN THE MODEL ---

# --- NEW: Define improved callbacks ---
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=20,  # Increased patience for better convergence
    restore_best_weights=True,
    verbose=1
)

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Reduce LR by half (less aggressive)
    patience=8,  # Wait longer before reducing
    min_lr=0.000001,  # Lower minimum learning rate
    verbose=1
)
# --- END NEW ---

print("Training model with improved architecture...")
EPOCHS = 200  # Increased epochs for better learning
BATCH_SIZE = 8  # Add batch size for better gradient updates

history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[early_stop_callback, reduce_lr_callback],
    verbose=1
)

print("Model training complete.")

# --- 6. EVALUATE THE MODEL ---
print("Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# --- 7. NEW: PLOT TRAINING HISTORY ---
print("Plotting training history...")

# Create model report directory if it doesn't exist
os.makedirs('model report', exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot Accuracy
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(loc='upper left')

# Plot Loss
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(loc='upper right')

plt.savefig('model report/training_history.png')
print("Training history plot saved as 'model report/training_history.png'")
# plt.show() # Uncomment this if you are running in a local environment

# --- 8. NEW: DETAILED CLASSIFICATION REPORT ---
print("Generating classification report...")
# Get predictions on the test set
y_pred_probs = model.predict(X_test)
# Convert probabilities to class labels
y_pred_labels = np.argmax(y_pred_probs, axis=1)
# Convert one-hot y_test back to class labels
y_true_labels = np.argmax(y_test, axis=1)

# Print the report
report = classification_report(
    y_true_labels,
    y_pred_labels,
    target_names=label_encoder.classes_,
    zero_division=0 # <-- FIX 1: Fix for the UndefinedMetricWarning
)

# Save the report to a file first
with open('model report/classification_report.txt', 'w', encoding='utf-8') as f: # <-- FIX 2: Fix for the UnicodeEncodeError
    f.write(report)
print("Classification report saved as 'model report/classification_report.txt'")

# Try to print, but handle encoding errors for Marathi text
try:
    print(report)
except UnicodeEncodeError:
    print("Classification report contains Marathi characters.")
    print("Please open 'model report/classification_report.txt' to view the full report.")

# --- NEW: SAVE TRAINING SUMMARY ---
print("Saving training summary...")
import datetime

summary = f"""
===============================================
ISL MODEL TRAINING SUMMARY
===============================================
Training Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET INFORMATION:
-------------------
Total Sequences: {len(sequences)}
Training Samples: {len(X_train)}
Test Samples: {len(X_test)}
Actions: {', '.join(actions)}
Number of Actions: {num_actions}
Sequence Length: {SEQUENCE_LENGTH} frames
Features per Frame: {NUM_FEATURES}

MODEL ARCHITECTURE:
------------------
{model.to_json()}

TRAINING PARAMETERS:
-------------------
Epochs Run: {len(history.history['loss'])}
Max Epochs: {EPOCHS}
Optimizer: Adam
Loss Function: Categorical Crossentropy
Early Stopping: Yes (patience=10, monitor=val_loss)
Learning Rate Reduction: Yes (factor=0.2, patience=5)

FINAL RESULTS:
--------------
Test Accuracy: {accuracy * 100:.2f}%
Test Loss: {loss:.4f}
Final Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%
Final Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%
Final Training Loss: {history.history['loss'][-1]:.4f}
Final Validation Loss: {history.history['val_loss'][-1]:.4f}

SAVED FILES:
-----------
Model: model/isl_model.keras
Label Encoder: model/label_encoder.pkl
Training History Plot: model report/training_history.png
Classification Report: model report/classification_report.txt
Training Summary: model report/training_summary.txt

===============================================
"""

with open('model report/training_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)
print("Training summary saved as 'model report/training_summary.txt'")
# --- END NEW ---

# --- 9. SAVE MODEL AND ENCODER (Renumbered) ---
# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

model.save('model/isl_model.keras')
print("Model saved as 'model/isl_model.keras'")

with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("Label encoder saved as 'model/label_encoder.pkl'")

print("\n--- SCRIPT FINISHED ---")