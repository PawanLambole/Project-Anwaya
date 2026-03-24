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

import argparse

# --- Parse arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Default', help='Name of the dataset folder')
args = parser.parse_args()

# --- 1. SET YOUR PARAMETERS ---
DATASET_NAME = args.dataset
PROCESSED_DATA_PATH = os.path.join('ISL_Processed', DATASET_NAME)
MODEL_DIR = os.path.join('model', DATASET_NAME)
os.makedirs(MODEL_DIR, exist_ok=True)
STOP_FLAG_PATH = os.path.join(MODEL_DIR, 'stop_training.flag')

print(f"Training on dataset: {DATASET_NAME}")

# Clean up any existing stop flag
if os.path.exists(STOP_FLAG_PATH):
    os.remove(STOP_FLAG_PATH)

RAW_DATA_PATH = os.path.join('ISL_Data', DATASET_NAME)

# Ensure actions are sorted so the LabelEncoder is consistent
if not os.path.exists(PROCESSED_DATA_PATH):
    print(f"Error: Directory {PROCESSED_DATA_PATH} does not exist.")
    exit(1)
    
# Filter processed actions to only include ones that still exist in raw data
valid_actions = set()
if os.path.exists(RAW_DATA_PATH):
    valid_actions = set([d for d in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, d))])

all_processed_actions = [d for d in os.listdir(PROCESSED_DATA_PATH) if os.path.isdir(os.path.join(PROCESSED_DATA_PATH, d))]

if valid_actions:
    valid_processed_actions = [d for d in all_processed_actions if d in valid_actions]
    zombie_folders = set(all_processed_actions) - valid_actions
    if zombie_folders:
        try:
            print(f"Warning: Ignoring {len(zombie_folders)} deleted/renamed folders found in processed data: {zombie_folders}")
        except UnicodeEncodeError:
            print(f"Warning: Ignoring {len(zombie_folders)} deleted/renamed folders found in processed data.")
else:
    valid_processed_actions = all_processed_actions # Fallback if raw data directory is missing
    
actions = np.array(sorted(valid_processed_actions))
if len(actions) == 0:
    print(f"Error: No valid matching actions found between {RAW_DATA_PATH} and {PROCESSED_DATA_PATH}.")
    exit(1)
    
num_actions = len(actions)

SEQUENCE_LENGTH = 30
NUM_FEATURES = 1662

# --- 2. LOAD, LABEL, AND PRE-PROCESS DATA (FIXED) ---
print("Loading data...")
sequences = []
labels = []

for i, action in enumerate(actions):
    print(f"[LOAD_PROGRESS] {i+1}/{num_actions}")
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
                # Silenced missing frame prints to avoid bloating the UI log during loading
                # print(f"Warning: Missing frame {frame_path}. Appending zeros.")
                window.append(np.zeros(NUM_FEATURES))
        
        sequences.append(window)
        labels.append(action)

print(f"[LOAD_PROGRESS] {num_actions}/{num_actions}")
print(f"Loaded {len(sequences)} total sequences.")

# --- 3. CONVERT LABELS AND SEQUENCES ---
label_encoder = LabelEncoder()
integer_encoded_labels = label_encoder.fit_transform(labels)
y = to_categorical(integer_encoded_labels, num_classes=num_actions)

X = np.array(sequences, dtype='float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=32, stratify=y)

print(f"Data shape (X_train): {X_train.shape}")
print(f"Labels shape (y_train): {y_train.shape}")

# --- 4. BUILD THE LSTM MODEL ---

print("Building model...")
model = Sequential()
model.add(Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_actions, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. TRAIN THE MODEL ---

# --- NEW: Define BOTH callbacks ---
from keras.callbacks import Callback

class GracefulStopCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if os.path.exists(STOP_FLAG_PATH):
            print(f"\n[INFO] Graceful stop requested (file {STOP_FLAG_PATH} found).")
            print("[INFO] Stopping training early but saving the model and generating reports.")
            self.model.stop_training = True

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2, # Reduce LR by a factor of 5 (1/5 = 0.2)
    patience=5, # Reduce if no improvement for 5 epochs
    min_lr=0.00001, # Don't go below this
    verbose=1
)
graceful_stop_callback = GracefulStopCallback()
# --- END NEW ---

print("Training model...")
EPOCHS = 150
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[early_stop_callback, reduce_lr_callback, graceful_stop_callback] # <-- NEW: Add graceful stop
)

print("="*50)
print("TRAINING COMPLETE")
print("="*50 + "\n")

print("Model training complete.")

# --- 6. EVALUATE THE MODEL ---
print("Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# --- 7. NEW: PLOT TRAINING HISTORY ---
print("Plotting training history...")

# Create model report directory if it doesn't exist
REPORT_DIR = os.path.join('model report', DATASET_NAME)
os.makedirs(REPORT_DIR, exist_ok=True)

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

plt.savefig(os.path.join(REPORT_DIR, 'training_history.png'))
print(f"Training history plot saved as '{REPORT_DIR}/training_history.png'")
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
report_path = os.path.join(REPORT_DIR, 'classification_report.txt')
with open(report_path, 'w', encoding='utf-8') as f: # <-- FIX 2: Fix for the UnicodeEncodeError
    f.write(report)
print(f"Classification report saved as '{report_path}'")

# Try to print, but handle encoding errors for Marathi text
try:
    print(report)
except UnicodeEncodeError:
    print("Classification report contains Marathi characters.")
    print(f"Please open '{report_path}' to view the full report.")

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

FINALLY SAVED FILES:
-----------
Model: model/{DATASET_NAME}/{DATASET_NAME}_model.keras
Label Encoder: model/{DATASET_NAME}/{DATASET_NAME}_label_encoder.pkl
Training History Plot: {REPORT_DIR}/training_history.png
Classification Report: {REPORT_DIR}/classification_report.txt
Training Summary: {REPORT_DIR}/training_summary.txt

===============================================
"""

summary_path = os.path.join(REPORT_DIR, 'training_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary)
print(f"Training summary saved as '{summary_path}'")
# --- END NEW ---

# --- 9. SAVE MODEL AND ENCODER (Renumbered) ---
# Create model directory if it doesn't exist
MODEL_DIR = os.path.join('model', DATASET_NAME)
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, f'{DATASET_NAME}_model.keras')
model.save(model_path)
print(f"Model saved as '{model_path}'")

encoder_path = os.path.join(MODEL_DIR, f'{DATASET_NAME}_label_encoder.pkl')
with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"Label encoder saved as '{encoder_path}'")

print("\n--- SCRIPT FINISHED ---")