# In src/train.py
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D

# --- 1. DATA PREPARATION ---

DATA_PATH = '../data/'

# Emotion mapping from the RAVDESS dataset filenames
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_features(file_path):
    """Extracts MFCCs from an audio file."""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load data and extract features
features = []
for dirpath, _, filenames in os.walk(DATA_PATH):
    for filename in filenames:
        if filename.endswith('.wav'):
            emotion_code = filename.split('-')[2]
            if emotion_code in emotion_map:
                emotion = emotion_map[emotion_code]
                file_path = os.path.join(dirpath, filename)
                data = extract_features(file_path)
                if data is not None:
                    features.append([data, emotion])

# Convert features and labels to numpy arrays
X = np.array([item[0] for item in features])
y = np.array([item[1] for item in features])

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Reshape data for the 1D CNN
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# --- 2. BUILD AND TRAIN THE MODEL ---

model = Sequential([
    Conv1D(256, 5, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=5),
    Dropout(0.3),
    Conv1D(128, 5, padding='same', activation='relu'),
    MaxPooling1D(pool_size=5),
    Dropout(0.3),
    GlobalAveragePooling1D(),
    Dense(len(emotion_map), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Starting model training...")
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# --- 3. SAVE THE MODEL ---
if not os.path.exists('../models'):
    os.makedirs('../models')
model.save('../models/speech_emotion_model.keras')
print("Model saved successfully.")

# Save the label encoder too, we'll need it for predictions
import pickle
with open('../models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("Label encoder saved.")