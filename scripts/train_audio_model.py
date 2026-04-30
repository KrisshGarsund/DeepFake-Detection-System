"""
Training script for the Audio Deepfake Detector.
Uses CNN on mel-spectrograms.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import glob

from config import (
    AUDIO_SAMPLE_RATE, AUDIO_DURATION, N_MELS, HOP_LENGTH,
    AUDIO_BATCH_SIZE, AUDIO_EPOCHS, AUDIO_LEARNING_RATE,
    AUDIO_MODEL_PATH, AUDIO_RAW_DIR, MODELS_DIR
)
from models.audio_detector import build_audio_model
from utils.preprocessing import extract_mel_spectrogram


# Fake audio sources from the dataset
FAKE_DIRS = [
    "FlashSpeech", "OpenAI", "VoiceBox", "VALLE",
    "xTTS", "NaturalSpeech3", "PromptTTS2", "seedtts_files"
]
REAL_DIRS = ["real_samples"]


def load_audio_data():
    """Load audio file paths and labels from extracted directories."""
    paths = []
    labels = []
    
    print("Scanning for audio files...")
    
    # Fake audio
    for fake_dir_name in FAKE_DIRS:
        fake_dir = os.path.join(AUDIO_RAW_DIR, fake_dir_name)
        if os.path.exists(fake_dir):
            wav_files = glob.glob(os.path.join(fake_dir, "**", "*.wav"), recursive=True)
            paths.extend(wav_files)
            labels.extend([1] * len(wav_files))
            print(f"  {fake_dir_name}: {len(wav_files)} fake audio files")
    
    # Real audio
    for real_dir_name in REAL_DIRS:
        real_dir = os.path.join(AUDIO_RAW_DIR, real_dir_name)
        if os.path.exists(real_dir):
            wav_files = glob.glob(os.path.join(real_dir, "**", "*.wav"), recursive=True)
            paths.extend(wav_files)
            labels.extend([0] * len(wav_files))
            print(f"  {real_dir_name}: {len(wav_files)} real audio files")
    
    print(f"\nTotal audio files: {len(paths)}")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Label {u} ({'Real' if u == 0 else 'Fake'}): {c}")
    
    return paths, labels


def preprocess_audio_batch(paths):
    """Extract mel spectrograms for a batch of audio files."""
    spectrograms = []
    valid_indices = []
    
    for i, path in enumerate(paths):
        try:
            mel = extract_mel_spectrogram(
                path, sr=AUDIO_SAMPLE_RATE, duration=AUDIO_DURATION,
                n_mels=N_MELS, hop_length=HOP_LENGTH
            )
            spectrograms.append(mel)
            valid_indices.append(i)
        except Exception as e:
            print(f"  Skipping {os.path.basename(path)}: {e}")
    
    return np.array(spectrograms), valid_indices


def train():
    """Main training function for audio model."""
    print("=" * 60)
    print("  DeepfakeGuard — Audio Model Training")
    print("=" * 60)
    
    # Load data
    paths, labels = load_audio_data()
    
    if len(paths) == 0:
        print("\n[ERROR] No audio files found! Please extract archive (1).zip first.")
        return
    
    # Preprocess: extract mel spectrograms
    print("\nExtracting mel spectrograms (this may take a while)...")
    
    # Process in chunks to manage memory
    chunk_size = 200
    all_specs = []
    all_labels_valid = []
    
    for start in range(0, len(paths), chunk_size):
        end = min(start + chunk_size, len(paths))
        chunk_paths = paths[start:end]
        chunk_labels = labels[start:end]
        
        print(f"  Processing {start + 1}-{end} of {len(paths)}...")
        specs, valid_idx = preprocess_audio_batch(chunk_paths)
        
        all_specs.extend(specs)
        all_labels_valid.extend([chunk_labels[i] for i in valid_idx])
    
    X = np.array(all_specs)
    y = np.array(all_labels_valid, dtype=np.float32)
    
    # Add channel dimension for CNN: (N, n_mels, time_steps, 1)
    X = X[..., np.newaxis]
    
    print(f"\nData shape: {X.shape}")
    print(f"Labels: {len(y)} ({sum(y == 0):.0f} real, {sum(y == 1):.0f} fake)")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")
    
    # Build model
    input_shape = X_train.shape[1:]  # (n_mels, time_steps, 1)
    print(f"\nBuilding audio CNN model (input: {input_shape})...")
    model = build_audio_model(input_shape=input_shape, learning_rate=AUDIO_LEARNING_RATE)
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            AUDIO_MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        )
    ]
    
    # Handle class imbalance
    n_real = sum(y_train == 0)
    n_fake = sum(y_train == 1)
    total = n_real + n_fake
    class_weight = {
        0: total / (2 * n_real) if n_real > 0 else 1.0,
        1: total / (2 * n_fake) if n_fake > 0 else 1.0
    }
    print(f"Class weights: {class_weight}")
    
    # Train
    print("\n--- Training ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=AUDIO_EPOCHS,
        batch_size=AUDIO_BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save
    model.save(AUDIO_MODEL_PATH)
    print(f"\nModel saved to: {AUDIO_MODEL_PATH}")
    
    # Evaluate
    print("\n--- Evaluation ---")
    y_pred_proba = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_val.astype(int), y_pred, target_names=["Real", "Fake"]))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_val.astype(int), y_pred))
    
    # Save metrics
    metrics = {
        "val_accuracy": float(max(history.history.get("val_accuracy", [0]))),
        "val_auc": float(max(history.history.get("val_auc", [0]))),
        "total_audio": len(paths),
        "input_shape": list(input_shape)
    }
    
    with open(os.path.join(MODELS_DIR, "audio_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved.")


if __name__ == "__main__":
    train()
