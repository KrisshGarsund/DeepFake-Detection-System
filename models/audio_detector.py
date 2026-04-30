"""
Audio Deepfake Detector — CNN on mel-spectrograms.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_audio_model(input_shape=(128, 216, 1), learning_rate=1e-4):
    """
    Build a CNN-based audio deepfake classifier.
    Input: mel-spectrogram of shape (n_mels, time_steps, 1).
    Default time_steps ~216 for 5s at sr=22050, hop=512.
    """
    
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        # Classifier
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")  # Binary: 0=Real, 1=Fake
    ], name="deepfake_audio_detector")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    
    return model
