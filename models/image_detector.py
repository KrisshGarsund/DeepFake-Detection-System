"""
Image Deepfake Detector — MobileNetV2-based binary classifier.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_image_model(input_shape=(224, 224, 3), learning_rate=1e-4):
    """Build a MobileNetV2-based deepfake image classifier."""
    
    # Load pre-trained MobileNetV2 (without top classification layers)
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")  # Binary: 0=Real, 1=Fake
    ], name="deepfake_image_detector")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    
    return model


def unfreeze_and_finetune(model, learning_rate=1e-5, unfreeze_from=100):
    """Unfreeze top layers of base model for fine-tuning."""
    base = model.layers[0]  # MobileNetV2
    base.trainable = True
    
    # Freeze all layers before `unfreeze_from`
    for layer in base.layers[:unfreeze_from]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    
    return model
