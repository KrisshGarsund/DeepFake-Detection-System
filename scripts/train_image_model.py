"""
Training script for the Image Deepfake Detector.
Uses MobileNetV2 transfer learning on the image dataset.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json

from config import (
    IMAGE_SIZE, IMAGE_BATCH_SIZE, IMAGE_EPOCHS, IMAGE_LEARNING_RATE,
    IMAGE_MODEL_PATH, IMAGE_RAW_DIR, TEST_LABELS_CSV, VAL_LABELS_CSV, MODELS_DIR
)
from models.image_detector import build_image_model, unfreeze_and_finetune


def build_image_index():
    """Build a dictionary mapping filename to its full absolute path for O(1) lookup."""
    index = {}
    print("Building image file index. Scanning available directories...")
    
    search_dirs = [
        IMAGE_RAW_DIR,
        os.path.join(IMAGE_RAW_DIR, "Dataset"),
        os.path.join(IMAGE_RAW_DIR, "train"),
        os.path.join(IMAGE_RAW_DIR, "test"),
        os.path.join(IMAGE_RAW_DIR, "val"),
        os.path.join(IMAGE_RAW_DIR, "my_real_vs_ai_dataset", "ai_images"),
        os.path.join(IMAGE_RAW_DIR, "my_real_vs_ai_dataset", "real_images")
    ]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    index[file] = os.path.join(root, file)
                    
    print(f"Indexed {len(index)} total reachable image files.")
    return index


def load_image_paths_and_labels():
    """Load image file paths and labels using the O(1) file index."""
    all_paths = []
    all_labels = []
    
    # Pre-build index of all available files
    file_index = build_image_index()
    
    # Try loading from CSV label files mapping to the index
    for csv_path in [VAL_LABELS_CSV, TEST_LABELS_CSV]:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} entries from {os.path.basename(csv_path)}")
            
            found = 0
            for _, row in df.iterrows():
                filename = row['filename']
                label = int(row['label'])
                
                # O(1) Lookup
                if filename in file_index:
                    all_paths.append(file_index[filename])
                    all_labels.append(label)
                    found += 1
            print(f"  -> Successfully matched {found} out of {len(df)} entries.")
    
    # If standard CSV data was low, fallback to scanning directories where class is inferred by folder name
    if len(all_paths) < 1000:
        print("Not enough images matched via CSV. Falling back to directory-based class detection...")
        for class_name, is_fake in [('real', False), ('Real', False), ('REAL', False), ('0', False), ('real_images', False), 
                                    ('fake', True), ('Fake', True), ('FAKE', True), ('1', True), ('ai_images', True)]:
            for split_dir in [IMAGE_RAW_DIR, os.path.join(IMAGE_RAW_DIR, "my_real_vs_ai_dataset")]:
                class_dir = os.path.join(split_dir, class_name)
                if os.path.exists(class_dir):
                    count = 0
                    for root, _, files in os.walk(class_dir):
                        for f in files:
                            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                                all_paths.append(os.path.join(root, f))
                                all_labels.append(1 if is_fake else 0)
                                count += 1
                    if count > 0:
                        print(f"Found {count} images in {class_dir}")
    
    print(f"\nTotal images successfully loaded: {len(all_paths)}")
    if len(all_paths) > 0:
        unique, counts = np.unique(all_labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Label {u} ({'Real' if u == 0 else 'Fake'}): {c}")
    
    return all_paths, all_labels


class ImageDataGenerator(keras.utils.Sequence):
    """Custom data generator for loading images in batches."""
    
    def __init__(self, paths, labels, batch_size=32, target_size=(224, 224),
                 augment=False, shuffle=True):
        self.paths = np.array(paths)
        self.labels = np.array(labels, dtype=np.float32)
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.paths))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return max(1, len(self.paths) // self.batch_size)
    
    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = self.paths[batch_idx]
        batch_labels = self.labels[batch_idx]
        
        images = []
        valid_labels = []
        
        for path, label in zip(batch_paths, batch_labels):
            try:
                img = keras.utils.load_img(path, target_size=self.target_size)
                arr = keras.utils.img_to_array(img) / 255.0
                
                if self.augment:
                    arr = self._augment(arr)
                
                images.append(arr)
                valid_labels.append(label)
            except Exception:
                continue
        
        if len(images) == 0:
            # Return a dummy batch
            images = [np.zeros((*self.target_size, 3), dtype=np.float32)]
            valid_labels = [0.0]
        
        return np.array(images), np.array(valid_labels)
    
    def _augment(self, img):
        """Apply random augmentations."""
        if np.random.random() > 0.5:
            img = np.fliplr(img)
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 1)
        return img
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def train():
    """Main training function."""
    print("=" * 60)
    print("  DeepfakeGuard — Image Model Training")
    print("=" * 60)
    
    # Load data
    paths, labels = load_image_paths_and_labels()
    
    if len(paths) == 0:
        print("\n[ERROR] No images found! Please ensure datasets are extracted.")
        print("Expected image data in:", IMAGE_RAW_DIR)
        return
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTrain: {len(train_paths)} | Val: {len(val_paths)}")
    
    # Create generators
    train_gen = ImageDataGenerator(
        train_paths, train_labels, batch_size=IMAGE_BATCH_SIZE,
        target_size=IMAGE_SIZE, augment=True, shuffle=True
    )
    val_gen = ImageDataGenerator(
        val_paths, val_labels, batch_size=IMAGE_BATCH_SIZE,
        target_size=IMAGE_SIZE, augment=False, shuffle=False
    )
    
    # Build model
    print("\nBuilding MobileNetV2 model...")
    model = build_image_model(
        input_shape=(*IMAGE_SIZE, 3),
        learning_rate=IMAGE_LEARNING_RATE
    )
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            IMAGE_MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        )
    ]
    
    # Phase 1: Train classification head (base frozen)
    print("\n--- Phase 1: Training classification head ---")
    history1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=IMAGE_EPOCHS, callbacks=callbacks, verbose=1
    )
    
    # Phase 2: Fine-tune top layers
    print("\n--- Phase 2: Fine-tuning top layers ---")
    model = unfreeze_and_finetune(model, learning_rate=1e-5, unfreeze_from=100)
    
    history2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=5, callbacks=callbacks, verbose=1
    )
    
    # Save final model
    model.save(IMAGE_MODEL_PATH)
    print(f"\nModel saved to: {IMAGE_MODEL_PATH}")
    
    # Evaluate
    print("\n--- Evaluation ---")
    val_gen_eval = ImageDataGenerator(
        val_paths, val_labels, batch_size=IMAGE_BATCH_SIZE,
        target_size=IMAGE_SIZE, augment=False, shuffle=False
    )
    
    y_pred_proba = []
    y_true = []
    for i in range(len(val_gen_eval)):
        X, y = val_gen_eval[i]
        pred = model.predict(X, verbose=0)
        y_pred_proba.extend(pred.flatten().tolist())
        y_true.extend(y.flatten().tolist())
    
    y_pred = [1 if p > 0.5 else 0 for p in y_pred_proba]
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # Save metrics
    metrics = {
        "val_accuracy": float(max(history1.history.get("val_accuracy", [0]))),
        "val_auc": float(max(history1.history.get("val_auc", [0]))),
        "total_images": len(paths),
        "train_size": len(train_paths),
        "val_size": len(val_paths)
    }
    
    metrics_path = os.path.join(MODELS_DIR, "image_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    train()
