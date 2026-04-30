"""
DeepfakeGuard — Central Configuration
"""
import os

# ─── Base Paths ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# Raw extracted data
AUDIO_RAW_DIR = os.path.join(DATA_DIR, "audio_raw")
VIDEO_RAW_DIR = os.path.join(DATA_DIR, "video_raw")
IMAGE_RAW_DIR = os.path.join(DATA_DIR, "image_raw")

# CSV label files
TEST_LABELS_CSV = os.path.join(BASE_DIR, "test_labels.csv")
VAL_LABELS_CSV = os.path.join(BASE_DIR, "val_labels.csv")
REALGUARD_CSV = os.path.join(BASE_DIR, "realguard_2025_dataset.csv")

# ─── Image Model Config ──────────────────────────────────────
IMAGE_SIZE = (224, 224)
IMAGE_BATCH_SIZE = 32
IMAGE_EPOCHS = 10
IMAGE_LEARNING_RATE = 1e-4
IMAGE_MODEL_PATH = os.path.join(MODELS_DIR, "image_detector.keras")

# ─── Audio Model Config ──────────────────────────────────────
AUDIO_SAMPLE_RATE = 22050
AUDIO_DURATION = 5  # seconds
N_MELS = 128
HOP_LENGTH = 512
AUDIO_BATCH_SIZE = 32
AUDIO_EPOCHS = 15
AUDIO_LEARNING_RATE = 1e-4
AUDIO_MODEL_PATH = os.path.join(MODELS_DIR, "audio_detector.keras")

# ─── Video Model Config ──────────────────────────────────────
VIDEO_FRAMES_TO_SAMPLE = 10
VIDEO_MODEL_PATH = IMAGE_MODEL_PATH  # Reuses image model

# ─── Detection Thresholds ────────────────────────────────────
FAKE_THRESHOLD = 0.5  # Above this = fake

# Ensure output dirs exist
os.makedirs(MODELS_DIR, exist_ok=True)
