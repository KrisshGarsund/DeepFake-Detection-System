"""
Preprocessing utilities for images, audio, and video frames.
"""
import numpy as np
import cv2
from PIL import Image
import os

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image for model input."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

def preprocess_image_array(img_array, target_size=(224, 224)):
    """Preprocess a numpy image array (BGR from OpenCV)."""
    if img_array is None:
        return None
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    return img_resized.astype(np.float32) / 255.0

def extract_frames(video_path, num_frames=10):
    """Extract evenly-spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames

def extract_mel_spectrogram(audio_path, sr=22050, duration=5, n_mels=128, hop_length=512):
    """Extract mel spectrogram from an audio file."""
    import librosa
    
    # Load audio
    y, sr_actual = librosa.load(audio_path, sr=sr, duration=duration)
    
    # Pad if shorter than expected
    expected_length = sr * duration
    if len(y) < expected_length:
        y = np.pad(y, (0, expected_length - len(y)), mode='constant')
    else:
        y = y[:expected_length]
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_db

def detect_faces(image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """Detect faces in an image using OpenCV Haar cascade."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
    )
    return faces

def crop_face(image, face_rect, margin=0.2):
    """Crop a face from an image with optional margin."""
    x, y, w, h = face_rect
    img_h, img_w = image.shape[:2]
    
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(img_w, x + w + margin_w)
    y2 = min(img_h, y + h + margin_h)
    
    return image[y1:y2, x1:x2]
