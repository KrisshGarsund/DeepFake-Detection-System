"""
Unified Deepfake Detection Pipeline.
Combines image, audio, and video detection into a single API.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    IMAGE_MODEL_PATH, AUDIO_MODEL_PATH, IMAGE_SIZE,
    AUDIO_SAMPLE_RATE, AUDIO_DURATION, N_MELS, HOP_LENGTH,
    VIDEO_FRAMES_TO_SAMPLE, FAKE_THRESHOLD
)
from utils.preprocessing import (
    preprocess_image, extract_mel_spectrogram, extract_frames,
    preprocess_image_array
)
from models.video_detector import VideoDetector


# Supported file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a'}


class DeepfakeDetector:
    """Multimodal deepfake detection pipeline."""
    
    def __init__(self):
        self.image_model = None
        self.audio_model = None
        self.video_detector = None
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models."""
        # Image model
        if os.path.exists(IMAGE_MODEL_PATH):
            print("[✓] Loading image detection model...")
            self.image_model = keras.models.load_model(IMAGE_MODEL_PATH)
            self.video_detector = VideoDetector(
                self.image_model,
                target_size=IMAGE_SIZE,
                num_frames=VIDEO_FRAMES_TO_SAMPLE
            )
        else:
            print("[✗] Image model not found. Run train_image_model.py first.")
        
        # Audio model
        if os.path.exists(AUDIO_MODEL_PATH):
            print("[✓] Loading audio detection model...")
            self.audio_model = keras.models.load_model(AUDIO_MODEL_PATH)
        else:
            print("[✗] Audio model not found. Run train_audio_model.py first.")
    
    def detect(self, file_path):
        """
        Auto-detect file type and run appropriate detection.
        Returns a result dict with prediction, confidence, modality, and explanation.
        """
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in IMAGE_EXTENSIONS:
            return self.detect_image(file_path)
        elif ext in VIDEO_EXTENSIONS:
            return self.detect_video(file_path)
        elif ext in AUDIO_EXTENSIONS:
            return self.detect_audio(file_path)
        else:
            return {"error": f"Unsupported file type: {ext}"}
    
    def detect_image(self, image_path):
        """Detect deepfake in a single image."""
        if self.image_model is None:
            return {"error": "Image model not loaded"}
        
        try:
            img = preprocess_image(image_path, target_size=IMAGE_SIZE)
            img_batch = np.expand_dims(img, axis=0)
            
            score = float(self.image_model.predict(img_batch, verbose=0)[0][0])
            is_fake = score > FAKE_THRESHOLD
            
            return {
                "file": os.path.basename(image_path),
                "modality": "image",
                "prediction": "FAKE" if is_fake else "REAL",
                "confidence": score if is_fake else (1 - score),
                "fake_probability": score,
                "explanation": self._image_explanation(score)
            }
        except Exception as e:
            return {"error": str(e), "file": os.path.basename(image_path)}
    
    def detect_video(self, video_path):
        """Detect deepfake in a video file."""
        if self.video_detector is None:
            return {"error": "Video model not loaded"}
        
        try:
            result = self.video_detector.predict(video_path)
            result["file"] = os.path.basename(video_path)
            result["modality"] = "video"
            result["explanation"] = self._video_explanation(result)
            return result
        except Exception as e:
            return {"error": str(e), "file": os.path.basename(video_path)}
    
    def detect_audio(self, audio_path):
        """Detect deepfake in an audio file."""
        if self.audio_model is None:
            return {"error": "Audio model not loaded"}
        
        try:
            mel = extract_mel_spectrogram(
                audio_path, sr=AUDIO_SAMPLE_RATE,
                duration=AUDIO_DURATION, n_mels=N_MELS,
                hop_length=HOP_LENGTH
            )
            mel_input = mel[np.newaxis, ..., np.newaxis]  # (1, n_mels, time, 1)
            
            score = float(self.audio_model.predict(mel_input, verbose=0)[0][0])
            is_fake = score > FAKE_THRESHOLD
            
            return {
                "file": os.path.basename(audio_path),
                "modality": "audio",
                "prediction": "FAKE" if is_fake else "REAL",
                "confidence": score if is_fake else (1 - score),
                "fake_probability": score,
                "explanation": self._audio_explanation(score)
            }
        except Exception as e:
            return {"error": str(e), "file": os.path.basename(audio_path)}
    
    def _image_explanation(self, score):
        """Generate human-readable explanation for image detection."""
        if score > 0.9:
            return "High confidence: Strong indicators of AI generation or manipulation detected in facial features, texture, and artifacts."
        elif score > 0.7:
            return "Moderate confidence: Some anomalies detected in image consistency and facial features."
        elif score > 0.5:
            return "Low confidence: Subtle irregularities detected. Manual review recommended."
        elif score > 0.3:
            return "Likely authentic: Minor anomalies typical of normal image compression."
        else:
            return "High confidence authentic: No significant indicators of manipulation detected."
    
    def _video_explanation(self, result):
        """Generate explanation for video detection."""
        scores = result.get("frame_scores", [])
        if not scores:
            return "Unable to analyze video frames."
        
        high_score_frames = sum(1 for s in scores if s > 0.5)
        total = len(scores)
        
        if high_score_frames > total * 0.7:
            return f"Strong deepfake indicators in {high_score_frames}/{total} analyzed frames. Consistent manipulation detected across the video."
        elif high_score_frames > total * 0.3:
            return f"Partial manipulation detected in {high_score_frames}/{total} frames. Possible splicing or intermittent deepfake content."
        else:
            return f"Only {high_score_frames}/{total} frames show anomalies. Video appears largely authentic."
    
    def _audio_explanation(self, score):
        """Generate explanation for audio detection."""
        if score > 0.9:
            return "High confidence: Audio exhibits strong characteristics of synthetic speech (TTS/voice cloning artifacts)."
        elif score > 0.7:
            return "Moderate confidence: Voice patterns suggest possible synthetic generation."
        elif score > 0.5:
            return "Low confidence: Some synthetic characteristics detected. Speaker verification recommended."
        elif score > 0.3:
            return "Likely authentic: Minor spectral anomalies within normal range."
        else:
            return "High confidence authentic: Natural speech patterns with no synthetic indicators."
