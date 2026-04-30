"""
Video Deepfake Detector — Frame extraction + image model + temporal aggregation.
"""
import numpy as np
import cv2


class VideoDetector:
    """Detect deepfakes in videos using frame-level image analysis."""
    
    def __init__(self, image_model, target_size=(224, 224), num_frames=10):
        self.image_model = image_model
        self.target_size = target_size
        self.num_frames = num_frames
    
    def extract_frames(self, video_path):
        """Extract evenly-spaced frames from video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError(f"No frames in video: {video_path}")
        
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, self.target_size)
                frames.append(frame_resized.astype(np.float32) / 255.0)
        
        cap.release()
        return np.array(frames)
    
    def predict(self, video_path):
        """
        Predict whether a video is deepfake.
        Returns dict with prediction, confidence, and per-frame scores.
        """
        frames = self.extract_frames(video_path)
        
        if len(frames) == 0:
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "frame_scores": [],
                "error": "No frames could be extracted"
            }
        
        # Get per-frame predictions
        frame_scores = self.image_model.predict(frames, verbose=0).flatten()
        
        # Temporal aggregation: average confidence
        avg_score = float(np.mean(frame_scores))
        
        return {
            "prediction": "FAKE" if avg_score > 0.5 else "REAL",
            "confidence": avg_score if avg_score > 0.5 else (1 - avg_score),
            "fake_probability": avg_score,
            "frame_scores": frame_scores.tolist(),
            "num_frames_analyzed": len(frames)
        }
