# VulnurisGuard System Architecture

The following diagrams illustrate the internal architecture and data flow of the VulnurisGuard Multimodal Deepfake Detection System.

## 1. High-Level Architecture Overview

This diagram shows how the decoupled Frontend and Backend communicate, and how the backend handles different modalities.

```mermaid
graph TD
    %% Entities
    User((User))
    Browser[Web Browser\nFrontend UI]
    FastAPI[FastAPI\nBackend Server]
    Pipeline[Detector Pipeline\nOrchestrator]
    
    %% Models
    ImgModel[(MobileNetV2\nImage Model)]
    AudioModel[(CNN Spectrogram\nAudio Model)]
    VidLogic[Frame Extractor\n+ Temporal Aggregator]

    %% Flow
    User -->|Drag & Drop Media| Browser
    Browser -->|POST /api/analyze\nmultipart/form-data| FastAPI
    
    FastAPI -->|Save Temp File| TempVol[(Temp Storage)]
    FastAPI -->|Route File| Pipeline
    
    Pipeline -->|If Image| ImgModel
    Pipeline -->|If Audio| AudioModel
    Pipeline -->|If Video| VidLogic
    
    VidLogic -->|Extract Frames| ImgModel
    
    ImgModel -->|Confidence Score| Pipeline
    AudioModel -->|Confidence Score| Pipeline
    VidLogic -->|Averaged Score| Pipeline
    
    Pipeline -->|Format JSON Result| FastAPI
    FastAPI -->|JSON Response| Browser
    Browser -->|Animate Gauge Chart| User
    
    %% Styling
    classDef frontend fill:#1e1e2e,stroke:#58a6ff,stroke-width:2px,color:#fff
    classDef backend fill:#2ea043,stroke:#1e1e2e,stroke-width:2px,color:#fff
    classDef model fill:#8a2be2,stroke:#1e1e2e,stroke-width:2px,color:#fff
    
    class Browser frontend
    class FastAPI backend
    class Pipeline backend
    class ImgModel,AudioModel model
```

## 2. Audio Processing Pipeline

Detailing how audio files are translated into image-like formats for Deep Learning classification.

```mermaid
sequenceDiagram
    participant API as FastAPI
    participant Pre as Preprocessing Util
    participant Lib as Librosa
    participant CNN as Audio CNN Model
    
    API->>Pre: raw_audio.wav
    Pre->>Lib: load(audio, sr=22050)
    Lib-->>Pre: waveform_data
    Pre->>Lib: extract_mel_spectrogram(waveform)
    Lib-->>Pre: mel_spectrogram_matrix
    Pre->>Pre: Resize to (128, 128, 1) & Normalize
    Pre->>CNN: predict(preprocessed_tensor)
    CNN-->>API: Probability % (Real vs Fake)
```

## 3. Video Processing Pipeline (Temporal Aggregation)

Detailing how videos are split into frames to detect temporal anomalies or frame-specific manipulation.

```mermaid
flowchart LR
    Video[MP4 Video\nUpload] --> CV[OpenCV\nVideoCapture]
    CV --> |Extract| F1[Frame 1]
    CV --> |Extract| F2[Frame 2]
    CV --> |...| FN[Frame N]
    
    F1 --> Model(Image Model)
    F2 --> Model
    FN --> Model
    
    Model --> |Score 1| Agg[Aggregator]
    Model --> |Score 2| Agg
    Model --> |Score N| Agg
    
    Agg --> |Average / Voting| Final[Final Global\nVideo Verdict]
```

## 4. Technology Stack

- **Frontend Core**: HTML5, Vanilla JavaScript, CSS3
- **Frontend Design**: CSS Glassmorphism, FontAwesome, Google Fonts
- **Backend Framework**: Python, FastAPI, Uvicorn (ASGI)
- **Machine Learning**: TensorFlow 2.x, Keras
- **Data Processing**: NumPy, OpenCV (cv2), Librosa, Pandas
