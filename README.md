# 🛡️ VulnurisGuard: Multimodal Deepfake Detection System

![VulnurisGuard Banner](https://img.shields.io/badge/Status-Hackathon_Ready-success?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-success?style=for-the-badge) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange?style=for-the-badge)

**VulnurisGuard** is a state-of-the-art, multimodal deepfake detection application built for the **Vulnuris Hackathon**. It analyzes **images, audio, and video** to determine if the media has been AI-generated or manipulated, empowering users to verify the authenticity of digital content.

---

## ✨ Key Features

- **🖼️ Image Detection**: Utilizes a fine-tuned MobileNetV2 architecture to detect AI-generated artifacts and deepfake manipulations in static images.
- **🎙️ Audio Detection**: Converts audio waveforms into Mel-Spectrograms and analyzes them using a custom Convolutional Neural Network (CNN) to detect voice cloning and synthetic speech.
- **🎥 Video Detection**: Employs temporal frame extraction and aggregation, evaluating individual frames to provide a cohesive confidence score for videos.
- **⚡ High-Performance API**: A fully decoupled, asynchronous `FastAPI` backend for rapid, scalable inference.
- **🎨 Modern Web Interface**: A beautiful, bespoke frontend featuring dark mode aesthetics, glassmorphism, dynamic gauge charts, and drag-and-drop file support.

---

## 🏗️ System Architecture

Our solution is divided into two decoupled components: a robust AI back-end and an interactive front-end.

1. **AI Processing Engine**: 
   - `models/`: Contains the Keras/TensorFlow architectures for image and audio classification.
   - `detector/pipeline.py`: A unified class that orchestrates preprocessing, feature extraction (e.g., Librosa for audio), and model inference.
2. **API Layer (`api.py`)**: 
   - Receives client uploads, temporarily stores files, runs the orchestration pipeline, and returns JSON-formatted predictions and confidence scores.
3. **Frontend Presentation**: 
   - A static HTML/CSS/JS single-page application that interacts dynamically with the API without requiring page reloads.

*(See `docs/system_architecture.md` for detailed Mermaid diagrams).*

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+ 
- Node/Live Server (optional, for frontend hosting)

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone <your-repo-url>
cd VulnurisGuard
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

### 2. Running the Application

You must run the backend and frontend simultaneously.

**Start the Backend (FastAPI):**
```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

**Open the Frontend:**
Simply navigate to `frontend/index.html` and open it in any modern web browser.
*(Tip: You can use VS Code's "Live Server" extension for the best experience).*

---

## 🧪 Testing the Models

Dataset structure should be mapped via the `scripts/prepare_data.py`. To retrain the models:

1. **Audio Model**: Run `python scripts/train_audio_model.py`
2. **Image Model**: Run `python scripts/train_image_model.py`

*Note: Pre-trained models are saved as `.h5` files in the `models/` directory.*

---

## 🎯 Impact & Future Roadmap

**The Problem:** Deepfakes threaten organizational security, personal identity, and public trust. 
**Our Solution:** An accessible, multimodal verification tool.

**Future Enhancements:**
- Live webcam and microphone streaming for real-time inference.
- Grad-CAM heatmap visualizations to highlight exactly *where* an image was manipulated.
- Integration with blockchain ledgers for immutable media provenance scoring.

---

Built with ❤️ by the VulnurisGuard Team.
