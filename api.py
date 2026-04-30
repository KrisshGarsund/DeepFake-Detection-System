import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from detector.pipeline import DeepfakeDetector
import uvicorn

app = FastAPI(title="Deepfake Detection API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize detector (lazy load or load on startup)
detector = None

@app.on_event("startup")
async def startup_event():
    global detector
    print("Initializing DeepfakeDetector...")
    try:
        detector = DeepfakeDetector()
        print("Detector initialized successfully.")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        # Not failing hard here so the API still loads, but endpoints will fail

os.makedirs("data/uploads", exist_ok=True)

@app.post("/api/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """
    Endpoint to analyze a file (image, video, or audio) for deepfakes.
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector models are not loaded.")

    try:
        # Save uploaded file temporarily
        file_path = os.path.join("data/uploads", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Detect
        result = detector.detect(file_path)

        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)

        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
