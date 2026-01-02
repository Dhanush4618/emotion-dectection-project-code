from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import base64
import cv2
import numpy as np
import onnxruntime as ort
try:
    import mediapipe as mp
except Exception:
    mp = None
    # defer error handling to runtime; application can run in demo mode without MediaPipe
import json
import logging
from typing import List, Dict, Any
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Emotion Detection API",
    description="Real-time emotion detection using computer vision and deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe face detection if available
mp_face_detection = None
if mp is not None:
    try:
        mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5
        )
    except Exception as e:
        logger.warning(f"MediaPipe initialization failed: {e}")
        mp_face_detection = None

# Initialize ONNX runtime session
try:
    session = ort.InferenceSession("models/emotion_model.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logger.info("ONNX model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load ONNX model: {e}")
    logger.info("Running in demo mode with dummy predictions")
    session = None
    input_name = None
    output_name = None

# Emotion labels
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

class EmotionDetector:
    def __init__(self):
        # face_detection may be None if mediapipe is not installed or failed to initialize
        self.face_detection = mp_face_detection
        self.session = session
        self.input_name = input_name
        # ensure detector has the output_name if model is loaded
        self.output_name = output_name
        self.emotion_labels = EMOTION_LABELS

        # Cache model input shape (canonicalized ints) for preprocessing
        try:
            if self.session is not None:
                raw_shape = self.session.get_inputs()[0].shape
                # Convert unknowns (strings) to 1
                self.model_input_shape = [int(s) if isinstance(s, int) or (isinstance(s, str) and s.isdigit()) else 1 for s in raw_shape]
            else:
                self.model_input_shape = None
        except Exception:
            self.model_input_shape = None

    def _is_nchw(self):
        s = self.model_input_shape
        if not s or len(s) != 4:
            return True
        # heuristic: second dim 1 or 3 likely channels in NCHW
        return s[1] in (1, 3)
    
    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face image for model input"""
        # If we know the model input shape, adapt preprocessing accordingly
        if self.model_input_shape and len(self.model_input_shape) == 4:
            # model shape is likely [N,H,W,C] or [N,C,H,W]
            if self._is_nchw():
                _, c, h, w = self.model_input_shape
                h = int(h) if h else 224
                w = int(w) if w else 224
                face_resized = cv2.resize(face_img, (w, h))
                face_normalized = face_resized.astype(np.float32) / 255.0
                face_tensor = np.transpose(face_normalized, (2, 0, 1))
                face_batch = np.expand_dims(face_tensor, axis=0)
                return face_batch
            else:
                _, h, w, c = self.model_input_shape
                h = int(h) if h else 48
                w = int(w) if w else 48
                face_resized = cv2.resize(face_img, (w, h))
                # If single-channel expected, convert
                if int(c) == 1:
                    face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                    face_resized = np.expand_dims(face_resized, axis=-1)
                face_normalized = face_resized.astype(np.float32) / 255.0
                face_batch = np.expand_dims(face_normalized, axis=0)  # NHWC
                return face_batch

        # Fallback: Resize to 224x224 NCHW
        face_resized = cv2.resize(face_img, (224, 224))
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_tensor = np.transpose(face_normalized, (2, 0, 1))
        face_batch = np.expand_dims(face_tensor, axis=0)
        return face_batch
    
    def detect_emotion(self, face_img: np.ndarray) -> Dict[str, Any]:
        """Detect emotion in face image"""
        if self.session is None:
            # Return dummy predictions for demo mode
            import random
            dummy_predictions = [random.random() for _ in range(7)]
            # Normalize to sum to 1
            total = sum(dummy_predictions)
            dummy_predictions = [p/total for p in dummy_predictions]
            emotion_idx = np.argmax(dummy_predictions)
            confidence = float(dummy_predictions[emotion_idx])
            emotion = self.emotion_labels[emotion_idx]
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "predictions": dummy_predictions,
                "inference_time_ms": 50.0,  # Simulated inference time
                "demo_mode": True
            }
        
        try:
            # Preprocess face
            input_tensor = self.preprocess_face(face_img)
            
            # Run inference
            start_time = time.time()
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            inference_time = time.time() - start_time
            
            # Get predictions
            predictions = outputs[0][0]
            emotion_idx = np.argmax(predictions)
            confidence = float(predictions[emotion_idx])
            emotion = self.emotion_labels[emotion_idx]
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "predictions": predictions.tolist(),
                "inference_time_ms": round(inference_time * 1000, 2)
            }
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            return {"error": str(e)}
    
    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process frame and return emotion detections for all faces"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # If MediaPipe is not available, return an empty detection list
        if self.face_detection is None:
            return []

        # Detect faces
        results = self.face_detection.process(rgb_frame)

        detections = []
        if results and getattr(results, "detections", None):
            height, width = frame.shape[:2]

            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)

                if w > 0 and h > 0:
                    # Extract face region
                    face_img = rgb_frame[y:y+h, x:x+w]

                    # Detect emotion
                    emotion_result = self.detect_emotion(face_img)

                    detection_result = {
                        "bbox": [x, y, x + w, y + h],
                        "emotion": emotion_result
                    }
                    detections.append(detection_result)

        return detections

# Initialize emotion detector
emotion_detector = EmotionDetector()

@app.get("/")
async def root():
    """Serve the main web interface"""
    return FileResponse("web-client/index.html")

@app.get("/app.js")
async def get_app_js():
    """Serve the JavaScript file"""
    return FileResponse("web-client/app.js")

@app.get("/api")
async def api_root():
    """API root endpoint"""
    return {
        "message": "AI Emotion Detection API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": session is not None,
        "timestamp": time.time()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time emotion detection"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive base64 encoded image
            data = await websocket.receive_text()
            
            try:
                # Parse base64 image
                if "," in data:
                    header, b64_data = data.split(",", 1)
                else:
                    b64_data = data
                
                # Decode base64 to image
                img_bytes = base64.b64decode(b64_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({"error": "Invalid image data"})
                    continue
                
                # Process frame for emotion detection
                detections = emotion_detector.process_frame(frame)
                
                # Send results back to client
                response = {
                    "detections": detections,
                    "timestamp": time.time()
                }
                
                await websocket.send_json(response)
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 