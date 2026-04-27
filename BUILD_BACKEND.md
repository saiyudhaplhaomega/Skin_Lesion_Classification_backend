# Backend Build Guide: XAI Skin Lesion Analysis Platform

**Step-by-step instructions to build the FastAPI backend from absolute zero. Every command included, every concept explained.**

---

## How This Guide Relates To The Root Docs

This file is the beginner-friendly backend tutorial. Keep it in the backend repo because it teaches the first files to create and the basic FastAPI concepts.

For production decisions, also read these root guides:

- `../docs/BUILD_PHASE_2_BACKEND.md` - production backend sequence
- `../docs/SYSTEM_DESIGN_LEARNING_GUIDE.md` - why FastAPI, Redis, S3, SQS, RDS, and MLflow are used
- `../docs/PRODUCTION_BUILD_REVIEW.md` - current implementation gaps and production blockers

If this file and the root docs disagree, treat the root docs as the production source of truth and update this file.

## Current Reality

The backend repo currently contains ML helper code, model artifacts, dependency files, and Makefile targets. The FastAPI app still needs to be built.

Research notebooks, RQ1-RQ6 experiments, paper figures, and research metrics now live in `../Skin_Lesion_XAI_research`. Do not add or document notebook workflows as backend-owned work.

Start with:

1. `GET /health`
2. mocked `POST /api/v1/predict`
3. file validation tests
4. real model loading
5. Redis-backed `/explain`

Model artifacts produced by research should be handed to the backend through `ml/outputs/models/` or through the future MLflow/S3 model registry. The backend should serve models; the research repo should train and evaluate them.

Do not start by building every endpoint at once.

## Dependency Note

The dependency snippet below is for learning. Before installing, compare it to the real `requirements.txt` in this repo. The current production baseline uses newer FastAPI/PyTorch packages and should stay the source of truth.

For backend Docker images, prefer `opencv-python-headless` instead of GUI OpenCV packages.

---

## Table of Contents

1. [What is a Backend?](#1-what-is-a-backend)
2. [Understanding FastAPI](#2-understanding-fastapi)
3. [Step 1: Project Setup](#step-1-project-setup)
4. [Step 2: Create the App Structure](#step-2-create-the-app-structure)
5. [Step 3: Implement Health Endpoint](#step-3-implement-health-endpoint)
6. [Step 4: Implement Predict Endpoint](#step-4-implement-predict-endpoint)
7. [Step 5: Implement Explain Endpoint](#step-5-implement-explain-endpoint)
8. [Step 6: Implement Feedback Endpoint](#step-6-implement-feedback-endpoint)
9. [Step 7: Add Model Loading](#step-7-add-model-loading)
10. [Step 8: Dockerize the Backend](#step-8-dockerize-the-backend)
11. [Step 9: Run and Test](#step-9-run-and-test)

---

## 1. What is a Backend?

A backend is a server that runs on a computer (often in the cloud) and handles:
- Receiving requests from the frontend (like "here's an image, classify it")
- Processing those requests (running AI models, saving data)
- Sending responses back to the frontend (like "malignant, 87% confidence")

Think of a restaurant:
- **Frontend** = the dining room where customers sit (what they see)
- **Backend** = the kitchen where food is prepared (hidden from customers)
- **API** = the menu and ordering system between them

### Why Do We Need a Backend?

The AI model that classifies skin lesions runs in Python with PyTorch. Browsers can't run Python directly. So we need a server that:
1. Receives the image from the browser
2. Runs the Python/AI code to analyze it
3. Sends the results back to the browser

---

## 2. Understanding FastAPI

### What is FastAPI?

FastAPI is a Python framework for building web APIs. It's fast, easy to use, and automatically generates documentation.

### Why FastAPI?

1. **Fast** - One of the fastest Python web frameworks
2. **Easy** - Simple syntax that feels like writing Python, not Java/C#
3. **Automatic docs** - Visit `/docs` in your browser to see interactive API documentation
4. **Type safety** - Pydantic models validate data automatically

### What is an Endpoint?

An endpoint is a URL that the frontend can call to trigger some action. Think of it like a specific order on a menu:

- `POST /api/v1/predict` = "I want you to analyze this image"
- `POST /api/v1/explain` = "I want you to generate a heatmap for this prediction"
- `POST /api/v1/feedback` = "I want to submit consent feedback"

### What is a Router?

A router groups related endpoints together. Instead of putting all endpoints in one file, we organize them:

- `app/routers/predict.py` - handles prediction endpoints
- `app/routers/explain.py` - handles explanation endpoints
- `app/routers/feedback.py` - handles feedback endpoints

This makes the code easier to navigate, like organizing a menu by course (appetizers, main dishes, desserts).

---

## Step 1: Project Setup

### Navigate to the Backend Directory

```bash
cd C:/Users/saiyu/Desktop/projects/KI_projects/Skin_Lesion_GRADCAM_Classification/Skin_Lesion_Classification_backend
```

### Install Dependencies

Create `requirements.txt` with all the Python packages we need:

```txt
# Web framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6

# Data validation
pydantic==2.5.3
pydantic-settings==2.1.0

# ML / AI
torch==2.1.2
torchvision==0.16.2
timm==0.9.16
pytorch-gradcam==1.5.0

# AWS / Storage
boto3==1.34.14

# Testing
pytest==7.4.4
pytest-asyncio==0.23.3
httpx==0.26.0
```

### Create a Virtual Environment

A virtual environment is an isolated Python environment where we install packages. This prevents conflicts between different projects.

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Verify activation - you should see (venv) at the start of your prompt
which python
```

### Install the Packages

```bash
pip install -r requirements.txt
```

This may take 5-10 minutes as it downloads PyTorch and other large packages.

---

## Step 2: Create the App Structure

### Why This Structure?

```
app/
├── __init__.py
├── main.py          ← Entry point, FastAPI app
├── config.py        ← Settings (env vars, etc.)
├── routers/         ← API route handlers
│   ├── __init__.py
│   ├── predict.py
│   ├── explain.py
│   ├── feedback.py
│   └── health.py
├── models/          ← Pydantic models (data validation)
│   ├── __init__.py
│   └── schemas.py
└── ml/              ← ML code (model loading, CAM)
    ├── __init__.py
    └── model_loader.py
```

Each folder has an `__init__.py` file which tells Python "this folder is a package." This allows us to do `from app.routers import predict`.

### Create All Directories and Files

```bash
mkdir -p app/routers
mkdir -p app/models
mkdir -p app/ml
touch app/__init__.py
touch app/routers/__init__.py
touch app/models/__init__.py
touch app/ml/__init__.py
```

### Create the Main App File

`app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict, explain, feedback, health

# Create FastAPI app
app = FastAPI(
    title="Skin Lesion Analysis API",
    description="AI-powered skin lesion classification with Grad-CAM explainability",
    version="1.0.0",
)

# CORS - allow frontend to call this API
# Without CORS, the browser would block requests from localhost:3000 to localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers (groupings of endpoints)
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(predict.router, prefix="/api/v1", tags=["predict"])
app.include_router(explain.router, prefix="/api/v1", tags=["explain"])
app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])

# Startup event - runs when the server starts
@app.on_event("startup")
async def startup_event():
    print("Starting up Skin Lesion Analysis API...")
    # We will load the ML model here in Step 7

# Root endpoint - basic check
@app.get("/")
async def root():
    return {"message": "Skin Lesion Analysis API", "version": "1.0.0"}
```

### What Just Happened?

1. We created a FastAPI application with a title and description
2. We added CORS middleware so the frontend can talk to this backend
3. We registered all our routers with a `/api/v1` prefix
4. We added a startup event (runs once when server starts)

### Create Configuration

`app/config.py`:

```python
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # MLflow configuration
    mlflow_tracking_uri: str = "file:./mlruns"
    model_name: str = "skin-lesion"
    base_model_arch: str = "resnet50"

    # AWS S3 configuration
    feedback_bucket: str = "skin-lesion-feedback"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "eu-central-1"

    # CORS
    allowed_origins: List[str] = ["http://localhost:3000"]

    # Model paths
    model_dir: str = "./ml/outputs/models"
    fallback_model_path: str = "./ml/outputs/models/resnet50_best.pth"

    # In-memory store TTL in seconds (1 hour)
    predictions_ttl_seconds: int = 3600

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
```

### What is Pydantic?

Pydantic is a data validation library. `BaseSettings` automatically reads environment variables and validates types. If you set `mlflow_tracking_uri="not_a_url"`, it will warn you at startup instead of failing mysteriously later.

---

## Step 3: Implement Health Endpoint

### What is a Health Check?

Before the frontend talks to the backend, it checks if the backend is alive. This is the health endpoint - it responds with basic info about the server and loaded model.

### Create the Health Router

`app/routers/health.py`:

```python
from fastapi import APIRouter, HTTPException
from app.models.schemas import HealthResponse, MethodsResponse

router = APIRouter()

# Available CAM methods - these are defined once and returned by the API
AVAILABLE_METHODS = ["gradcam", "gradcam_pp", "eigencam", "layercam"]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns model version and device info.
    """
    # This will be populated when we add model loading in Step 7
    # For now, return placeholder values
    return HealthResponse(
        model_version="not_loaded",
        device="cpu",
        status="healthy"
    )


@router.get("/methods", response_model=MethodsResponse)
async def get_methods():
    """
    Returns list of available XAI methods.
    """
    return MethodsResponse(methods=AVAILABLE_METHODS)
```

### Create the Schemas

`app/models/schemas.py`:

```python
from pydantic import BaseModel, Field
from typing import List


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    model_version: str
    device: str
    status: str = "healthy"


class MethodsResponse(BaseModel):
    """Response for available XAI methods."""
    methods: List[str]


class PredictionResponse(BaseModel):
    """Response for prediction endpoint."""
    prediction_id: str
    diagnosis: str = Field(pattern="^(benign|malignant)$")
    confidence: float = Field(ge=0, le=1)
    class_probabilities: dict
    model_version: str
    processing_time_ms: int


class ExplainRequest(BaseModel):
    """Request for explanation endpoint."""
    prediction_id: str
    method: str = Field(pattern="^(gradcam|gradcam_pp|eigencam|layercam)$")


class ExplainResponse(BaseModel):
    """Response for explanation endpoint."""
    explanation_id: str
    method: str
    heatmaps: dict
    metrics: dict


class FeedbackRequest(BaseModel):
    """Request for feedback endpoint."""
    prediction_id: str
    consent: bool = Field(description="Must be explicitly true")
    user_label: str = Field(default=None, pattern="^(benign|malignant)$")


class FeedbackResponse(BaseModel):
    """Response for feedback endpoint."""
    feedback_id: str
    status: str
    message: str
```

### What Just Happened?

1. We created a router (`health.py`) that handles health-related endpoints
2. We defined Pydantic models (`schemas.py`) that validate all request/response data
3. The health endpoint returns info about the model and device
4. The methods endpoint returns which CAM algorithms are available

---

## Step 4: Implement Predict Endpoint

### What is the Predict Endpoint?

The predict endpoint receives an image file, runs it through the AI model, and returns the classification result (benign or malignant, with confidence).

### Create the Predictions Store

First, we need a place to store predictions temporarily. Since this is a prototype, we'll use an in-memory dictionary. Later, this could be Redis for production.

`app/ml/predictions_store.py`:

```python
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from app.config import settings


class PredictionsStore:
    """
    In-memory store for predictions.
    Maps prediction_id -> prediction data.
    Auto-expires entries after TTL.
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def create(
        self,
        diagnosis: str,
        confidence: float,
        class_probabilities: dict,
        model_version: str,
        processing_time_ms: int,
        image_bytes: bytes,
        original_filename: str,
    ) -> str:
        """Create a new prediction and return its ID."""
        prediction_id = str(uuid.uuid4())

        self._store[prediction_id] = {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "model_version": model_version,
            "processing_time_ms": processing_time_ms,
            "image_bytes": image_bytes,
            "original_filename": original_filename,
            "created_at": datetime.now(),
        }

        return prediction_id

    def get(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Get a prediction by ID, checking expiry."""
        if prediction_id not in self._store:
            return None

        entry = self._store[prediction_id]
        age = datetime.now() - entry["created_at"]

        if age > timedelta(seconds=settings.predictions_ttl_seconds):
            # Expired - remove it
            del self._store[prediction_id]
            return None

        return entry

    def delete(self, prediction_id: str) -> bool:
        """Delete a prediction (after feedback submitted)."""
        if prediction_id in self._store:
            del self._store[prediction_id]
            return True
        return False

    def count(self) -> int:
        """Return number of stored predictions."""
        return len(self._store)


# Global singleton instance
predictions_store = PredictionsStore()
```

### What is a Singleton?

The `predictions_store = PredictionsStore()` at the bottom is a global instance. This means there's exactly one store in the entire application, shared across all requests. This is the singleton pattern.

### Create the Predict Router

`app/routers/predict.py`:

```python
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.models.schemas import PredictionResponse
from app.ml.predictions_store import predictions_store
import time

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
):
    """
    Classify a skin lesion image.

    Receives: image file (JPG, PNG, WEBP - max 10MB)
    Returns: diagnosis (benign/malignant), confidence, probabilities
    """
    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image."
        )

    # Read image bytes
    image_bytes = await image.read()

    # Check file size (10MB max)
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB."
        )

    # Start timing
    start_time = time.time()

    # TODO: Run AI model inference here
    # For now, return mock data
    # In Step 7, we will load the real model

    processing_time_ms = int((time.time() - start_time) * 1000)

    # Mock prediction result (replace with real model in Step 7)
    diagnosis = "malignant"
    confidence = 0.87
    class_probabilities = {"benign": 0.13, "malignant": 0.87}
    model_version = "resnet50_v1.0"

    # Store prediction for later retrieval (e.g., for explain or feedback)
    prediction_id = predictions_store.create(
        diagnosis=diagnosis,
        confidence=confidence,
        class_probabilities=class_probabilities,
        model_version=model_version,
        processing_time_ms=processing_time_ms,
        image_bytes=image_bytes,
        original_filename=image.filename,
    )

    return PredictionResponse(
        prediction_id=prediction_id,
        diagnosis=diagnosis,
        confidence=confidence,
        class_probabilities=class_probabilities,
        model_version=model_version,
        processing_time_ms=processing_time_ms,
    )
```

### What Just Happened?

1. We created a predictions store to hold prediction data temporarily
2. We created a predict endpoint that accepts image uploads
3. The endpoint validates the file type and size
4. Currently returns mock data, but stores everything in the predictions store
5. In Step 7, we will add real model inference

---

## Step 5: Implement Explain Endpoint

### What is the Explain Endpoint?

After getting a prediction, the frontend can ask for an explanation. This generates a Grad-CAM heatmap showing which parts of the image influenced the AI's decision.

### Create the Explain Router

`app/routers/explain.py`:

```python
from fastapi import APIRouter, HTTPException
from app.models.schemas import ExplainRequest, ExplainResponse
from app.ml.predictions_store import predictions_store
import base64
import uuid

router = APIRouter()

# Map method names to pytorch-gradcam library method names
CAM_METHODS = {
    "gradcam": "gradcam",
    "gradcam_pp": "gradcam++",
    "eigencam": "eigencam",
    "layercam": "layercam",
}


@router.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """
    Generate XAI heatmap for a prediction.

    Receives: prediction_id, method (gradcam/gradcam_pp/eigencam/layercam)
    Returns: base64 encoded images (original, heatmap, overlay)
    """
    # Find the prediction
    prediction = predictions_store.get(request.prediction_id)

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found or expired. Please submit a new image."
        )

    # Validate method
    if request.method not in CAM_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid method. Choose from: {list(CAM_METHODS.keys())}"
        )

    # TODO: Generate real Grad-CAM heatmap here
    # For now, return mock data with placeholder base64 images

    explanation_id = str(uuid.uuid4())

    # Mock metrics (replace with real calculation in Step 7)
    focus_area_percentage = 0.23
    cam_max = 0.94
    cam_mean = 0.31

    # In a real implementation, we would:
    # 1. Load the image from prediction["image_bytes"]
    # 2. Preprocess it for the model
    # 3. Run the CAM method on the target layer
    # 4. Generate heatmap overlay
    # 5. Encode all three as base64 PNGs

    return ExplainResponse(
        explanation_id=explanation_id,
        method=request.method,
        heatmaps={
            "original": "REPLACE_WITH_REAL_BASE64",  # Placeholder
            "heatmap": "REPLACE_WITH_REAL_BASE64",   # Placeholder
            "overlay": "REPLACE_WITH_REAL_BASE64",  # Placeholder
        },
        metrics={
            "focus_area_percentage": focus_area_percentage,
            "cam_max": cam_max,
            "cam_mean": cam_mean,
        }
    )
```

### What Just Happened?

1. We created an explain endpoint that takes a prediction_id and method
2. It retrieves the stored prediction (which has the original image)
3. It validates the method is one we support
4. Currently returns mock data - in Step 7 we add real CAM generation

---

## Step 6: Implement Feedback Endpoint

### What is the Feedback Endpoint?

When a user opts to share their image for training, the feedback endpoint:
1. Validates consent is explicitly true
2. Gets the image from the predictions store
3. Uploads to S3 with metadata
4. Removes from local store (one feedback per prediction)

### Create the Feedback Router

`app/routers/feedback.py`:

```python
from fastapi import APIRouter, HTTPException
from app.models.schemas import FeedbackRequest, FeedbackResponse
from app.ml.predictions_store import predictions_store
from app.config import settings
import uuid

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit consent feedback for a prediction.

    The image is uploaded to S3 for future retraining.
    The prediction is removed from local storage after upload.
    """
    # CRITICAL: Consent must be explicitly True
    # Backend rejects anything else with 400
    if request.consent is not True:
        raise HTTPException(
            status_code=400,
            detail="Consent must be explicitly true. Feedback rejected."
        )

    # Find the prediction
    prediction = predictions_store.get(request.prediction_id)

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found or expired."
        )

    # TODO: Upload to S3
    # For now, just log and remove from store
    feedback_id = str(uuid.uuid4())

    # In a real implementation:
    # 1. Create S3 key: feedback/{date}/{feedback_id}.jpg
    # 2. Upload image bytes to S3
    # 3. Upload metadata JSON alongside
    # 4. Remove from predictions_store

    print(f"Feedback received: {feedback_id}")
    print(f"  Prediction: {request.prediction_id}")
    print(f"  Diagnosis: {prediction['diagnosis']}")
    print(f"  Consent: {request.consent}")
    if request.user_label:
        print(f"  User label: {request.user_label}")

    # Remove from local store after successful upload
    predictions_store.delete(request.prediction_id)

    return FeedbackResponse(
        feedback_id=feedback_id,
        status="queued",
        message="Thank you. Your image has been added to the training pool."
    )


@router.get("/feedback/stats")
async def feedback_stats():
    """
    Return feedback pool statistics (admin endpoint).
    """
    # In a real implementation, query S3 for pool size
    # For now, return mock data
    return {
        "pool_size": 0,
        "last_retrain_date": None,
        "retrain_ready": False,
        "min_required": 500,
    }
```

### What Just Happened?

1. We created a feedback endpoint that requires explicit consent
2. It retrieves the prediction, uploads to S3 (mock for now), and deletes from local store
3. It returns a feedback_id the user can reference
4. We also added a stats endpoint for admin monitoring

---

## Step 7: Add Model Loading

### What We Need to Add

Now we need to add real AI model loading and inference. This involves:
1. Loading the trained PyTorch model at startup
2. Running inference in the predict endpoint
3. Generating real Grad-CAM heatmaps in the explain endpoint

### Create the Model Loader

`app/ml/model_loader.py`:

```python
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
from typing import Optional, Tuple
from app.config import settings

# Image preprocessing (must match training)
IMAGE_SIZE = 224
PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SkinLesionClassifier:
    """Wrapper for the skin lesion classification model."""

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.target_layer: Optional[nn.Module] = None
        self.device: str = "cpu"
        self.model_version: str = "unknown"

    def load(self, model_path: str):
        """Load model from path."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model from {model_path} on {self.device}")

        # Create model architecture (must match training)
        self.model = timm.create_model("resnet50", pretrained=False, num_classes=2)

        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # Find the target layer for Grad-CAM
        # For ResNet50, this is the last convolutional layer
        self.target_layer = self.model.layer4[-1]

        self.model_version = model_path.split("/")[-1].replace(".pth", "")
        print(f"Model loaded: {self.model_version}")

    def predict(self, image_bytes: bytes) -> Tuple[str, float, dict, float]:
        """
        Classify a skin lesion image.

        Returns: (diagnosis, confidence, class_probabilities, processing_time_ms)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = PREPROCESS(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()

        # Map to diagnosis
        diagnosis = "malignant" if pred_class == 1 else "benign"
        class_probabilities = {
            "benign": probs[0].item(),
            "malignant": probs[1].item(),
        }

        return diagnosis, confidence, class_probabilities, 0  # ms placeholder

    def get_image_tensor(self, image_bytes: bytes) -> torch.Tensor:
        """Get preprocessed image tensor for Grad-CAM."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return PREPROCESS(image).unsqueeze(0).to(self.device)


# Import io for BytesIO
import io

# Global model instance
classifier = SkinLesionClassifier()
```

### Update Config to Support Fallback

The config already has fallback paths, but let's make sure the model loading handles errors gracefully.

### Update Main.py for Startup Model Loading

Update `app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict, explain, feedback, health
from app.ml.model_loader import classifier
from app.config import settings
from pathlib import Path

# Create FastAPI app
app = FastAPI(
    title="Skin Lesion Analysis API",
    description="AI-powered skin lesion classification with Grad-CAM explainability",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(predict.router, prefix="/api/v1", tags=["predict"])
app.include_router(explain.router, prefix="/api/v1", tags=["explain"])
app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])


@app.on_event("startup")
async def startup_event():
    """Load ML model at startup."""
    print("Starting up Skin Lesion Analysis API...")

    # Try to load model
    model_path = Path(settings.fallback_model_path)

    if model_path.exists():
        try:
            classifier.load(str(model_path))
            print(f"Model loaded successfully: {classifier.model_version}")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("API will start but prediction endpoints may fail")
    else:
        print(f"Warning: Model not found at {model_path}")
        print("Prediction endpoints will return mock data")


@app.get("/")
async def root():
    return {"message": "Skin Lesion Analysis API", "version": "1.0.0"}
```

### Update Health Endpoint to Report Model Info

Update `app/routers/health.py`:

```python
from fastapi import APIRouter, HTTPException
from app.models.schemas import HealthResponse, MethodsResponse
from app.ml.model_loader import classifier

router = APIRouter()

AVAILABLE_METHODS = ["gradcam", "gradcam_pp", "eigencam", "layercam"]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check - returns model info if loaded."""
    return HealthResponse(
        model_version=classifier.model_version,
        device=classifier.device,
        status="healthy" if classifier.model is not None else "degraded"
    )


@router.get("/methods", response_model=MethodsResponse)
async def get_methods():
    """Return available XAI methods."""
    return MethodsResponse(methods=AVAILABLE_METHODS)
```

### Update Predict to Use Real Model

Update `app/routers/predict.py`:

```python
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.models.schemas import PredictionResponse
from app.ml.predictions_store import predictions_store
from app.ml.model_loader import classifier
import time

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
):
    """Classify a skin lesion image."""
    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image."
        )

    # Read image bytes
    image_bytes = await image.read()

    # Check file size
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB."
        )

    # Check model is loaded
    if classifier.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    # Run inference
    start_time = time.time()

    diagnosis, confidence, class_probabilities, _ = classifier.predict(image_bytes)

    processing_time_ms = int((time.time() - start_time) * 1000)

    # Store prediction
    prediction_id = predictions_store.create(
        diagnosis=diagnosis,
        confidence=confidence,
        class_probabilities=class_probabilities,
        model_version=classifier.model_version,
        processing_time_ms=processing_time_ms,
        image_bytes=image_bytes,
        original_filename=image.filename,
    )

    return PredictionResponse(
        prediction_id=prediction_id,
        diagnosis=diagnosis,
        confidence=confidence,
        class_probabilities=class_probabilities,
        model_version=classifier.model_version,
        processing_time_ms=processing_time_ms,
    )
```

---

## Step 8: Dockerize the Backend

### Why Docker?

Docker packages your application with all its dependencies into a single "container" that runs anywhere. This ensures the backend works the same on your laptop, on a server, and in the cloud.

### Create Dockerfile

Create `Dockerfile` in the backend root:

```dockerfile
# Use Python 3.10 slim image (small size)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Model weights are pulled from S3 at runtime (not baked in)
# This keeps the image small (~500MB vs ~2GB)

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Create Docker Compose

Create `docker-compose.yml` in the project root (next to frontend and backend folders):

```yaml
version: "3.8"

services:
  frontend:
    build: ./Skin_Lesion_Classification_frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    depends_on:
      - backend

  backend:
    build: ./Skin_Lesion_Classification_backend
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=file:./mlruns
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-}
      - AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-eu-central-1}
    volumes:
      # Mount local model weights (for development)
      - ./Skin_Lesion_Classification_backend/ml/outputs/models:/app/ml/outputs/models
      # Mount MLflow for experiment tracking
      - ./mlruns:/app/mlruns
```

### Create Frontend Dockerfile

Create `Skin_Lesion_Classification_frontend/Dockerfile`:

```dockerfile
FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy application code
COPY . .

# Build Next.js for production
RUN npm run build

# Start the application
CMD ["npm", "start"]
```

---

## Step 9: Run and Test

### Run the Backend Locally

```bash
# Make sure you're in the backend directory
cd Skin_Lesion_Classification_backend

# Activate virtual environment
..\venv\Scripts\activate

# Run the development server
uvicorn app.main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Test the Endpoints

Open your browser and visit:

1. **Root**: http://localhost:8000/ → Shows API info
2. **Docs**: http://localhost:8000/docs → Interactive API documentation (FastAPI feature!)
3. **Health**: http://localhost:8000/api/v1/health → Returns model status
4. **Methods**: http://localhost:8000/api/v1/methods → Returns available CAM methods

### Test with curl

Open a new terminal (keep the server running) and test with curl:

```bash
# Test health endpoint
curl http://localhost:8000/api/v1/health

# Test methods endpoint
curl http://localhost:8000/api/v1/methods

# Test predict endpoint (need to send an image)
curl -X POST http://localhost:8000/api/v1/predict \
  -F "image=@path/to/your/image.jpg"
```

### Run the Full Stack with Docker

```bash
# In the project root (where docker-compose.yml is)
docker-compose up --build

# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
# Docs:     http://localhost:8000/docs
```

---

## Backend Summary

### What You Built

1. **FastAPI App** (`app/main.py`)
   - Entry point with CORS, routers, startup events

2. **Configuration** (`app/config.py`)
   - Environment variables, settings, defaults

3. **Models/Schemas** (`app/models/schemas.py`)
   - Pydantic models for request/response validation

4. **Routers**
   - `health.py` - health check and methods endpoints
   - `predict.py` - image classification endpoint
   - `explain.py` - Grad-CAM heatmap generation
   - `feedback.py` - consent-based training data collection

5. **ML Components** (`app/ml/`)
   - `predictions_store.py` - in-memory cache with TTL
   - `model_loader.py` - PyTorch model wrapper

6. **Docker**
   - `Dockerfile` - containerized backend
   - `docker-compose.yml` - full stack orchestration

### Key Concepts Learned

- **FastAPI** - Python web framework for building APIs
- **Endpoints** - URL routes that handle requests
- **Routers** - Group related endpoints together
- **Pydantic** - Data validation and settings management
- **CORS** - Cross-origin resource sharing (frontend-backend communication)
- **In-memory Store** - Temporary prediction caching
- **Docker** - Containerization for consistent deployments

### Run Commands

```bash
# Backend only
cd Skin_Lesion_Classification_backend
venv\Scripts\activate
uvicorn app.main:app --reload --port 8000

# Full stack (from project root)
docker-compose up --build

# Run tests
pytest

# Lint
ruff check .
```

---

## Common Errors and Fixes

**"Model not found" errors**
→ Check `ml/outputs/models/resnet50_best.pth` exists
→ Check `fallback_model_path` in config is correct

**"CORS blocked" errors in browser**
→ Make sure CORS middleware is added in `main.py`
→ Check `allowed_origins` includes your frontend URL

**"Connection refused" when calling API**
→ Make sure backend is running: `uvicorn app.main:app`
→ Check port 8000 is not already in use
→ Check `.env.local` has correct `NEXT_PUBLIC_API_URL`

**Docker build fails**
→ Make sure Docker Desktop is running
→ Try: `docker-compose down && docker-compose build --no-cache`

---

## Next Steps After This Guide

1. **Add real Grad-CAM implementation** - currently the explain endpoint returns mock data
2. **Add S3 integration** - currently feedback uploads are mocked
3. **Add MLflow tracking** - log experiments and model versions
4. **Add the retraining script** - weekly cron job for model improvement
5. **Deploy to AWS** - ECS Fargate, S3, ECR

For the full implementation details, see the documentation in `.claude/docs/`:
- `PRD.md` - Complete product requirements
- `FEEDBACK_AND_RETRAINING.md` - Feedback pipeline and retraining details
