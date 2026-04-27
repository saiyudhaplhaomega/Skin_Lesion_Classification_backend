# Skin Lesion Classification Backend

FastAPI backend for skin lesion inference and explainability.

This repo owns the service layer: API endpoints, model loading for serving, Grad-CAM generation, consent/retraining API flows, and deployment packaging. Research notebooks and RQ1-RQ6 experiments live in the separate research repo.

## Repo Boundaries

| Repo | Responsibility |
| --- | --- |
| `Skin_Lesion_Classification_backend` | FastAPI API, inference runtime, explainability runtime, backend tests, Docker/ECS packaging |
| `Skin_Lesion_XAI_research` | HAM10000 notebooks, RQ1-RQ6 experiments, training scripts, figures, and metrics |
| `Skin_Lesion_Classification_frontend` | Next.js UI that calls this backend |
| `Skin_Lesion_GRADCAM_Classification` | Workspace-level architecture, Terraform, security docs, and build roadmap |

## Current State

The repo currently contains:

- `ml/src/` shared ML modules used by training and serving
- `ml/data/processed/metadata_with_paths.csv`
- model artifacts under `ml/outputs/models/`
- backend dependency files and Makefile targets

The FastAPI app still needs to be built. Start with the smallest useful API surface:

1. `GET /health`
2. mocked `POST /api/v1/predict`
3. image validation tests
4. real model loading from `ml/outputs/models/`
5. Redis-backed `POST /api/v1/explain`

## Quick Start

```bash
make setup
make run
```

Equivalent manual commands:

```bash
py -3.13 -m venv skin-lesion-env
skin-lesion-env\Scripts\python.exe -m pip install --upgrade pip
skin-lesion-env\Scripts\python.exe -m pip install -r requirements-dev.txt
skin-lesion-env\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

## Expected API Surface

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Report service and model-load status |
| `POST /api/v1/predict` | Accept an image and return diagnosis, confidence, model version, and prediction ID |
| `POST /api/v1/explain` | Return Grad-CAM or related heatmap outputs for a prediction |
| `POST /api/v1/consent` | Persist opted-in training cases through the consent pipeline |
| `GET /docs` | FastAPI interactive API documentation |

## Model Artifacts

Research and training work should happen in `../Skin_Lesion_XAI_research`. Backend serving consumes the resulting artifacts from:

```text
ml/outputs/models/
  resnet50_best.pth
  efficientnet_b2_best.pth
  mobilenetv2_100_best.pth
  checkpoints/
```

For local development, keep the backend and research repos as siblings so the research path helpers can find `Skin_Lesion_Classification_backend/ml/`.

## Build Guide

Use [`BUILD_BACKEND.md`](BUILD_BACKEND.md) for a beginner-friendly implementation walkthrough.

For production sequencing and architecture decisions, use the root docs:

- `../docs/BUILD_PHASE_2_BACKEND.md`
- `../docs/PRODUCTION_BUILD_REVIEW.md`
- `../docs/SYSTEM_DESIGN_LEARNING_GUIDE.md`

## Verification

```bash
make test
make lint
make typecheck
```

Those targets require the backend app/test structure to exist. Until then, use them as the acceptance criteria for the next backend implementation pass.
