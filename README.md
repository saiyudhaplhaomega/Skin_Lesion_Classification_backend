# Skin Lesion Classification Backend

FastAPI backend for skin lesion inference, image-quality checks, explainability, and safe AI explanations.

This repo owns the service layer: API endpoints, model loading for serving, image-quality gating, Grad-CAM generation, LLM/RAG explanation orchestration, consent/retraining API flows, and deployment packaging. Research notebooks and RQ1-RQ6 experiments live in the separate research repo.

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
2. `POST /api/v1/image-quality`
3. mocked `POST /api/v1/predict`
4. image validation and retake-guidance tests
5. real model loading from `ml/outputs/models/`
6. Redis-backed `POST /api/v1/explain`
7. rule-based explanation fallback
8. LLM/RAG explanation endpoint
9. CrewAI expert-panel workflow after the single-agent explanation path is safe and testable

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
| `POST /api/v1/image-quality` | Score blur, lighting, framing, resolution, glare, and return retake guidance or proceed-with-warning flags |
| `POST /api/v1/predict` | Accept an image and return diagnosis, confidence, model version, and prediction ID |
| `POST /api/v1/explain` | Return Grad-CAM or related heatmap outputs for a prediction |
| `POST /api/v1/explain-llm` | Return a guarded, RAG-grounded natural-language explanation of the prediction and heatmap |
| `POST /api/v1/chat` | Follow-up chat over one analysis, constrained by RAG policy and safety guardrails |
| `POST /api/v1/consent` | Persist opted-in training cases through the consent pipeline |
| `GET /docs` | FastAPI interactive API documentation |

## Product Pipeline

Build the backend pipeline in this order:

```text
image upload
-> image-quality gate
-> prediction
-> Grad-CAM / heatmap
-> explanation facts builder
-> rule-based fallback
-> RAG policy retrieval
-> online/local LLM explanation
-> safety validator
-> final response
```

The image-quality gate should return actionable guidance such as better lighting, steadier capture, centered framing, less glare, original-file upload instead of screenshots, or using a better camera if available.

Poor images should not always be hard-blocked. Return `requires_user_acknowledgement = true` and allow the frontend to offer `Retake Photo` or `Proceed Anyway`.

## LLM and RAG Guardrails

The LLM layer must explain structured model outputs; it must not diagnose from the image.

Allowed:

- explain the model prediction and confidence
- explain what Grad-CAM highlights mean
- explain image-quality limitations
- recommend professional review for concerning or uncertain cases

Blocked:

- definitive diagnosis claims
- treatment recommendations
- telling users to ignore doctors
- interpreting heatmaps as disease proof

Use deterministic templates as fallback when LLM providers, local models, or RAG retrieval fail.

CrewAI is approved as a later expert-panel feature. It should consume the same structured prediction, image-quality, heatmap, and RAG facts as the normal explanation endpoint. Do not let CrewAI bypass the safety validator or answer as a doctor.

## Scale Direction

The backend should be built sharding-ready even while local development starts simple.

- Include `user_id` on analysis, prediction, heatmap, explanation, chat, and audit records.
- Add `tenant_id` later when clinic or organization accounts exist.
- Use cursor pagination for large doctor/admin lists.
- Store images, overlays, and reports in S3-compatible object storage, not Postgres.
- Keep API containers stateless.
- Put slow prediction, Grad-CAM, LLM, report generation, and consent pipeline work onto queues.
- Start with one Postgres database, then add indexes, partitioning, read replicas, and only then physical sharding.

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

Jupyter kernels are registered from the research repo, not here:

```bash
cd ../Skin_Lesion_XAI_research
make setup
make register-kernel
```

## Build Guide

Use [`BUILD_BACKEND.md`](BUILD_BACKEND.md) for a beginner-friendly implementation walkthrough.

For production sequencing and architecture decisions, use the root docs:

- `../docs/BUILD_PHASE_2_BACKEND.md`
- `../docs/PRODUCTION_BUILD_REVIEW.md`
- `../docs/SYSTEM_DESIGN_LEARNING_GUIDE.md`
- `../docs/PRODUCT_LAUNCH_STRATEGY.md`
- `../docs/BUILD_GUIDE_AUDIT.md`

## Verification

```bash
make test
make lint
make typecheck
```

Those targets require the backend app/test structure to exist. Until then, use them as the acceptance criteria for the next backend implementation pass.
