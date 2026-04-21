# Skin Lesion Classification Backend

FastAPI service for dermoscopy image classification (benign/malignant) with Grad-CAM explainability, plus all PyTorch ML research code.

## Quick Start

**1. Download data + verify splits** — run `00_setup_and_sanity.ipynb`

**2. Train a model** — run Cells 6-9 in `00_setup_and_sanity.ipynb` (ResNet50, ~15 min on GPU, ~1-2h on CPU)

**3. Run experiments** — RQ1 through RQ5 notebooks in `notebooks/`

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `00_setup_and_sanity.ipynb` | Environment check, HAM10000 download, patient-level split, DataLoader sanity, **training (Cells 6-9)** |
| `RQ1_cam_variant_comparison.ipynb` | Which CAM method best localizes lesions? |
| `RQ2_faithfulness.ipynb` | Do CAM faithfulness metrics correlate with clinical trust? |
| `RQ3_backbone_xai_quality.ipynb` | Does backbone architecture affect XAI quality? |
| `RQ4_agreement_vs_uncertainty.ipynb` | Does inter-method disagreement predict misclassification? |
| `RQ5_temporal_xai.ipynb` | How does Grad-CAM attention evolve during training? |
| `PAPER_RESULTS_TABLE.ipynb` | Compile all RQ results into LaTeX tables |

## Data

HAM10000 is downloaded automatically by `00_setup_and_sanity.ipynb`:
- `ml/data/processed/metadata_with_paths.csv` — labels and image paths
- `ml/data/processed/raw/images/` — 10,015 dermoscopy images

## Models

After training, checkpoints are saved to `ml/outputs/models/`:
- `resnet50_best.pth` — best ResNet50 by validation AUC
- `checkpoints/resnet50_epoch*.pth` — intermediate checkpoints for RQ5

## API (not yet implemented — see `app/` in future)

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/predict` | Upload image → benign/malignant + confidence |
| `POST /api/v1/explain` | Generate Grad-CAM heatmap |
| `POST /api/v1/feedback` | Opt-in feedback for retraining |
