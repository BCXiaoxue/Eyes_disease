# RetinaScope

RetinaScope is a local Streamlit app for binocular fundus image review, multi-label ODIR-style risk prediction, Grad-CAM visualization, local knowledge retrieval, AI consultation, and file-backed reviewer feedback.

This project is for graduation design, research demos, and screening assistance. It does not replace professional ophthalmic diagnosis or treatment decisions.

## Quick Start

```powershell
cd D:\U_files\project_code\eyes_diseases_code
python -m streamlit run app.py
```

Run tests:

```powershell
python -B -m unittest discover -s tests -v
```

Optional model smoke test:

```powershell
python -B scripts\smoke_model.py --device cpu
python -B scripts\smoke_model.py --device cuda
```

## Main Features

- Separate left-eye and right-eye image upload.
- Training-directory case selection from `train/images/<id>_left.jpg` and `<id>_right.jpg`.
- Side-by-side binocular stitching before model inference.
- 8-label risk output with calibrated thresholds.
- On-demand Swin Grad-CAM visualization.
- Local Markdown knowledge retrieval from `knowledge/*.md`.
- AI consultation through the DeepSeek OpenAI-compatible API.
- Local JSONL feedback storage in `artifacts/feedback/`.

The app opens directly to the workspace and stores reviewer feedback locally.

## Key Paths

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit UI and workflow entry point |
| `utils/model.py` | Model loading, prediction, and Grad-CAM helpers |
| `utils/binocular_label_graph.py` | Binocular multi-label experiment model utilities |
| `utils/rag.py` | Local Markdown retrieval |
| `utils/consult.py` | AI consultation message construction |
| `utils/logger.py` | Feedback save/load entry point |
| `utils/storage.py` | Local JSONL feedback persistence |
| `utils/paths.py` | Project path constants |
| `knowledge/` | Local ophthalmology knowledge files |
| `assets/overview/` | Overview page images |
| `artifacts/models/` | Runtime model checkpoints and evaluation metadata |
| `artifacts/feedback/` | Local reviewer feedback records |
| `training/` | Training and evaluation scripts |
| `tests/` | Unit tests |
| `deployment/` | Docker and Streamlit deployment config |

## Environment

Copy `.env.example` to `.env` when local secrets or model overrides are needed.

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash

RETINASCOPE_MODEL_FILE=best_swin_tiny_linear_asl.pth
RETINASCOPE_ENABLE_TTA=1
RETINASCOPE_TTA_EVAL_FILE=eval_swin_tiny_linear_asl_tta.json
```

## Data And Artifacts

Expected runtime inputs:

- `data/label.csv`
- `train/images/*_left.jpg`
- `train/images/*_right.jpg`
- `knowledge/*.md`
- `assets/overview/*.png`
- `artifacts/models/best_swin_tiny_linear_asl.pth`
- `artifacts/models/eval_swin_tiny_linear_asl_tta.json`

Generated local feedback is written under `artifacts/feedback/`.

## Deployment

Validate Docker Compose config:

```powershell
docker compose -f deployment\docker-compose.yml config --quiet
```

Run with Docker Compose:

```powershell
docker compose -f deployment\docker-compose.yml up --build
```

## Notes

- Do not commit `.env`, API keys, database files, real patient data, or private model checkpoints.
- The app writes feedback locally by design.
- If CUDA is explicitly requested and unavailable, model utilities raise an error instead of silently falling back to CPU.
