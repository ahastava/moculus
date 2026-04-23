# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MoCoLUS — synthetic lung POCUS training simulator. Frontend: vanilla JS single-page app served by FastAPI. Backend: a pixel-space conditional DDPM with ControlNet-style structural-guide injection (`ControlNetPOCUS`, ~69M params), trained on ~14k real clinical POCUS frames across 10 pathology classes. Production target is a Docker CPU image that serves pre-rendered frames from `data/frame_cache.npz` — no GPU at runtime.

For deep architecture and clinical reference, see `ARCHITECTURE.md`. For data-sourcing pipeline see `DIAPHRAGM_DATA.md`. For retraining workflow see `RETRAINING.md`.

## Common Commands

All Python entry points assume the venv at `~/moculus_env` and `KERAS_BACKEND=torch` (set by `run.sh`). Activate manually if invoking Python directly:

```bash
source ~/moculus_env/bin/activate
export KERAS_BACKEND=torch
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
```

### Run

```bash
./run.sh --web              # FastAPI web UI on :8000 (the standard mode)
./run.sh --docker           # docker compose up --build (GPU)
./run.sh --train [...]      # legacy entry — calls src.train_zea_diffusion
./run.sh --dataset          # build HDF5 training set
./run.sh --preview          # generate preview images
```

`run.sh` has a `--background` flag for `--train` that detaches into a tmux session named `moculus-train`.

### Train the production diffusion model

The actual production model is trained via `src.train_realistic` (NOT `src.train_zea_diffusion` from `run.sh --train`):

```bash
# Full run from scratch
python -m src.train_realistic --epochs 300 --batch-size 8 --lr 1e-4

# Fine-tune from existing checkpoint (resets optimizer)
python -m src.train_realistic --resume checkpoints/realistic_v4_ab/best.pt \
    --finetune --epochs 20 --lr 3e-5 --output-dir checkpoints/realistic_v5

# Resume an interrupted run (keeps optimizer state — omit --finetune)
python -m src.train_realistic --resume checkpoints/realistic_v4_ab/latest.pt

# LoRA fine-tune (Phase 5, zone-aware)
python -m src.train_realistic --resume checkpoints/realistic_v2_finetune/latest.pt \
    --finetune --lora --lora-rank 16 --lora-alpha 32 \
    --output-dir checkpoints/realistic_v2_diaphragm_lora \
    --epochs 20 --batch-size 16 --lr 3e-4
```

### Cache pipeline (required after retraining)

```bash
python scripts/scan_class_balance.py           # per-class distribution
python scripts/generate_balanced.py            # synth fill underrepresented classes
python scripts/validate_synthetic.py           # quality gate
python scripts/smart_cache_gen.py --n-frames 16 --ddim-steps 30
python scripts/cache_dashboard.py              # writes dashboard.png
```

`smart_cache_gen.py` supports `--only-zones` and `--diaphragm-model` for partial regen.

### Docker

```bash
# CPU (deployment) — bakes in frame_cache.npz
docker build -f Dockerfile.cpu -t ahastava/moculus:latest .

# GPU (dev/training)
docker compose up --build

# Multi-arch publish
docker buildx build --platform linux/amd64,linux/arm64 \
  -f Dockerfile.cpu -t ahastava/moculus:latest --push \
  --build-arg CACHEBUST=$(date +%s) .
```

### Pre-commit

`detect-secrets` against `.secrets.baseline` is the only hook. There is **no test suite, linter, or type-checker configured** — don't invent commands for these.

## Architecture — Things You Need to Know to Be Useful

### The two production models and trauma routing

`realistic_generator.py` loads two checkpoints and routes per-class:

```
TRAUMA_CLASSES = {1, 5, 7}   # PTX, Effusion, Lung Point
class ∈ TRAUMA_CLASSES  →  checkpoints/realistic_v2_finetune/latest.pt
otherwise               →  checkpoints/realistic_v4_ab/best.pt
```

The default model path lives at `src/poc_image_stack.py` ~line 1109 (`_DEFAULT_MODEL` / `_TRAUMA_MODEL`). After retraining, **update those constants** — the web server reads them at startup.

### Class label space (don't renumber)

`train_realistic.py` defines a labeling scheme that the on-disk metadata and embedding tables both depend on:

```
NUM_PATHOLOGY_CLASSES = 10
MMODE_CLASS_OFFSET    = 11    # M-mode class n = n + 11
NULL_CLASS            = 21    # CFG dropout slot
NUM_EMBEDDINGS        = 22    # 10 bmode + 10 mmode + null + gap

NUM_ZONE_REGIONS = 9          # ZoneRegion enum from clinical_frames.py
ZONE_REGION_NULL = 7          # CFG dropout for zone embedding
```

`ZoneRegion` integer values in `clinical_frames.py` are persisted into `metadata.csv` and into trained zone embeddings — renumbering breaks both.

### Frame pipeline (3 stages)

1. **`clinical_frames.py` → `ClinicalFrameGenerator.generate(class)`** — physics-based synthetic guide (tissue layers, A/B-lines, rib shadows, TGC, speckle). Output `[256,256] float32`.
2. **`anatomy_bank.py` → `LesionAnatomyBank`** — DiffUltra-style texture bank + spatial PMF. Applied 50% during training, 100% during inference.
3. **`realistic_generator.py` → ControlNet DDPM** — DDIM sample with classifier-free guidance. Per-class guidance scale overrides defined in this file (e.g. PTX=6.0, default=4.0).

### Why pixel-space, not latent

The Stable Diffusion VAE is trained on natural photos and turns ultrasound speckle into oil-painting artifacts on decode. **Do not propose latent-space variants** — the design choice is intentional and documented in the `train_realistic.py` module docstring.

### Why no database

Sessions live for the duration of a WebSocket. State is `WebSimSession` per connection, garbage collected on disconnect. Training metrics go to `metrics.csv` + TensorBoard. Frames live in `frame_cache.npz`. This is a training simulator with no persistence requirements — don't add one.

### Web request lifecycle

`web_server.py` `/ws/simulator`: probe `(nx, ny)` → `get_interpolated_frame_data` → Gaussian-weighted blend across 8 zone anchors (σ=0.14) → JPEG (q=92) → base64 → WebSocket. Cache lookup + blend + encode is ~2-5ms; WebSocket RTT dominates.

### Frontend

`static/app.js` is ~1700 lines of vanilla JS (no framework). Key classes: `WebSocketManager`, `BmodeRenderer`, `MmodeRenderer`, `ProbeOverlayRenderer`, `BLEProbeManager`. BLE is Nordic UART (`6e400001-...`); IMU CSV stream `"YPR=yaw,pitch,roll,..."` at ~50Hz, throttled to ~12Hz on the wire.

### Augmentation policy (training)

Only clinically-valid augmentations are allowed in `train_realistic.py`: horizontal flip, brightness ±15%, contrast ±10%, additive Gaussian noise. **Never add** vertical flip, rotation, color jitter, or random erasing — they violate ultrasound physics and the docstring explicitly forbids them.

### Ignored / excluded paths

`data/`, `checkpoints/`, `*.h5`, `*.pt`, `*.npz`, `*.safetensors`, `notebooks/`, `POCUS CEWIT GitLab/` (legacy hardware code) are gitignored. The `dist/` directory holds installer scripts shipped with the Docker image (`install.sh`, `moculus.sh`, `start-moculus.{sh,bat}`).

## Conventions

- The `--web` mode is the canonical run target. The PyQt6/VNC GUI in `simulator_bridge.py` and `run.sh --vnc` is deprecated legacy.
- `run.sh --train` invokes `src.train_zea_diffusion` (legacy zea bridge) — for the production model use `python -m src.train_realistic` directly.
- Docs reference an `_sources/` tree under `data/` for COVID-BLUES and COVIDx-US raw videos. That tree is gitignored — bootstrap it locally before running ingest scripts.
