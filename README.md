# MoCoLUS — Motion-Compensated Lung Ultrasound Simulator

Synthetic lung POCUS training simulator with real-time web-based visualization, BLE probe IMU integration, and AI-generated ultrasound frames. Uses a ControlNet-guided pixel-space diffusion model trained on ~14,000 real clinical POCUS images to produce photorealistic B-mode and M-mode frames across 10 pathology classes and 15 clinical scenarios.

Runs on the **ThinkStation PX** (`ahastava@thinkstationpgx-9c7e`) — headless aarch64 Ubuntu 24.04, NVIDIA GB10 Blackwell GPU.

> For detailed system architecture, data flow diagrams, and developer onboarding, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Quick Start

### Option 1: Docker (recommended for deployment)

```bash
# Build CPU image with pre-rendered frame cache (no GPU needed at runtime)
docker build -f Dockerfile.cpu -t ahastava/moculus:latest .

# Run
docker run -p 8000:8000 ahastava/moculus:latest

# Open in browser
open http://localhost:8000
```

### Option 2: Local Python (for development)

```bash
pip install -r requirements.txt
./run.sh --web
# → http://localhost:8000
```

### Option 3: VS Code Remote SSH

```bash
ssh ahastava@thinkstationpgx-9c7e
cd ~/moculus
./run.sh --web
# Forward port 8000 in VS Code PORTS tab → http://localhost:8000
```

---

## Run Modes

| Command | What it does |
|---------|-------------|
| `./run.sh --web` | **Web UI** on http://localhost:8000 (recommended) |
| `./run.sh --docker` | Build and run via Docker Compose (GPU) |
| `./run.sh --train` | Train ControlNet DDPM on clinical frames |
| `./run.sh --dataset` | Build HDF5 training dataset |
| `./run.sh --preview` | Generate training preview images |
| `./run.sh` | PyQt6 GUI (legacy, deprecated) |
| `./run.sh --help` | Show all options |

---

## Web Interface

The web UI has two tabs:

### Simulator Tab

- **B-mode / M-mode display** — real-time ultrasound frames streamed at 30fps via WebSocket
- **Chest zone map** — interactive SVG with 8 BLUE protocol zones (click to examine)
- **Free probe mode** — drag the probe across the chest; frames interpolate smoothly between zones
- **Practice mode** — select a scenario, examine zones, see findings in real-time
- **Test mode** — random hidden case, examine all 8 zones, submit your diagnosis
- **Patient case** — randomized demographics, vitals, chief complaint, history
- **Playback** — play/pause, frame slider, freeze
- **BLE probe** — connect physical probe via Web Bluetooth for IMU-based navigation

### Training Studio Tab

- **Train diffusion model** — configure epochs, batch size, image size; real-time loss chart via SSE
- **Build HDF5 dataset** — generate training data with progress tracking
- **Checkpoints** — list and manage model checkpoints

---

## Train the Diffusion Model

MoCoLUS uses a pixel-space conditional DDPM with ControlNet-style structural guide injection, trained on real clinical POCUS images.

```bash
# Full training run (from scratch)
python -m src.train_realistic --epochs 300 --batch-size 8 --lr 1e-4

# Fine-tune from existing checkpoint
python -m src.train_realistic --resume checkpoints/realistic_v4_ab/best.pt \
    --finetune --epochs 20 --lr 3e-5

# Resume interrupted training
python -m src.train_realistic --resume checkpoints/realistic_v4_ab/latest.pt

# Run in background (survives SSH disconnect)
nohup python -u -m src.train_realistic --epochs 300 --batch-size 8 \
    > checkpoints/training.log 2>&1 & disown
```

### Training Pipeline

```
Real POCUS images (14,258 train / 1,470 val)
        ↓
ClinicalFrameGenerator ──→ Structural guide (physics-based)
        ↓
LesionAnatomyBank ────────→ Real lesion texture + PMF placement (50% prob)
        ↓
ControlNetPOCUS (69.2M params)
  ├── UNet2DModel (denoising path, 1 input channel)
  ├── GuideEncoder (4-level CNN → multi-scale features)
  └── Zero-conv injection at every decoder level
        ↓
MSE loss on noise prediction + classifier-free guidance (10% dropout)
        ↓
Checkpoints:
  ├── realistic_v4_ab/best.pt    (base model, all pathologies)
  └── realistic_v2_finetune/     (trauma-specialized: PTX, effusion, lung point)
```

### Why Pixel-Space (not Latent)?

The Stable Diffusion VAE is trained on natural photographs. When encoding ultrasound speckle, the decoder produces oil-painting artifacts. Operating in pixel space avoids this entirely.

---

## Frame Cache (Offline Deployment)

For CPU-only deployment (no GPU at runtime), pre-render all frames:

```bash
# Generate frame cache (all 15 scenarios × 8 zones × 16-32 frames)
python scripts/smart_cache_gen.py --n-frames 16 --ddim-steps 30

# Monitor progress
tail -f checkpoints/smart_cache.log

# View visual dashboard
python scripts/cache_dashboard.py
# → checkpoints/benchmarks/cache_progress/dashboard.png
```

The cache (`data/frame_cache.npz`) is baked into the Docker CPU image. At runtime, frames load instantly from the cache — no neural network inference needed.

---

## Docker

### CPU Image (deployment, no GPU required)

```bash
docker build -f Dockerfile.cpu -t ahastava/moculus:latest .
docker run -p 8000:8000 ahastava/moculus:latest
```

Ships with `frame_cache.npz` — all frames pre-rendered on GPU, served from cache at runtime.

### GPU Image (training + live generation)

```bash
docker compose up --build
```

### Offline Transfer

```bash
docker save ahastava/moculus:latest | gzip > moculus.tar.gz
# On offline machine:
docker load < moculus.tar.gz
docker run -p 8000:8000 ahastava/moculus:latest
```

---

## Two Deployment Modes

### Mode 1: Online — Training & Development

```bash
./run.sh --web    # → http://localhost:8000
```

Full access: train models, generate datasets, create previews, live diffusion generation.

### Mode 2: Offline — Field Use with BLE Probe

Deploy via Docker on any machine (laptop, tablet, field workstation):

1. Transfer Docker image
2. `docker run -p 8000:8000 ahastava/moculus:latest`
3. Open browser → http://localhost:8000
4. Connect BLE probe via Web Bluetooth

All frames served from pre-rendered cache. No GPU, no internet required.

---

## Project Structure

```
moculus/
  run.sh                              Single entry point for all modes
  Dockerfile.cpu                      CPU-only image with frame cache
  Dockerfile.gpu                      GPU image for training
  docker-compose.yml                  Docker Compose with GPU support
  requirements.txt                    Python dependencies
  ARCHITECTURE.md                     System architecture deep dive

  src/
    web_server.py                     FastAPI server (WebSocket + REST)
    poc_image_stack.py                BLUE protocol zones, scenarios, stack orchestrator
    clinical_frames.py                Physics-based structural guide generator (10 pathologies)
    realistic_generator.py            ControlNet DDPM inference wrapper + trauma routing
    anatomy_bank.py                   Lesion-Anatomy Bank (DiffUltra concept)
    train_realistic.py                ControlNet DDPM training loop
    lung_us_generator.py              Legacy zea bridge + IMU/grid dataclasses
    lung_us_dataset.py                HDF5 dataset builder
    simulator_bridge.py               GUI adapter layer (legacy)
    acquire_pocus_data.py             Real POCUS data acquisition pipeline
    generate_cache.py                 Basic frame cache generator

  scripts/
    smart_cache_gen.py                Self-correcting cache gen with quality gating
    cache_dashboard.py                Visual progress dashboard
    generate_balanced.py              Balanced dataset generation
    validate_synthetic.py             Synthetic frame quality validation
    scan_class_balance.py             Class distribution analysis
    benchmark_integration.py          Physics vs ControlNet comparison
    benchmark_finetune.py             Model version comparison
    finetune_trauma.sh                Trauma-specialized fine-tuning script

  static/
    index.html                        Single-page web app
    app.js                            Frontend (WebSocket, canvas, BLE, zone map)
    style.css                         Dark clinical theme
    probe3d.js                        3D probe visualization (Three.js)
    probe_mesh.stl                    Probe 3D model

  checkpoints/                        Model weights (gitignored)
    realistic_v4_ab/best.pt           Production base model
    realistic_v2_finetune/latest.pt   Trauma-specialized model
    anatomy_bank.pt                   Lesion texture bank

  data/                               Training data (gitignored)
    real_pocus/processed/             14,258 labeled clinical frames
    frame_cache.npz                   Pre-rendered frame cache for Docker
    DATA_PROVENANCE.md                Complete dataset licensing & attribution
```

---

## Pathologies

| # | Pathology | B-mode Appearance | M-mode |
|---|-----------|-------------------|--------|
| 0 | Normal A-Profile | Pleural line + A-lines | Seashore |
| 1 | Pneumothorax | A-lines, no sliding | Stratosphere |
| 2 | Focal B-Lines | 1-2 vertical artifacts | Seashore |
| 3 | Diffuse B-Lines | ≥3 B-lines (B-profile) | Seashore |
| 4 | Consolidation | Tissue-like + air bronchograms | Stratosphere |
| 5 | Pleural Effusion | Anechoic fluid + quad sign | Seashore |
| 6 | ARDS / White Lung | Confluent B-lines | Stratosphere |
| 7 | Lung Point | Sliding / no sliding transition | Mixed |
| 8 | Pleural Thickening | Irregular pleural line | Seashore |
| 9 | Interstitial Syndrome | B-lines + subpleural consolidations | Seashore |

## Clinical Scenarios (15)

| Scenario | Diagnosis | BLUE Profile |
|----------|-----------|--------------|
| `normal` | Normal lung exam | Bilateral A-profile + sliding |
| `copd_exacerbation` | COPD/asthma (normal LUS!) | Bilateral A-profile + sliding |
| `left_pneumothorax` | Left tension pneumothorax | Left A-prime, Right A-profile |
| `right_pneumothorax` | Right tension pneumothorax | Right A-prime, Left A-profile |
| `left_pneumothorax_with_lung_point` | Partial PTX with lung point | A-prime + lung point |
| `pulmonary_edema` | Cardiogenic pulmonary edema | Bilateral B-profile |
| `pulmonary_edema_with_effusion` | Severe CHF + bilateral effusions | B-profile + effusions |
| `ards` | ARDS / white lung | Bilateral white lung, absent sliding |
| `left_pneumonia` | Left lower lobe pneumonia | Left A/B-profile + PLAPS |
| `right_pneumonia` | Right lower lobe pneumonia | Right A/B-profile + PLAPS |
| `bilateral_pneumonia` | COVID-19 / viral pneumonia | Bilateral interstitial + consolidation |
| `right_pleural_effusion` | Right pleural effusion | Anterior A-profile + posterior effusion |
| `bilateral_effusion` | Bilateral pleural effusions | Focal B-lines + bilateral effusions |
| `left_hemothorax` | Hemothorax (trauma) | Left B-lines + posterior effusion |
| `pneumonia_with_effusion` | Pneumonia + parapneumonic effusion | B-profile + consolidation + effusion |

---

## First-Time Setup

### Using Docker (recommended)

```bash
docker build -f Dockerfile.cpu -t ahastava/moculus:latest .
docker run -p 8000:8000 ahastava/moculus:latest
```

### Manual Setup

```bash
python3 -m venv ~/moculus_env
source ~/moculus_env/bin/activate
pip install -r requirements.txt
pip install bleak  # optional, for BLE probe

./run.sh --web
```
