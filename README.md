# MoCoLUS — Motion-Compensated Lung Ultrasound Simulator

Synthetic lung POCUS training simulator with real-time web-based visualization, BLE probe IMU integration, and clinically accurate pathology rendering. Uses zea DiffusionModel for photorealistic image refinement trained on physics-based clinical frames.

Runs on the **ThinkStation PX** (`ahastava@thinkstationpgx-9c7e`) — headless aarch64 Ubuntu 24.04, NVIDIA GB10 GPU.

---

## Quick Start

### Option 1: Docker (recommended)

```bash
# Build and run with GPU support
docker compose up --build

# Open in browser
open http://localhost:8000
```

### Option 2: Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Launch web UI
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
| `./run.sh --docker` | Build and run via Docker Compose |
| `./run.sh --train` | Train zea diffusion model on clinical frames |
| `./run.sh --dataset` | Build HDF5 training dataset |
| `./run.sh --preview` | Generate training preview images |
| `./run.sh` | PyQt6 GUI (auto-detect display, legacy) |
| `./run.sh --vnc` | PyQt6 GUI with VNC (deprecated) |
| `./run.sh --help` | Show all options |

---

## Web Interface

The web UI has two tabs:

### Simulator Tab

- **B-mode / M-mode display** — real-time ultrasound frames streamed at 30fps via WebSocket
- **Chest zone map** — interactive SVG with 8 BLUE protocol zones (click to examine)
- **Practice mode** — select a scenario, examine zones, see findings in real-time
- **Test mode** — random hidden case, examine all 8 zones, submit your diagnosis
- **Patient case** — randomized demographics, vitals, chief complaint, history
- **Playback** — play/pause, frame slider, freeze
- **BLE probe** — connect to physical probe via server-side BLE

### Training Studio Tab

- **Train diffusion model** — configure epochs, batch size, image size; real-time loss chart via SSE
- **Build HDF5 dataset** — generate training data with progress tracking
- **Generate previews** — pathology grid, B-mode/M-mode grid, scenario views
- **Checkpoints** — list and manage model checkpoints

---

## Train the Diffusion Model

```bash
# Quick test (< 5 seconds)
./run.sh --train --epochs 3 --n-per-class 8 --batch-size 2 --image-size 64 --from-scratch

# Full training run
./run.sh --train --epochs 100 --batch-size 4

# Train from scratch (no pretrained echonet init)
./run.sh --train --from-scratch --epochs 200

# Resume from checkpoint
./run.sh --train --resume checkpoints/zea_lung_pocus/latest.pt

# Generate samples from trained model
./run.sh --train --sample-only

# Run in background (survives SSH disconnect)
./run.sh --train --epochs 100 --background
```

Or use the **Training Studio** tab in the web UI for training with real-time progress.

### Training Pipeline

```
ClinicalFrameGenerator (numpy/scipy)    →  10 pathologies, fresh each epoch
        ↓                                  (speckle, noise, augmentation)
zea DiffusionModel (unet_time_conditional) ←  pretrained from echonet-dynamic
        ↓                                     (transfer learning: cardiac → lung)
PyTorch training loop                    →  cosine noise schedule, EMA, AMP
        ↓
checkpoints/zea_lung_pocus/              →  loadable via from_pretrained()
        ↓
Web UI / MoCoLUSLungUSGenerator          →  diffusion sampling + scan conversion
```

---

## Build HDF5 Dataset

```bash
./run.sh --dataset --n-per-class 500 --output data/lung_us_moculus.h5
```

Generates train/val/test splits with paired IMU data and grid cell labels across 4 vehicle motion profiles (static, ambulance, helicopter, highway).

---

## Docker

### Build

```bash
docker build -t moculus .
```

### Run

```bash
# With GPU
docker run --gpus all -p 8000:8000 -v ./checkpoints:/app/checkpoints -v ./data:/app/data moculus

# Or use docker compose
docker compose up --build
```

### Offline Mode

The Docker image contains everything needed to run the simulator without internet:

```bash
# Save image for offline transfer
docker save moculus | gzip > moculus.tar.gz

# On offline machine
docker load < moculus.tar.gz
docker run --gpus all -p 8000:8000 moculus
```

---

## Two Deployment Modes

### Mode 1: Online — Training & Development

Connect via VS Code Remote SSH or use the web UI directly:

```bash
./run.sh --web
# → http://localhost:8000
```

Features available in online mode:
- Train new diffusion models
- Generate HDF5 datasets
- Create training previews
- Full simulator with all scenarios

### Mode 2: Offline — In Ambulance with BLE Probe

Deploy via Docker for field use:

1. Transfer the Docker image to the field machine
2. Run: `docker run --gpus all -p 8000:8000 moculus`
3. Open http://localhost:8000 on any device on the local network
4. Use the BLE probe section to connect the physical probe

BLE requires `bleak` and host Bluetooth access. In Docker, use `--privileged` or `--device` flags for Bluetooth.

### Practice vs Testing Mode

| Mode | Use Case |
|------|----------|
| **Practice** | Pick a scenario from the dropdown, examine zones, see all findings. Good for learning BLUE protocol. |
| **Test Me** | Random patient case (hidden scenario). Examine all 8 zones, submit your diagnosis, get instant feedback with BLUE protocol explanation. Each launch generates unique images. |

---

## Project Structure

```
moculus/
  run.sh                          Single entry point for all modes
  Dockerfile                      Multi-stage Docker build (no VNC/PyQt6 needed)
  docker-compose.yml              Docker Compose with GPU support
  requirements.txt                Python dependencies
  static/
    index.html                    Web UI (single-page app)
    style.css                     Dark clinical theme
    app.js                        Frontend logic (WebSocket, canvas rendering)
  src/
    web_server.py                 FastAPI server (WebSocket + REST API)
    clinical_frames.py            Clinically accurate frame generator (10 pathologies)
    train_zea_diffusion.py        Zea DiffusionModel training on clinical frames
    train_diffusion.py            Custom U-Net diffusion training (alternative)
    lung_us_generator.py          Generator with zea diffusion + scan conversion
    lung_us_dataset.py            HDF5 dataset builder (10 classes + IMU)
    simulator_bridge.py           GUI ↔ simulator adapter layer
    poc_image_stack.py            BLUE protocol zones, scenarios, temporal stacks
    training_preview.py           Matplotlib preview renderer
  POCUS CEWIT GitLab/
    ultrasound/US_Image_Reader/GUI/
      main_simulator.py           Legacy PyQt6 GUI (still available via ./run.sh)
  .vscode/
    settings.json                 Python interpreter, env vars
    launch.json                   Debug configs
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
| 7 | Lung Point | Sliding ↔ no sliding transition | Mixed |
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
docker compose up --build
```

### Manual Setup

```bash
python3 -m venv ~/moculus_env
source ~/moculus_env/bin/activate
pip install -r requirements.txt
pip install bleak  # optional, for BLE probe

./run.sh --web
```
