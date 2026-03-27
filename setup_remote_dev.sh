#!/usr/bin/env bash
# =============================================================================
# MoCoLUS Remote Development Setup
# Target: ahastava@thinkstationpgx-9c7e (ThinkStation PX — NVIDIA RTX/A-series)
# =============================================================================
# Run this script FROM YOUR LOCAL MACHINE to configure:
#   1. SSH config for seamless VSCode connection
#   2. Remote environment setup on the ThinkStation
#   3. NVIDIA GPU development tools
#   4. VSCode workspace settings
# =============================================================================

set -euo pipefail

REMOTE_HOST="thinkstationpgx-9c7e"
REMOTE_USER="ahastava"
REMOTE_ALIAS="thinkstation-pgx"
PROJECT_DIR="~/moculus"
ENV_NAME="moculus_env"

# Colors
GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }

# =============================================================================
# STEP 1: Configure local SSH
# =============================================================================
info "Configuring SSH for ${REMOTE_USER}@${REMOTE_HOST}..."

SSH_CONFIG="$HOME/.ssh/config"
ENTRY_MARKER="# MoCoLUS ThinkStation PX"

if ! grep -q "$ENTRY_MARKER" "$SSH_CONFIG" 2>/dev/null; then
    cat >> "$SSH_CONFIG" <<EOF

${ENTRY_MARKER}
Host ${REMOTE_ALIAS}
    HostName ${REMOTE_HOST}
    User ${REMOTE_USER}
    ForwardAgent yes
    ServerAliveInterval 30
    ServerAliveCountMax 20
    Compression yes
EOF
    success "SSH config entry added for ${REMOTE_ALIAS}"
else
    warn "SSH config entry already exists for ${REMOTE_ALIAS}"
fi

# Test connection (disable ControlMaster for the test — stale sockets cause failures on Windows)
info "Testing SSH connection to ${REMOTE_ALIAS}..."
if ssh -o ConnectTimeout=10 -o ControlPath=none "${REMOTE_ALIAS}" "echo 'SSH connection successful'" 2>&1; then
    success "SSH connection verified"
else
    warn "SSH connection test failed. Ensure host is reachable and you have keys set up."
    warn "Run: ssh-copy-id ${REMOTE_USER}@${REMOTE_HOST}"
    exit 1
fi

# =============================================================================
# STEP 2: Remote environment setup
# =============================================================================
info "Setting up remote Python environment on ThinkStation..."

ssh "${REMOTE_ALIAS}" bash <<'REMOTE_SETUP'
set -euo pipefail

PROJECT_DIR="$HOME/moculus"
ENV_NAME="moculus_env"

echo "[REMOTE] Creating project structure..."
mkdir -p "$PROJECT_DIR"/{src/zea_integration,data,notebooks,configs,.vscode}

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "[REMOTE] Python: $PYTHON_VERSION"

# Create virtual environment if not exists
if [ ! -d "$HOME/$ENV_NAME" ]; then
    echo "[REMOTE] Creating virtual environment: $ENV_NAME"
    python3 -m venv "$HOME/$ENV_NAME"
fi

source "$HOME/$ENV_NAME/bin/activate"
export KERAS_BACKEND=torch

# Persist KERAS_BACKEND in venv activation
if ! grep -q "KERAS_BACKEND" "$HOME/$ENV_NAME/bin/activate" 2>/dev/null; then
    echo 'export KERAS_BACKEND=torch' >> "$HOME/$ENV_NAME/bin/activate"
fi

echo "[REMOTE] Upgrading pip..."
pip install --upgrade pip wheel setuptools -q

# Detect NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "[REMOTE] GPU detected: $GPU_INFO (CUDA $CUDA_VERSION)"
    GPU_AVAILABLE=true
else
    echo "[REMOTE] No NVIDIA GPU detected — CPU-only mode"
    GPU_AVAILABLE=false
fi

# Detect CUDA major version and architecture for correct PyTorch index
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
ARCH=$(uname -m)
echo "[REMOTE] System arch: $ARCH, CUDA major: $CUDA_MAJOR"

echo "[REMOTE] Installing core ML stack..."
if [ "$ARCH" = "aarch64" ]; then
    # NVIDIA Grace Blackwell (GB10) / Jetson — use NVIDIA's PyTorch wheel
    echo "[REMOTE] Detected aarch64 — installing PyTorch with CUDA from NVIDIA index..."
    pip install torch torchvision torchaudio --index-url https://pypi.nvidia.com -q \
        || pip install torch torchvision torchaudio --index-url https://developer.download.nvidia.com/compute/redist/jp/v61 -q \
        || { echo "[REMOTE] NVIDIA index failed, trying default PyPI..."; pip install torch torchvision torchaudio -q; }
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
fi
pip install "keras[torch]>=3.0" -q
pip install h5py tqdm scipy numpy matplotlib ipykernel ipywidgets -q

echo "[REMOTE] Installing zea..."
pip install zea -q

echo "[REMOTE] Installing MoCoLUS extras..."
pip install tensorboard wandb einops timm -q

# Register Jupyter kernel
python3 -m ipykernel install --user --name "$ENV_NAME" --display-name "MoCoLUS (zea)"

echo "[REMOTE] Environment setup complete!"
python3 -c "
import torch, keras
print(f'  torch: {torch.__version__}  keras: {keras.__version__}')
try:
    import zea
    print(f'  zea: {zea.__version__}')
except Exception as e:
    # zea viewer requires tkinter (GUI) — OK for headless training
    print(f'  zea import partial (tkinter not available — install python3-tk for full GUI support)')
"

if [ "$GPU_AVAILABLE" = true ]; then
    python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}  GPUs: {torch.cuda.device_count()}')"
fi

REMOTE_SETUP

success "Remote environment configured"

# =============================================================================
# STEP 3: Copy project files to remote
# =============================================================================
info "Syncing project files to ThinkStation..."

# Create local .env if not exists
if [ ! -f .env ]; then
    cat > .env <<'EOF'
KERAS_BACKEND=torch
CUDA_VISIBLE_DEVICES=0
PYTHONPATH=./src
ZEA_DATA_DIR=./data
ZEA_MODEL_DIR=./models
WANDB_MODE=offline
EOF
    success "Created .env"
fi

# Sync source files (use scp on Windows where rsync may not be available)
if command -v rsync &>/dev/null; then
    rsync -av --progress \
        ./ \
        "${REMOTE_ALIAS}:${PROJECT_DIR}/" \
        --exclude="__pycache__" \
        --exclude="*.pyc" \
        --exclude=".git" \
        --exclude="data/*.h5" \
        --exclude="*.egg-info" \
        --exclude=".idea"
else
    info "rsync not available, using scp..."
    # Copy key project files via scp
    for f in *.py *.yaml *.sh .env; do
        [ -f "$f" ] && scp "$f" "${REMOTE_ALIAS}:${PROJECT_DIR}/"
    done
    # Copy notebooks
    for f in *.ipynb; do
        [ -f "$f" ] && scp "$f" "${REMOTE_ALIAS}:${PROJECT_DIR}/"
    done
fi

success "Files synced to remote"

# =============================================================================
# STEP 4: Write VSCode workspace settings
# =============================================================================
info "Writing VSCode workspace settings..."

mkdir -p .vscode

cat > .vscode/settings.json <<EOF
{
    "python.defaultInterpreterPath": "~/moculus_env/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.envFile": "\${workspaceFolder}/.env",
    "python.analysis.extraPaths": ["\${workspaceFolder}/src"],
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,

    "jupyter.notebookFileRoot": "\${workspaceFolder}",
    "jupyter.kernelSpecFindPaths": [],

    "terminal.integrated.env.linux": {
        "KERAS_BACKEND": "torch",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": "\${workspaceFolder}/src",
        "ZEA_DATA_DIR": "\${workspaceFolder}/data",
        "ZEA_MODEL_DIR": "\${workspaceFolder}/models"
    },

    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.ipynb_checkpoints": true
    },

    "editor.rulers": [88],
    "editor.formatOnSave": true,

    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter"
    }
}
EOF

cat > .vscode/launch.json <<'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Train Lung US Diffusion",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/zea_integration/train_diffusion.py",
            "args": ["--config", "configs/lung_us_diffusion.yaml"],
            "console": "integratedTerminal",
            "env": {
                "KERAS_BACKEND": "torch",
                "CUDA_VISIBLE_DEVICES": "0"
            }
        },
        {
            "name": "Build Lung US Dataset",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/zea_integration/lung_us_dataset.py",
            "args": [
                "--n-per-class", "500",
                "--output", "data/lung_us_moculus.h5",
                "--vehicle-types", "static", "ambulance_ground", "helicopter_hover"
            ],
            "console": "integratedTerminal"
        },
        {
            "name": "Test Generator (Physics Only)",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/zea_integration/lung_us_generator.py",
            "console": "integratedTerminal",
            "env": {"KERAS_BACKEND": "torch"}
        }
    ]
}
EOF

cat > .vscode/extensions.json <<'EOF'
{
    "recommendations": [
        "ms-vscode-remote.remote-ssh",
        "ms-vscode-remote.remote-ssh-edit",
        "ms-python.python",
        "ms-python.debugpy",
        "ms-python.black-formatter",
        "ms-toolsai.jupyter",
        "ms-toolsai.jupyter-renderers",
        "ms-toolsai.vscode-jupyter-slideshow",
        "eamodio.gitlens",
        "nvidia.nsight-vscode-edition"
    ]
}
EOF

success "VSCode workspace settings written"

# Sync updated .vscode to remote
if command -v rsync &>/dev/null; then
    rsync -av .vscode/ "${REMOTE_ALIAS}:${PROJECT_DIR}/.vscode/"
else
    scp .vscode/*.json "${REMOTE_ALIAS}:${PROJECT_DIR}/.vscode/"
fi

# =============================================================================
# STEP 5: NVIDIA Nsight / GPU verification on remote
# =============================================================================
info "Verifying NVIDIA GPU setup on ThinkStation..."

ssh "${REMOTE_ALIAS}" bash <<'GPU_CHECK'
source ~/moculus_env/bin/activate
export KERAS_BACKEND=torch

echo "=== NVIDIA-SMI ==="
nvidia-smi 2>/dev/null || echo "nvidia-smi not found"

echo ""
echo "=== PyTorch CUDA ==="
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1e9
        print(f'  GPU {i}: {props.name} | {mem_gb:.1f} GB | CUDA {props.major}.{props.minor}')
        print(f'  Compute Capability: sm_{props.major}{props.minor}')
"

echo ""
echo "=== zea import check ==="
python3 -c "
import keras
print(f'Keras backend: {keras.backend.backend()}')
try:
    import zea
    from zea.models.diffusion import DiffusionModel
    print(f'zea version: {zea.__version__}')
    print('All imports OK')
except ImportError as e:
    if 'tkinter' in str(e):
        print('zea core installed (tkinter GUI not available — OK for headless training)')
        print('To fix: sudo apt install python3-tk')
    else:
        raise
"

GPU_CHECK

success "GPU setup verified"

# =============================================================================
# STEP 6: Print VSCode connection instructions
# =============================================================================
echo ""
echo "================================================================"
echo -e "  ${GREEN}MoCoLUS Remote Dev Environment Ready!${NC}"
echo "================================================================"
echo ""
echo "  To connect in VSCode:"
echo "  1. Press Ctrl+Shift+P → 'Remote-SSH: Connect to Host'"
echo "  2. Select: ${REMOTE_ALIAS}"
echo "  3. Open folder: ${PROJECT_DIR}"
echo ""
echo "  NVIDIA Nsight GPU Profiling:"
echo "  - Install 'nvidia.nsight-vscode-edition' extension on remote"
echo "  - F5 with 'Train Lung US Diffusion' launch config"
echo "  - Use Nsight Systems: nsys profile python train_diffusion.py"
echo ""
echo "  Quick start:"
echo "  ssh ${REMOTE_ALIAS}"
echo "  source ~/moculus_env/bin/activate"
echo "  cd ${PROJECT_DIR}"
echo "  python src/zea_integration/lung_us_dataset.py --n-per-class 100"
echo ""
echo "================================================================"
