#!/usr/bin/env bash
# MoCoLUS Simulator — Single entry point
# ========================================
# Usage:
#   ./run.sh              Launch GUI (auto-detect display)
#   ./run.sh --vnc        Launch GUI with noVNC viewer (port 6080)
#   ./run.sh --preview    Generate training preview images
#   ./run.sh --train      Train zea diffusion model
#   ./run.sh --dataset    Build HDF5 training dataset
#   --background          Add to --train to run in detached tmux session
#
# VS Code Remote:
#   1. ./run.sh --vnc
#   2. In VS Code PORTS tab, forward port 6080
#   3. Open http://localhost:6080/vnc.html in Simple Browser

set -euo pipefail
cd "$(dirname "$0")"

source ~/moculus_env/bin/activate
export KERAS_BACKEND=torch
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

GUI_SCRIPT="POCUS CEWIT GitLab/ultrasound/US_Image_Reader/GUI/main_simulator.py"
NOVNC_DIR="$HOME/.local/share/novnc"

cleanup() {
    local pids="${XVFB_PID:-} ${VNC_PID:-} ${WS_PID:-}"
    for p in $pids; do kill "$p" 2>/dev/null || true; done
    rm -f /tmp/.X99-lock
}
trap cleanup EXIT

start_xvfb() {
    if ! xdpyinfo -display :99 >/dev/null 2>&1; then
        rm -f /tmp/.X99-lock
        Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX &
        XVFB_PID=$!; sleep 1
    fi
    export DISPLAY=:99
}

start_vnc() {
    start_xvfb
    x11vnc -display :99 -forever -nopw -shared -rfbport 5900 -quiet &
    VNC_PID=$!; sleep 1

    if [[ -d "$NOVNC_DIR" ]]; then
        websockify --web="$NOVNC_DIR" 6080 localhost:5900 &
        WS_PID=$!; sleep 1
        echo ""
        echo "========================================"
        echo "  VNC: localhost:5900"
        echo "  Web: http://localhost:6080/vnc.html"
        echo "  Press Ctrl+C to stop."
        echo "========================================"
        echo ""
    else
        echo "VNC on localhost:5900. noVNC not installed."
        echo "Install: mkdir -p $NOVNC_DIR && wget -qO- https://github.com/novnc/noVNC/archive/refs/tags/v1.5.0.tar.gz | tar xz --strip-components=1 -C $NOVNC_DIR"
    fi
}

case "${1:---auto}" in
    --web)
        echo ""
        echo "========================================"
        echo "  MoCoLUS Web UI"
        echo "  http://localhost:8000"
        echo "  Press Ctrl+C to stop."
        echo "========================================"
        echo ""
        python -m uvicorn src.web_server:app --host 0.0.0.0 --port "${PORT:-8000}"
        ;;
    --docker)
        docker compose up --build
        ;;
    --vnc)
        echo "NOTE: --vnc is deprecated. Use --web for a better experience."
        echo "      ./run.sh --web → http://localhost:8000"
        echo ""
        start_vnc
        python "$GUI_SCRIPT"
        ;;
    --preview)
        python -m src.training_preview "${@:2}"
        ;;
    --train)
        shift
        # Check for --background flag
        bg=false; args=()
        for a in "$@"; do
            if [[ "$a" == "--background" ]]; then bg=true; else args+=("$a"); fi
        done
        if $bg; then
            tmux new-session -d -s moculus-train \
                "cd $(pwd) && source ~/moculus_env/bin/activate && KERAS_BACKEND=torch PYTHONPATH=$(pwd)/src:${PYTHONPATH:-} python -m src.train_zea_diffusion ${args[*]:-}; echo 'Done. Press Enter to close.'; read"
            echo "Training started in background tmux session 'moculus-train'"
            echo "  tmux attach -t moculus-train    # view progress"
            echo "  tmux kill-session -t moculus-train  # stop"
        else
            echo "Training zea DiffusionModel on 10 lung POCUS pathologies..."
            python -m src.train_zea_diffusion "${args[@]:-}"
        fi
        ;;
    --dataset)
        echo "Building HDF5 training dataset..."
        python -m src.lung_us_dataset "${@:2}"
        ;;
    --help|-h)
        echo "Usage: ./run.sh [MODE] [OPTIONS]"
        echo ""
        echo "Modes:"
        echo "  --web       Launch web UI on http://localhost:8000 (recommended)"
        echo "  --docker    Build and run via Docker Compose"
        echo "  (none)      Launch PyQt6 GUI (auto-detect display)"
        echo "  --vnc       Launch PyQt6 GUI with VNC (deprecated, use --web)"
        echo "  --preview   Generate training preview images"
        echo "  --train     Train zea diffusion model on clinical frames"
        echo "  --dataset   Build HDF5 training dataset"
        echo ""
        echo "Training options (--train):"
        echo "  --epochs N          Number of training epochs (default: 100)"
        echo "  --batch-size N      Batch size (default: 4)"
        echo "  --n-per-class N     Samples per class per epoch (default: 200)"
        echo "  --from-scratch      Don't use pretrained echonet weights"
        echo "  --resume PATH       Resume from checkpoint"
        echo "  --sample-only       Generate samples from trained model"
        echo "  --background        Run in tmux (survives SSH disconnect)"
        echo ""
        echo "Dataset options (--dataset):"
        echo "  --n-per-class N     Samples per class (default: 500)"
        echo "  --output PATH       Output HDF5 path"
        ;;
    --auto|*)
        if [[ -n "${DISPLAY:-}" ]] && xdpyinfo >/dev/null 2>&1; then
            python "$GUI_SCRIPT"
        elif command -v Xvfb &>/dev/null; then
            start_xvfb
            echo "Note: GUI rendering on :99 but not visible. Use --vnc for remote access."
            python "$GUI_SCRIPT"
        else
            echo "No display and no Xvfb. Install: sudo apt-get install -y xvfb"
            exit 1
        fi
        ;;
esac
