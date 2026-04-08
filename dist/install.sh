#!/bin/bash
# MoCoLUS Simulator — One-Line Installer
#
# Installs Docker (if needed), pulls the simulator image, creates
# desktop shortcuts and the moculus command.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ahastava/moculus/main/dist/install.sh | bash
#
# Or download and run:
#   chmod +x install.sh && ./install.sh

set -e

IMAGE="ahastava/moculus:latest"
NAME="moculus"
PORT=8000

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║   MoCoLUS POCUS Simulator Installer  ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# ── 1. Install Docker if needed ──
if ! command -v docker &> /dev/null; then
    echo "[1/4] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "  Docker installed. You may need to log out and back in for group permissions."
else
    echo "[1/4] Docker already installed."
fi

# ── 2. Pull simulator image ──
echo "[2/4] Pulling MoCoLUS image (~600 MB)..."
docker pull "$IMAGE"

# ── 3. Install moculus command ──
echo "[3/4] Installing 'moculus' command..."

SCRIPT_DIR="$HOME/.local/bin"
mkdir -p "$SCRIPT_DIR"

cat > "$SCRIPT_DIR/moculus" << 'SCRIPT'
#!/bin/bash
IMAGE="ahastava/moculus:latest"
NAME="moculus"
PORT=8000

case "${1:-start}" in
    start)
        if docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
            echo "MoCoLUS is already running at http://localhost:${PORT}"
            exit 0
        fi
        docker rm "$NAME" 2>/dev/null
        echo "Starting MoCoLUS..."
        docker run -d --name "$NAME" -p ${PORT}:8000 --restart unless-stopped "$IMAGE" > /dev/null
        sleep 2
        echo "MoCoLUS is running at: http://localhost:${PORT}"
        xdg-open "http://localhost:${PORT}" 2>/dev/null || open "http://localhost:${PORT}" 2>/dev/null || true
        ;;
    stop)
        echo "Stopping MoCoLUS..."
        docker stop "$NAME" 2>/dev/null && docker rm "$NAME" 2>/dev/null
        echo "Stopped."
        ;;
    update)
        echo "Pulling latest version..."
        docker pull "$IMAGE"
        docker stop "$NAME" 2>/dev/null; docker rm "$NAME" 2>/dev/null
        docker run -d --name "$NAME" -p ${PORT}:8000 --restart unless-stopped "$IMAGE" > /dev/null
        sleep 2
        echo "Updated and running at: http://localhost:${PORT}"
        xdg-open "http://localhost:${PORT}" 2>/dev/null || open "http://localhost:${PORT}" 2>/dev/null || true
        ;;
    status)
        if docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
            echo "MoCoLUS is running at http://localhost:${PORT}"
            docker ps --filter "name=${NAME}" --format "  Uptime: {{.Status}}"
        else
            echo "MoCoLUS is not running."
            echo "  Start: moculus start"
        fi
        ;;
    *)
        echo "MoCoLUS Simulator"
        echo "  moculus start    Start simulator (opens browser)"
        echo "  moculus stop     Stop simulator"
        echo "  moculus update   Pull latest version and restart"
        echo "  moculus status   Check if running"
        ;;
esac
SCRIPT

chmod +x "$SCRIPT_DIR/moculus"

# Ensure ~/.local/bin is in PATH
if ! echo "$PATH" | grep -q "$SCRIPT_DIR"; then
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [ -f "$rc" ]; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$rc"
        fi
    done
    export PATH="$SCRIPT_DIR:$PATH"
fi

# ── 4. Create desktop shortcut ──
echo "[4/4] Creating desktop shortcut..."

DESKTOP_DIR="$HOME/Desktop"
APPLICATIONS_DIR="$HOME/.local/share/applications"
mkdir -p "$APPLICATIONS_DIR"

# Download icon (or use a simple embedded one)
ICON_DIR="$HOME/.local/share/icons"
mkdir -p "$ICON_DIR"

# Generate a simple icon using Python if available
python3 -c "
from PIL import Image, ImageDraw, ImageFont
img = Image.new('RGBA', (256, 256), (20, 25, 40, 255))
draw = ImageDraw.Draw(img)
# Lung-shaped oval
draw.ellipse([40, 50, 120, 200], fill=(50, 160, 220, 200))
draw.ellipse([136, 50, 216, 200], fill=(50, 160, 220, 200))
# Pleural line
draw.line([(30, 130), (226, 130)], fill=(255, 255, 255, 200), width=4)
# Text
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 28)
except: font = ImageFont.load_default()
draw.text((48, 210), 'MoCoLUS', fill=(200, 215, 245), font=font)
img.save('$ICON_DIR/moculus.png')
" 2>/dev/null || true

# .desktop file (works on Linux with GNOME, KDE, XFCE, etc.)
DESKTOP_ENTRY="[Desktop Entry]
Name=MoCoLUS Simulator
Comment=Lung Ultrasound POCUS Training Simulator
Exec=bash -c 'docker start moculus 2>/dev/null || docker run -d --name moculus -p 8000:8000 --restart unless-stopped ahastava/moculus:latest; sleep 2; xdg-open http://localhost:8000'
Icon=$ICON_DIR/moculus.png
Type=Application
Categories=Education;Science;
Terminal=false
StartupNotify=true"

# Install to applications menu
echo "$DESKTOP_ENTRY" > "$APPLICATIONS_DIR/moculus.desktop"
chmod +x "$APPLICATIONS_DIR/moculus.desktop"

# Copy to Desktop if it exists
if [ -d "$DESKTOP_DIR" ]; then
    echo "$DESKTOP_ENTRY" > "$DESKTOP_DIR/MoCoLUS Simulator.desktop"
    chmod +x "$DESKTOP_DIR/MoCoLUS Simulator.desktop"
    # Mark as trusted on GNOME
    gio set "$DESKTOP_DIR/MoCoLUS Simulator.desktop" metadata::trusted true 2>/dev/null || true
fi

# macOS .command file (if on Mac)
if [[ "$OSTYPE" == "darwin"* ]]; then
    cat > "$DESKTOP_DIR/MoCoLUS Simulator.command" << 'MAC'
#!/bin/bash
docker start moculus 2>/dev/null || docker run -d --name moculus -p 8000:8000 --restart unless-stopped ahastava/moculus:latest
sleep 2
open http://localhost:8000
MAC
    chmod +x "$DESKTOP_DIR/MoCoLUS Simulator.command"
fi

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║         INSTALL COMPLETE             ║"
echo "  ╠══════════════════════════════════════╣"
echo "  ║                                      ║"
echo "  ║  Desktop shortcut created            ║"
echo "  ║  Terminal command: moculus start      ║"
echo "  ║                                      ║"
echo "  ║  Double-click the desktop icon or    ║"
echo "  ║  type 'moculus start' to launch.     ║"
echo "  ║                                      ║"
echo "  ╚══════════════════════════════════════╝"
echo ""
