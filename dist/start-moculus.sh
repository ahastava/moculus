#!/bin/bash
# MoCoLUS POCUS Trainer — Launcher for macOS / Linux

IMAGE="ahastava/moculus:latest"
CONTAINER="moculus"

echo ""
echo "  ============================================"
echo "   MoCoLUS - Point-of-Care Lung US Simulator"
echo "  ============================================"
echo ""

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    echo "  [!] Docker is not installed."
    echo "      Download it from: https://www.docker.com/products/docker-desktop"
    echo "      Install it, restart your computer, then run this script again."
    echo ""
    exit 1
fi

# Check Docker daemon is running
if ! docker info &> /dev/null; then
    echo "  [!] Docker is not running."
    # Try to start Docker Desktop on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "      Starting Docker Desktop..."
        open -a Docker
    else
        echo "      Please start Docker Desktop, then run this script again."
        exit 1
    fi
    echo "      Waiting for Docker to start (this may take a minute)..."
    while ! docker info &> /dev/null; do
        sleep 3
    done
    echo "      Docker is ready."
    echo ""
fi

# Stop old container if running
if docker ps -q -f name="$CONTAINER" | grep -q .; then
    echo "  Stopping previous session..."
    docker stop "$CONTAINER" > /dev/null 2>&1
    docker rm "$CONTAINER" > /dev/null 2>&1
fi

# Pull latest image
echo "  Downloading latest MoCoLUS (first time may take a few minutes)..."
docker pull "$IMAGE"
if [ $? -ne 0 ]; then
    echo ""
    echo "  [!] Failed to download. Check your internet connection."
    exit 1
fi

# Run container
echo ""
echo "  Starting MoCoLUS..."
docker run -d --name "$CONTAINER" -p 8000:8000 "$IMAGE" > /dev/null 2>&1

# Wait for server to be ready
echo "  Waiting for server to start..."
until curl -s http://localhost:8000/api/scenarios > /dev/null 2>&1; do
    sleep 2
done

echo ""
echo "  ============================================"
echo "   MoCoLUS is running!"
echo "   Opening browser..."
echo ""
echo "   If it doesn't open, go to:"
echo "   http://localhost:8000"
echo ""
echo "   To stop: press Ctrl+C"
echo "  ============================================"
echo ""

# Open browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:8000
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8000
fi

# Keep running, stop on Ctrl+C
trap 'echo ""; echo "  Stopping MoCoLUS..."; docker stop "$CONTAINER" > /dev/null 2>&1; docker rm "$CONTAINER" > /dev/null 2>&1; echo "  Done."; exit 0' INT TERM
echo "  Press Ctrl+C to stop MoCoLUS..."
while true; do sleep 1; done
