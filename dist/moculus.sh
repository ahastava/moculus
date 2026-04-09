#!/bin/bash
# MoCoLUS Simulator — Start/Stop/Update
#
# First time:  curl -fsSL https://get.docker.com | sh
#              (installs Docker)
#
# Usage:
#   ./moculus.sh start     Start the simulator
#   ./moculus.sh stop      Stop the simulator
#   ./moculus.sh update    Pull latest version, restart, and clean old images
#   ./moculus.sh clean     Remove dangling images and stopped containers
#   ./moculus.sh status    Check if running
#   ./moculus.sh           Same as start

IMAGE="ahastava/moculus:latest"
NAME="moculus"
PORT=8000

case "${1:-start}" in
    start)
        if docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
            echo "MoCoLUS is already running at http://localhost:${PORT}"
            exit 0
        fi
        # Remove stopped container if exists
        docker rm "$NAME" 2>/dev/null
        echo "Starting MoCoLUS..."
        docker run -d --name "$NAME" -p ${PORT}:8000 --restart unless-stopped "$IMAGE"
        echo ""
        echo "  MoCoLUS is running at: http://localhost:${PORT}"
        echo "  Stop with: ./moculus.sh stop"
        ;;
    stop)
        echo "Stopping MoCoLUS..."
        docker stop "$NAME" 2>/dev/null
        docker rm "$NAME" 2>/dev/null
        echo "Stopped."
        ;;
    update)
        echo "Pulling latest version..."
        docker pull "$IMAGE"
        echo "Restarting..."
        docker stop "$NAME" 2>/dev/null
        docker rm "$NAME" 2>/dev/null
        docker run -d --name "$NAME" -p ${PORT}:8000 --restart unless-stopped "$IMAGE"
        echo "Cleaning up old images..."
        docker image prune -f >/dev/null
        echo ""
        echo "  Updated and running at: http://localhost:${PORT}"
        ;;
    clean)
        echo "Removing dangling images and stopped containers..."
        docker container prune -f
        docker image prune -f
        echo "Done."
        ;;
    status)
        if docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
            echo "MoCoLUS is running at http://localhost:${PORT}"
            docker ps --filter "name=${NAME}" --format "  Uptime: {{.Status}}"
        else
            echo "MoCoLUS is not running. Start with: ./moculus.sh start"
        fi
        ;;
    *)
        echo "Usage: ./moculus.sh [start|stop|update|clean|status]"
        ;;
esac
