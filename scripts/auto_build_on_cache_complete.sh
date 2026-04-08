#!/bin/bash
# Auto-build Docker image and export portable .tar.gz after cache generation completes.
#
# Usage:
#   nohup bash scripts/auto_build_on_cache_complete.sh &
#   disown
#
# What it does:
#   1. Waits for frame_cache.npz to appear (smart_cache_gen.py to finish)
#   2. Validates the cache file
#   3. Builds Docker CPU image
#   4. Exports portable .tar.gz
#   5. Logs everything to checkpoints/auto_build.log

set -e

LOG="/home/ahastava/moculus/checkpoints/auto_build.log"
CACHE="/home/ahastava/moculus/data/frame_cache.npz"
IMAGE="ahastava/moculus:latest"
EXPORT="/home/ahastava/moculus/dist/moculus-latest.tar.gz"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

cd /home/ahastava/moculus

log "Waiting for cache generation to complete..."
log "Watching: $CACHE"

# Wait for smart_cache_gen.py to finish
while true; do
    # Check if the process is still running
    if ! pgrep -f "smart_cache_gen" > /dev/null 2>&1; then
        if [ -f "$CACHE" ]; then
            log "Cache generation process finished. Cache file exists."
            break
        else
            log "ERROR: Cache gen process ended but no cache file found."
            exit 1
        fi
    fi
    sleep 60
done

# Validate cache
log "Validating cache..."
CACHE_SIZE=$(stat -f%z "$CACHE" 2>/dev/null || stat -c%s "$CACHE" 2>/dev/null)
CACHE_MB=$((CACHE_SIZE / 1024 / 1024))
log "Cache size: ${CACHE_MB} MB"

if [ "$CACHE_MB" -lt 50 ]; then
    log "ERROR: Cache too small (${CACHE_MB} MB). Something went wrong."
    exit 1
fi

ZONE_COUNT=$(python3 -c "
import numpy as np
c = np.load('$CACHE', allow_pickle=True)
n = len([k for k in c.files if k.endswith('/bmode')])
print(n)
" 2>/dev/null)
log "Cache contains $ZONE_COUNT zone stacks"

if [ "$ZONE_COUNT" -ne 120 ]; then
    log "WARNING: Expected 120 zones, got $ZONE_COUNT"
fi

# Build Docker image
log "Building Docker image: $IMAGE"
docker build -f Dockerfile.cpu -t "$IMAGE" . 2>&1 | tee -a "$LOG"
log "Docker build complete"

# Show image size
DOCKER_SIZE=$(docker images "$IMAGE" --format "{{.Size}}")
log "Image size: $DOCKER_SIZE"

# Quick smoke test
log "Smoke testing..."
docker run -d --name moculus-smoke -p 8097:8000 "$IMAGE"
sleep 8
if curl -s http://localhost:8097/api/scenarios | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)} scenarios')" 2>/dev/null; then
    log "Smoke test PASSED"
else
    log "WARNING: Smoke test failed"
fi
docker stop moculus-smoke && docker rm moculus-smoke 2>/dev/null

# Export portable .tar.gz
mkdir -p "$(dirname "$EXPORT")"
log "Exporting portable image to: $EXPORT"
docker save "$IMAGE" | gzip > "$EXPORT"
EXPORT_SIZE=$(stat -f%z "$EXPORT" 2>/dev/null || stat -c%s "$EXPORT" 2>/dev/null)
EXPORT_MB=$((EXPORT_SIZE / 1024 / 1024))
log "Export complete: ${EXPORT_MB} MB"

log ""
log "============================================"
log "  BUILD COMPLETE"
log "============================================"
log "  Docker image:  $IMAGE ($DOCKER_SIZE)"
log "  Portable file: $EXPORT (${EXPORT_MB} MB)"
log ""
log "  To deploy on target device:"
log "    1. Copy $EXPORT to target"
log "    2. docker load < moculus-latest.tar.gz"
log "    3. docker run -p 8000:8000 $IMAGE"
log "============================================"
