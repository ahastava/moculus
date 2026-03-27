#!/usr/bin/env bash
# MoCoLUS — Deploy to DockerHub + GitLab
# =======================================
# Usage:
#   ./deploy.sh                 Build, push Docker image, and update GitLab
#   ./deploy.sh --docker-only   Only build and push Docker image
#   ./deploy.sh --git-only      Only push to GitLab
#   ./deploy.sh -m "message"    Custom commit message
#
# Prerequisites:
#   - docker login (DockerHub)
#   - SSH key for git@gitlab.cc.stonybrook.edu

set -euo pipefail
cd "$(dirname "$0")"

DOCKER_IMAGE="ahastava/moculus:latest"
GITLAB_REMOTE="git@gitlab.cc.stonybrook.edu:pocultrasound/ultrasound.git"
GITLAB_BRANCH="moculus-web-simulator"
DOCKERFILE="Dockerfile.cpu"

# Files to sync to GitLab (UI + Docker only, no Arduino/BLE hardware)
SYNC_FILES=(
  src/ static/ configs/ dist/
  Dockerfile.gpu Dockerfile.cpu docker-compose.yml
  requirements.txt run.sh deploy.sh
  .dockerignore .env .gitignore
)

DO_DOCKER=true
DO_GIT=true
COMMIT_MSG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --docker-only) DO_GIT=false; shift ;;
    --git-only) DO_DOCKER=false; shift ;;
    -m) COMMIT_MSG="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Docker Build & Push ──────────────────────────────────────────────
if $DO_DOCKER; then
  echo "══ Building and pushing $DOCKER_IMAGE (amd64 + arm64) ══"

  # Ensure buildx builder exists
  if ! docker buildx inspect multiarch &>/dev/null; then
    echo "Creating multiarch builder..."
    docker buildx create --name multiarch --use --driver docker-container \
      --platform linux/amd64,linux/arm64
    docker buildx inspect multiarch --bootstrap
  fi
  docker buildx use multiarch

  docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -f "$DOCKERFILE" \
    -t "$DOCKER_IMAGE" \
    --push \
    --build-arg CACHEBUST=$(date +%s) \
    .

  echo "✓ Docker image pushed: $DOCKER_IMAGE"
fi

# ── GitLab Push ──────────────────────────────────────────────────────
if $DO_GIT; then
  echo "══ Pushing UI/Docker files to GitLab ══"

  TMPDIR=$(mktemp -d)
  trap "rm -rf $TMPDIR" EXIT

  git clone --single-branch -b "$GITLAB_BRANCH" "$GITLAB_REMOTE" "$TMPDIR" 2>/dev/null \
    || git clone "$GITLAB_REMOTE" "$TMPDIR" 2>/dev/null

  # Switch to feature branch (create if needed)
  git -C "$TMPDIR" checkout "$GITLAB_BRANCH" 2>/dev/null \
    || git -C "$TMPDIR" checkout -b "$GITLAB_BRANCH"

  # Sync files (delete old versions, copy fresh)
  for f in "${SYNC_FILES[@]}"; do
    rm -rf "${TMPDIR:?}/${f}"
    if [ -e "$f" ]; then
      cp -r "$f" "$TMPDIR/$f"
    fi
  done

  # Commit and push
  git -C "$TMPDIR" add -A
  if git -C "$TMPDIR" diff --cached --quiet; then
    echo "No changes to push to GitLab."
  else
    if [ -z "$COMMIT_MSG" ]; then
      COMMIT_MSG="update MoCoLUS web simulator ($(date +%Y-%m-%d))"
    fi
    git -C "$TMPDIR" commit -m "$COMMIT_MSG"
    git -C "$TMPDIR" push origin "$GITLAB_BRANCH"
    echo "✓ Pushed to GitLab: $GITLAB_BRANCH"
  fi
fi

echo "══ Deploy complete ══"
