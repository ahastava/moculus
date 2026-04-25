# Deploy Cheat Sheet

How to update / push a new version of MoCoLUS — Docker image (DockerHub) and source (GitLab) — from this repo.

## TL;DR

```bash
./deploy.sh                       # Docker build+push  +  GitLab push (the normal path)
./deploy.sh --docker-only         # just the image
./deploy.sh --git-only            # just the repo
./deploy.sh -m "fix probe overlay drift"   # custom commit message
```

## Prereqs (one-time)

```bash
docker login                      # DockerHub creds for ahastava/*
ssh -T git@gitlab.cc.stonybrook.edu   # SSH key registered for the GitLab mirror
ls -lh data/frame_cache.npz       # MUST EXIST — Dockerfile.cpu line 55 COPYs it into the image
```

If `frame_cache.npz` is missing, regenerate before any Docker build:

```bash
python scripts/smart_cache_gen.py --n-frames 16 --ddim-steps 30
```

Note: `origin` may be the `gitlab-sbu:` SSH alias while `deploy.sh` hard-codes `gitlab.cc.stonybrook.edu`. If the deploy clone fails on auth, either add the key to that host in `~/.ssh/config` or edit `GITLAB_REMOTE` in `deploy.sh`.

## What `deploy.sh` actually does

1. **Docker (`deploy.sh` lines 44–65)** — `buildx` builds a multi-arch image (`linux/amd64` + `linux/arm64`) from `Dockerfile.cpu`, tags it `ahastava/moculus:latest`, and `--push`es to DockerHub. Cache-busts with `--build-arg CACHEBUST=$(date +%s)` so the source-copy layer always re-runs even when only code changed.
2. **Git (`deploy.sh` lines 68–101)** — clones the GitLab remote into a temp dir, copies the curated `SYNC_FILES` subset (UI + Docker only — no Arduino/BLE, no `data/`, no `checkpoints/`), commits, and pushes to the `moculus-web-simulator` branch. The temp-dir-clone pattern keeps the public mirror clean of files that shouldn't ship.

## Manual equivalents

**Docker only (multi-arch, what production uses):**
```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -f Dockerfile.cpu -t ahastava/moculus:latest --push \
  --build-arg CACHEBUST=$(date +%s) .
```

**Local single-arch smoke test before the slow multi-arch push:**
```bash
docker build -f Dockerfile.cpu -t moculus:test .
docker run --rm -p 8000:8000 moculus:test
# hit http://localhost:8000 — if probe + frames load, the image is good
```

**Git only (normal feature work on this branch):**
```bash
git status
git add <specific files>          # avoid `git add -A` — repo has gitignored heavy dirs
git commit -m "your message"
git push origin moculus-web-simulator
```

## Cheat-sheet

| Goal | Command |
|---|---|
| Ship UI changes end-to-end | `./deploy.sh` |
| Push image only | `./deploy.sh --docker-only` |
| Push code only | `./deploy.sh --git-only` |
| Local build smoke test | `docker build -f Dockerfile.cpu -t moculus:test . && docker run -p 8000:8000 moculus:test` |
| Regenerate frame cache (required before image build) | `python scripts/smart_cache_gen.py --n-frames 16 --ddim-steps 30` |
| Inspect the multi-arch builder | `docker buildx inspect multiarch` |
| Save image for offline transfer | `docker save ahastava/moculus:latest \| gzip > moculus.tar.gz` |
| Load on target machine | `docker load < moculus.tar.gz && docker run -p 8000:8000 ahastava/moculus:latest` |
