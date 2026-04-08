# MoCoLUS — Retraining Guide

How to retrain the diffusion model with new clinical data and deploy updated frames.

---

## Prerequisites

- SSH access to the ThinkStation (`ahastava@thinkstationpgx-9c7e`)
- New POCUS images (grayscale, any resolution — will be resized to 256x256)
- Pathology labels for each image (classes 0–9)

---

## Step 1: Add New Data

Place images in the training directory:

```
data/real_pocus/processed/images/
```

Update the metadata CSV:

```
data/real_pocus/processed/metadata.csv
```

Each row needs these columns:

| Column | Example | Description |
|---|---|---|
| `filename` | `pocus_2500.png` | Image filename (in `images/` directory) |
| `pathology_class` | `4` | Class ID (see table below) |
| `split` | `train` | `train` or `val` (use 85/15 split) |
| `source_dataset` | `NewHospital-2026` | Source identifier for provenance tracking |

### Pathology Classes

| ID | Pathology |
|----|-----------|
| 0 | Normal A-Profile |
| 1 | Pneumothorax |
| 2 | Focal B-Lines |
| 3 | Diffuse B-Lines |
| 4 | Consolidation |
| 5 | Pleural Effusion |
| 6 | ARDS / White Lung |
| 7 | Lung Point |
| 8 | Pleural Thickening |
| 9 | Interstitial Syndrome |

---

## Step 2: Check Class Balance

```bash
python scripts/scan_class_balance.py
```

This shows the per-class distribution. If any class has significantly fewer samples, the training will automatically oversample it via `WeightedRandomSampler`.

If you want to generate synthetic frames for underrepresented classes:

```bash
python scripts/generate_balanced.py --target-per-class 1500
```

---

## Step 3: Retrain

Fine-tune from the current production model:

```bash
nohup python -u -m src.train_realistic \
    --resume checkpoints/realistic_v4_ab/best.pt \
    --finetune \
    --epochs 20 \
    --batch-size 8 \
    --lr 3e-5 \
    --output-dir checkpoints/realistic_v5 \
    > checkpoints/training.log 2>&1 &
disown
```

**Parameters you can adjust:**

| Parameter | Default | When to change |
|---|---|---|
| `--epochs` | 20 | More data → more epochs (up to 50). Watch val_loss plateau. |
| `--lr` | 3e-5 | Lower (1e-5) if fine-tuning on small additions. Higher (1e-4) if retraining from scratch. |
| `--batch-size` | 8 | Increase to 16 if GPU memory allows (faster training). |
| `--finetune` | on | Omit to resume training (keeps optimizer state). Use `--finetune` to reset optimizer (recommended for new data). |

### Monitor Training

```bash
# Live log
tail -f checkpoints/training.log

# Metrics
cat checkpoints/realistic_v5/metrics.csv

# Sample images (generated every 5 epochs)
# Look at: checkpoints/realistic_v5/latest_samples.png
```

### When to Stop

- **val_loss stops decreasing** for 5+ epochs → training is done
- **val_loss increases** while train_loss decreases → overfitting, stop and use earlier checkpoint
- Typical fine-tune: 15–25 epochs on the GB10 GPU (~30 min/epoch)

---

## Step 4: Update Model Path

Edit `src/poc_image_stack.py` line ~1109:

```python
_DEFAULT_MODEL = "checkpoints/realistic_v5/best.pt"    # ← update this
```

If you also retrained the trauma model (classes 1, 5, 7):

```python
_TRAUMA_MODEL = "checkpoints/realistic_v5_trauma/latest.pt"
```

---

## Step 5: Regenerate Frame Cache

This pre-renders all 120 zones (15 scenarios × 8 zones) with the new model:

```bash
nohup python -u scripts/smart_cache_gen.py \
    --n-frames 16 \
    --ddim-steps 30 \
    --max-retries 1 \
    > checkpoints/smart_cache.log 2>&1 &
disown
```

**Takes ~3 hours.** Monitor:

```bash
# Live progress
tail -f checkpoints/smart_cache.log

# Visual dashboard (updates after each zone)
python scripts/cache_dashboard.py
# Open: checkpoints/benchmarks/cache_progress/dashboard.png

# Quick status
grep "SCENARIO COMPLETE" checkpoints/smart_cache.log
```

---

## Step 6: Build and Push Docker Image

After the cache finishes:

```bash
# Build
docker build -f Dockerfile.cpu -t ahastava/moculus:latest .

# Test locally
docker run -d --name moculus-test -p 8090:8000 ahastava/moculus:latest
# Open http://localhost:8090, verify frames look correct
docker stop moculus-test && docker rm moculus-test

# Push to Docker Hub
docker push ahastava/moculus:latest
```

Or use the automated build script (watches for cache completion, builds, tests, exports):

```bash
nohup bash scripts/auto_build_on_cache_complete.sh &
disown
```

---

## Step 7: Update Target Devices

On each target device:

```bash
moculus update
```

This pulls the latest image from Docker Hub and restarts the simulator.

---

## Full Timeline

| Step | Time | Command |
|------|------|---------|
| Add data | Manual | Copy images + update CSV |
| Check balance | ~5 sec | `python scripts/scan_class_balance.py` |
| Retrain | ~10 hours (20 epochs) | `python -m src.train_realistic ...` |
| Update model path | ~1 min | Edit `poc_image_stack.py` |
| Regenerate cache | ~3 hours | `python scripts/smart_cache_gen.py ...` |
| Build + push Docker | ~5 min | `docker build ... && docker push ...` |
| Update devices | ~2 min each | `moculus update` |

---

## Troubleshooting

**Training loss not decreasing:**
- Learning rate too low → try `--lr 1e-4`
- Data issue → check images are grayscale, properly labeled, not corrupted

**Cache generation failures:**
- Check `checkpoints/smart_cache.log` for specific zone failures
- Run `python scripts/cache_dashboard.py` to see visual quality report
- Failed zones are usually B-line/ARDS (expected) or PTX (may need guidance scale tuning)

**Docker image not updating on target:**
- Verify push: `docker pull ahastava/moculus:latest` on target should show new layers
- Check `docker images ahastava/moculus` for the timestamp

**Model path not found after retrain:**
- Verify `checkpoints/realistic_v5/best.pt` exists
- Check you updated `_DEFAULT_MODEL` in `src/poc_image_stack.py`
