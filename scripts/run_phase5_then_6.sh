#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Chain script: Phase 5 LoRA fine-tune → Phase 6 cache regen → Docker build
# ─────────────────────────────────────────────────────────────────────────
#
# Runs the full GPU pipeline end-to-end without manual intervention.
# Designed to be launched inside tmux so it survives SSH disconnects:
#
#     tmux new -s moculus_gpu
#     ./scripts/run_phase5_then_6.sh
#     # Ctrl+B then D to detach
#
# Reattach later: tmux attach -t moculus_gpu
#
# Each phase only proceeds if the previous one wrote its expected
# output. Logs are tee'd to checkpoints/phase56_run.log so you can
# inspect them after the fact.
#
# Total expected wall-clock on GB10:
#   Phase 5 (LoRA fine-tune):  ~4 hours
#   Phase 6 (cache regen):     ~2 hours
#   Phase 7 (docker build):    ~10-30 minutes
#   ─────────────────────────────────────
#   Total:                     ~6-7 hours

set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
LOG="$ROOT/checkpoints/phase56_run.log"
mkdir -p "$ROOT/checkpoints"

LORA_OUT="checkpoints/realistic_v2_diaphragm_lora"
LORA_DELTA="$LORA_OUT/latest_lora.pt"

log() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$ts] $*" | tee -a "$LOG"
}

log "════════════════════════════════════════════════════════════════"
log "Phase 5 + 6 chain run starting"
log "  ROOT:       $ROOT"
log "  LORA_OUT:   $LORA_OUT"
log "  LOG:        $LOG"
log "════════════════════════════════════════════════════════════════"

# ─ Pre-flight checks ─
if [[ ! -f "checkpoints/realistic_v2_finetune/latest.pt" ]]; then
    log "ERROR: base checkpoint not found at checkpoints/realistic_v2_finetune/latest.pt"
    exit 1
fi
if [[ ! -f "checkpoints/anatomy_bank.pt" ]]; then
    log "ERROR: base anatomy bank not found at checkpoints/anatomy_bank.pt"
    exit 1
fi
if [[ ! -f "checkpoints/anatomy_bank_diaphragm.pt" ]]; then
    log "ERROR: diaphragm bank not found — run scripts/build_diaphragm_bank.py first"
    exit 1
fi
if [[ ! -f "data/real_pocus/processed/metadata.csv" ]]; then
    log "ERROR: training metadata.csv not found"
    exit 1
fi

DIAPHRAGM_ROWS=$(awk -F, 'NR>1 && $8 != "0" && $8 != "" {n++} END {print n+0}' data/real_pocus/processed/metadata.csv)
log "Pre-flight: $DIAPHRAGM_ROWS rows with zone_region > 0 in metadata.csv"
if (( DIAPHRAGM_ROWS == 0 )); then
    log "ERROR: no diaphragm-labeled rows in metadata.csv. Run the ingest pipeline first."
    exit 1
fi

# ─ Phase 5: LoRA fine-tune ─
log ""
log "════════════════════════════════════════════════════════════════"
log "Phase 5 — LoRA fine-tune (~4h on GB10)"
log "════════════════════════════════════════════════════════════════"
P5_START=$(date +%s)

python3 -m src.train_realistic \
    --resume checkpoints/realistic_v2_finetune/latest.pt \
    --finetune \
    --lora --lora-rank 16 --lora-alpha 32 \
    --output-dir "$LORA_OUT" \
    --epochs 20 --batch-size 16 --lr 3e-4 \
    --cfg-dropout 0.10 --sample-every 3 --save-every 5 2>&1 | tee -a "$LOG"

P5_EXIT=${PIPESTATUS[0]}
P5_END=$(date +%s)
P5_DURATION=$(( (P5_END - P5_START) / 60 ))
log ""
log "Phase 5 finished with exit $P5_EXIT after ${P5_DURATION} min"

if (( P5_EXIT != 0 )); then
    log "ERROR: Phase 5 training failed. Aborting before Phase 6."
    exit 1
fi
if [[ ! -f "$LORA_DELTA" ]]; then
    log "ERROR: Phase 5 didn't write $LORA_DELTA. Aborting."
    exit 1
fi

DELTA_SIZE_MB=$(stat -c %s "$LORA_DELTA" 2>/dev/null || echo 0)
DELTA_SIZE_MB=$(( DELTA_SIZE_MB / 1024 / 1024 ))
log "LoRA delta saved: $LORA_DELTA (${DELTA_SIZE_MB} MB)"

# ─ Phase 6: cache regeneration ─
log ""
log "════════════════════════════════════════════════════════════════"
log "Phase 6 — cache regen for lower zones (~2h on GB10)"
log "════════════════════════════════════════════════════════════════"
P6_START=$(date +%s)

# Back up the current cache so we can byte-compare upper-zone entries afterwards
if [[ -f "data/frame_cache.npz" ]]; then
    cp "data/frame_cache.npz" "data/frame_cache.npz.before_phase6.bak"
    log "Backed up existing cache to data/frame_cache.npz.before_phase6.bak"
fi

python3 scripts/smart_cache_gen.py \
    --only-zones LOWER_BLUE_L,LOWER_BLUE_R,PLAPS_L,PLAPS_R,DIAPHRAGM_L,DIAPHRAGM_R \
    --diaphragm-model "$LORA_DELTA" \
    --max-retries 3 \
    --n-frames 16 --ddim-steps 30 2>&1 | tee -a "$LOG"

P6_EXIT=${PIPESTATUS[0]}
P6_END=$(date +%s)
P6_DURATION=$(( (P6_END - P6_START) / 60 ))
log ""
log "Phase 6 finished with exit $P6_EXIT after ${P6_DURATION} min"

if (( P6_EXIT != 0 )); then
    log "ERROR: Phase 6 cache regen failed. Aborting before Docker build."
    exit 1
fi

# ─ Verify upper-zone byte-equality (regression check) ─
log ""
log "Validating upper-zone byte equality..."
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import numpy as np
import sys
from pathlib import Path

old_path = Path("data/frame_cache.npz.before_phase6.bak")
new_path = Path("data/frame_cache.npz")
if not old_path.exists():
    print("  WARNING: no backup to compare against; skipping check")
    sys.exit(0)

old = np.load(old_path, allow_pickle=True)
new = np.load(new_path, allow_pickle=True)

mismatches = 0
checked = 0
for key in old.files:
    if "UPPER_BLUE" in key and key.endswith(("/bmode", "/mmode")):
        checked += 1
        if not np.array_equal(old[key], new[key]):
            mismatches += 1
            print(f"  MISMATCH: {key}")

print(f"Checked {checked} upper-zone entries, {mismatches} mismatches")
if mismatches > 0:
    print("FAIL: upper zone regression detected")
    sys.exit(1)
print("PASS: upper zones byte-identical")
PY

VALIDATE_EXIT=${PIPESTATUS[0]}
if (( VALIDATE_EXIT != 0 )); then
    log "ERROR: upper-zone byte equality check failed. Aborting Docker build."
    exit 1
fi

# ─ Phase 7: Docker build (optional but recommended) ─
log ""
log "════════════════════════════════════════════════════════════════"
log "Phase 7 — Docker build + export"
log "════════════════════════════════════════════════════════════════"
P7_START=$(date +%s)

if [[ -f "scripts/auto_build_on_cache_complete.sh" ]]; then
    bash scripts/auto_build_on_cache_complete.sh 2>&1 | tee -a "$LOG"
    P7_EXIT=${PIPESTATUS[0]}
else
    log "scripts/auto_build_on_cache_complete.sh not found, skipping Docker build"
    P7_EXIT=0
fi

P7_END=$(date +%s)
P7_DURATION=$(( (P7_END - P7_START) / 60 ))
log ""
log "Phase 7 finished with exit $P7_EXIT after ${P7_DURATION} min"

# ─ Summary ─
TOTAL_DURATION=$(( (P7_END - P5_START) / 60 ))
log ""
log "════════════════════════════════════════════════════════════════"
log "Chain run complete"
log "  Phase 5: ${P5_DURATION} min"
log "  Phase 6: ${P6_DURATION} min"
log "  Phase 7: ${P7_DURATION} min"
log "  Total:   ${TOTAL_DURATION} min"
log "════════════════════════════════════════════════════════════════"
log ""
log "Outputs:"
log "  LoRA delta:     $LORA_DELTA"
log "  Updated cache:  data/frame_cache.npz"
log "  Old cache:      data/frame_cache.npz.before_phase6.bak"
[[ -f dist/moculus-latest.tar.gz ]] && log "  Docker image:   dist/moculus-latest.tar.gz"
log ""
log "Log file: $LOG"
