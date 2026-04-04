#!/bin/bash
# Fine-tune the realistic DDPM with boosted trauma pathology data.
#
# Prerequisites:
#   1. Base training run completed (or reached a good checkpoint)
#   2. Synthetic data generated: python -c "..." (see README or generate_synth_trauma.py)
#   3. metadata.csv updated with synthetic entries
#
# What this does:
#   - Resumes from the best checkpoint of the base run
#   - Uses a lower learning rate (1e-5) for fine-tuning stability
#   - The enriched dataset (with 1500 extra synthetic trauma frames) is
#     automatically loaded since metadata.csv was already updated
#   - WeightedRandomSampler in train_realistic.py upsamples rare classes,
#     so the new synthetic data for classes 1, 5, 7 gets proportionally
#     higher sampling weight
#   - Runs for 50 additional epochs with samples every 5 epochs
#
# Enhanced structural guides:
#   - Pneumothorax (class 1): now generates tension PTX variants with
#     thicker pleural line, more A-lines, darker sub-pleural space,
#     wider rib shadow spacing (hyperexpansion)
#   - Pleural effusion (class 5): 30% chance of echogenic hemothorax
#     variant with dependent debris layering, heterogeneous echoes,
#     and scattered bright foci (clot fragments)
#   - Lung point (class 7): unchanged but 500 more synthetic samples
#
# Usage:
#   # In tmux:
#   tmux new-session -s finetune
#   bash scripts/finetune_trauma.sh
#
#   # Monitor:
#   tail -f checkpoints/realistic_v2_finetune/training.log

set -e

CHECKPOINT="checkpoints/realistic_v2/best.pt"
OUTPUT_DIR="checkpoints/realistic_v2_finetune"

echo "=== Trauma Fine-Tuning ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Output:     ${OUTPUT_DIR}"
echo ""

python -m src.train_realistic \
    --resume "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 250 \
    --batch-size 8 \
    --lr 1e-5 \
    --sample-every 5 \
    --save-every 10
