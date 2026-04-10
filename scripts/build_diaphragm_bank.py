#!/usr/bin/env python3
"""
Build a DiaphragmAnatomyBank from labeled diaphragmatic data (Phase 4).

This script scans `data/real_pocus/processed/metadata.csv` for rows where
both `pathology_class` is in {4, 5, 6, 9} (the diaphragm-relevant lesion
classes) and `zone_region > 0` (lower BLUE / PLAPS / Diaphragm), extracts
texture patches and per-(class, zone) spatial PMFs, and writes the result
to `checkpoints/anatomy_bank_diaphragm.pt`.

The bank is empty until Phase 3 data sourcing adds rows with non-zero
`zone_region` values. When empty, the script still writes a valid
(empty) bank file so callers can `load()` it without checking existence
first — RealisticLungUSGenerator and the train script both gracefully
handle empty banks.

Usage:
    python3 scripts/build_diaphragm_bank.py
    python3 scripts/build_diaphragm_bank.py --max-patches-per-zone 200
    python3 scripts/build_diaphragm_bank.py --output checkpoints/anatomy_bank_diaphragm_v2.pt

Run AFTER:
    1. Phase 3 data sourcing has added diaphragm-labeled rows to metadata.csv
    2. The new images are present in data/real_pocus/processed/images/

Run BEFORE:
    Phase 5 LoRA fine-tuning, so the train script picks up the new bank
    and the dataset returns zone-enhanced guides during training.
"""

import argparse
import sys
from pathlib import Path

# Allow `python3 scripts/build_diaphragm_bank.py` from any working dir
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anatomy_bank import DiaphragmAnatomyBank


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the zone-aware diaphragm anatomy bank from labeled data.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/real_pocus/processed",
        help="Path to the processed POCUS directory containing metadata.csv "
        "and the images/ subdirectory (default: data/real_pocus/processed).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/anatomy_bank_diaphragm.pt",
        help="Output path for the bank file (default: "
        "checkpoints/anatomy_bank_diaphragm.pt).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Texture patch size in pixels (default: 64).",
    )
    parser.add_argument(
        "--max-patches-per-zone",
        type=int,
        default=100,
        help="Max texture patches per (class, zone) pair (default: 100).",
    )
    args = parser.parse_args()

    processed = ROOT / args.processed_dir
    if not processed.exists():
        print(f"ERROR: processed dir not found: {processed}", file=sys.stderr)
        return 1

    print(f"Building DiaphragmAnatomyBank from {processed}")
    print(f"  patch_size: {args.patch_size}")
    print(f"  max_patches_per_zone: {args.max_patches_per_zone}")
    print()

    bank = DiaphragmAnatomyBank.from_dataset(
        processed_dir=str(processed),
        patch_size=args.patch_size,
        max_patches_per_zone=args.max_patches_per_zone,
    )

    print()
    print(bank.summary())
    print()

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    bank.save(str(output))

    size_mb = output.stat().st_size / 1e6
    print(f"Saved to {output} ({size_mb:.2f} MB)")

    if not bank.textures_by_zone:
        print()
        print("NOTE: bank is empty because no rows in metadata.csv have")
        print("      both (pathology_class IN {4, 5, 6, 9}) AND")
        print("      (zone_region > 0). Run Phase 3 data sourcing first.")
        print()
        print("      The empty bank still loads cleanly — MergedAnatomyBank")
        print("      will fall back to the class-only base bank for every")
        print("      lower-zone sample until this bank is populated.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
