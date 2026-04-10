#!/usr/bin/env python3
"""
Merge labeled diaphragm candidates into the main metadata.csv (Phase 3).

Workflow context
----------------
This script is the third step in the Phase 3 data sourcing pipeline:

  1. python3 -m src.acquire_diaphragm_data --staging ... --output ...
       → produces data/diaphragm_candidates/diap_cand_*.png + manifest.csv

  2. Open scripts/labeling_review.html in your browser, load the
     candidates, label each (pathology_class, zone_region), download
     the resulting diaphragm_labels.csv

  3. Run THIS script to:
       a. Add the `zone_region` column to data/real_pocus/processed/metadata.csv
          if it doesn't already exist (one-time migration of legacy rows
          to zone_region=0)
       b. Copy each labeled candidate PNG into data/real_pocus/processed/images/
          with a stable `diaphragm_NNNNN.png` filename
       c. Append a new row to metadata.csv for each labeled candidate
       d. Print a summary of (class, zone) coverage

  4. Run scripts/build_diaphragm_bank.py to rebuild the
     anatomy_bank_diaphragm.pt with the new patches.

  5. Run the Phase 5 LoRA fine-tune to train on the new data.

Idempotent behavior
-------------------
* Re-running on the same labels CSV is safe — already-merged rows are
  detected by their content-addressed filename and skipped.
* The zone_region column migration on metadata.csv is also idempotent;
  it's a no-op once the column exists.

Validation
----------
* Refuses to merge rows where pathology_class or zone_region is empty
* Refuses to merge if the candidate file is missing from the candidates dir
* Validates pathology_class is in [0..9] and zone_region is in [1..6]
* Reports per-(class, zone) counts so you can see coverage gaps

Usage
-----
    python3 scripts/merge_diaphragm_labels.py \\
        --labels data/diaphragm_candidates/diaphragm_labels.csv \\
        --candidates-dir data/diaphragm_candidates \\
        --processed-dir data/real_pocus/processed
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

# Must match clinical_frames.py / anatomy_bank.py
PATHOLOGY_CLASS_NAMES = {
    0: "normal_a_profile",
    1: "pneumothorax",
    2: "b_lines_focal",
    3: "b_lines_diffuse",
    4: "consolidation",
    5: "pleural_effusion",
    6: "ards_white_lung",
    7: "lung_point",
    8: "pleural_thickening",
    9: "interstitial_syndrome",
}

ZONE_REGION_NAMES = {
    1: "LOWER_BLUE_L",
    2: "LOWER_BLUE_R",
    3: "PLAPS_L",
    4: "PLAPS_R",
    5: "DIAPHRAGM_L",
    6: "DIAPHRAGM_R",
}

# Standard MoCoLUS metadata schema (post-Phase-3)
EXPECTED_FIELDS = [
    "filename",
    "pathology_class",
    "pathology_name",
    "source_label",
    "source_file",
    "source_dataset",
    "split",
    "zone_region",
]


def migrate_metadata_add_zone_region(metadata_path: Path) -> bool:
    """
    Ensure metadata.csv has the `zone_region` column. If not, add it
    with default value 0 for every existing row. Idempotent.

    Returns True if the file was modified, False if no migration needed.
    """
    with open(metadata_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"{metadata_path} has no header")
        if "zone_region" in reader.fieldnames:
            return False
        old_rows = list(reader)
        old_fields = list(reader.fieldnames)

    new_fields = old_fields + ["zone_region"]
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        for row in old_rows:
            row["zone_region"] = "0"
            writer.writerow(row)
    return True


def load_existing_source_files(metadata_path: Path) -> set:
    """
    Return the set of `source_file` values already present in metadata.csv.

    This is the idempotency key for re-running the merge. The destination
    filename for a new row is `diaphragm_NNNNN.png`, but we track the
    *original* candidate filename in the `source_file` column. So if the
    user re-runs the merge with the same labels CSV, we can detect the
    overlap by checking each row's filename against the set of already-
    merged source_file values.
    """
    sources: set = set()
    with open(metadata_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("source_file", "")
            if src:
                sources.add(src)
    return sources


def next_diaphragm_index(images_dir: Path) -> int:
    """Find the next free `diaphragm_NNNNN.png` index in the images dir."""
    existing = sorted(images_dir.glob("diaphragm_*.png"))
    if not existing:
        return 0
    last = existing[-1].stem  # diaphragm_00042
    try:
        return int(last.split("_", 1)[1]) + 1
    except (IndexError, ValueError):
        return len(existing)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to the diaphragm_labels.csv produced by labeling_review.html",
    )
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        required=True,
        help="Directory containing the diap_cand_*.png files (output of "
        "src.acquire_diaphragm_data).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "real_pocus" / "processed",
        help="Path to the processed POCUS directory (default: "
        "data/real_pocus/processed).",
    )
    parser.add_argument(
        "--source-dataset",
        type=str,
        default="MoCoLUS-Diaphragm",
        help="Value to write into the source_dataset column for new rows.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val"),
        help="Train/val split assignment for new rows (default: train).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the labels CSV and report what would be merged "
        "without writing anything.",
    )
    args = parser.parse_args(argv)

    labels_path = args.labels.resolve()
    candidates_dir = args.candidates_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    metadata_path = processed_dir / "metadata.csv"
    images_dir = processed_dir / "images"

    if not labels_path.exists():
        print(f"ERROR: labels CSV not found: {labels_path}", file=sys.stderr)
        return 1
    if not candidates_dir.exists():
        print(f"ERROR: candidates dir not found: {candidates_dir}", file=sys.stderr)
        return 1
    if not metadata_path.exists():
        print(f"ERROR: metadata.csv not found: {metadata_path}", file=sys.stderr)
        return 1
    if not images_dir.exists():
        print(f"ERROR: images dir not found: {images_dir}", file=sys.stderr)
        return 1

    # ── 1. Migrate metadata.csv to have zone_region column ──
    print(f"Step 1: ensure metadata.csv has zone_region column")
    if args.dry_run:
        with open(metadata_path) as f:
            reader = csv.DictReader(f)
            has_col = "zone_region" in (reader.fieldnames or [])
        print(f"  zone_region column present: {has_col} (dry-run, no changes)")
    else:
        migrated = migrate_metadata_add_zone_region(metadata_path)
        if migrated:
            print(f"  Migrated {metadata_path}: added zone_region column "
                  f"(default 0 for all existing rows)")
        else:
            print(f"  zone_region column already present, no migration needed")

    # ── 2. Read labeled rows ──
    print()
    print(f"Step 2: read labeled rows from {labels_path}")
    with open(labels_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    valid_rows: List[Dict[str, str]] = []
    invalid_rows: List[Tuple[Dict[str, str], str]] = []
    for row in all_rows:
        if row.get("status") != "labeled":
            continue  # skip blank or 'skipped'
        try:
            cls_id = int(row["pathology_class"])
            zone_id = int(row["zone_region"])
        except (TypeError, ValueError):
            invalid_rows.append((row, "non-integer class/zone"))
            continue
        if cls_id not in PATHOLOGY_CLASS_NAMES:
            invalid_rows.append((row, f"class {cls_id} out of range"))
            continue
        if zone_id not in ZONE_REGION_NAMES:
            invalid_rows.append((row, f"zone {zone_id} out of range"))
            continue
        if not row.get("filename"):
            invalid_rows.append((row, "missing filename"))
            continue
        valid_rows.append(row)

    print(f"  {len(all_rows)} total rows in labels CSV")
    print(f"  {len(valid_rows)} valid labeled rows")
    if invalid_rows:
        print(f"  {len(invalid_rows)} invalid rows (skipped):")
        for row, reason in invalid_rows[:10]:
            print(f"    - {row.get('filename', '?')}: {reason}")
        if len(invalid_rows) > 10:
            print(f"    ... and {len(invalid_rows) - 10} more")

    # ── 3. Verify candidate files exist + filter dupes already in metadata ──
    print()
    print(f"Step 3: verify candidate files + check for duplicates")
    existing_sources = load_existing_source_files(metadata_path)
    to_merge: List[Tuple[Dict[str, str], Path]] = []
    missing_files: List[str] = []
    skipped_dupes = 0
    for row in valid_rows:
        cand_path = candidates_dir / row["filename"]
        if not cand_path.exists():
            missing_files.append(row["filename"])
            continue
        # Idempotency: the candidate filename is recorded in the `source_file`
        # column of any previously-merged row, so we check that set here.
        # Re-running with the same labels CSV is a no-op.
        if row["filename"] in existing_sources:
            skipped_dupes += 1
            continue
        to_merge.append((row, cand_path))

    if skipped_dupes:
        print(f"  {skipped_dupes} rows already merged in a previous run (idempotent skip)")

    if missing_files:
        print(f"  WARNING: {len(missing_files)} candidate files missing from {candidates_dir}:")
        for f in missing_files[:5]:
            print(f"    - {f}")

    print(f"  {len(to_merge)} new rows ready to merge")
    if not to_merge:
        print()
        print("Nothing to do.")
        return 0

    # ── 4. Per (class, zone) coverage report ──
    print()
    print("Step 4: coverage report")
    coverage: Dict[Tuple[int, int], int] = {}
    for row, _ in to_merge:
        key = (int(row["pathology_class"]), int(row["zone_region"]))
        coverage[key] = coverage.get(key, 0) + 1
    print(f"  {len(coverage)} unique (class, zone) pairs:")
    for (cls_id, zone_id), count in sorted(coverage.items()):
        print(
            f"    class {cls_id} ({PATHOLOGY_CLASS_NAMES[cls_id]:>22}) "
            f"× zone {zone_id} ({ZONE_REGION_NAMES[zone_id]:>14}): {count} frames"
        )

    if args.dry_run:
        print()
        print("Dry-run complete — no files written.")
        return 0

    # ── 5. Copy images + append to metadata.csv ──
    print()
    print(f"Step 5: copy images to {images_dir} + append to {metadata_path}")
    next_idx = next_diaphragm_index(images_dir)
    appended = 0
    with open(metadata_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_FIELDS)
        for row, cand_path in to_merge:
            new_name = f"diaphragm_{next_idx:05d}.png"
            dest = images_dir / new_name
            shutil.copy2(cand_path, dest)

            cls_id = int(row["pathology_class"])
            zone_id = int(row["zone_region"])
            writer.writerow({
                "filename": new_name,
                "pathology_class": cls_id,
                "pathology_name": PATHOLOGY_CLASS_NAMES[cls_id],
                "source_label": ZONE_REGION_NAMES[zone_id],
                "source_file": row["filename"],  # original candidate name
                "source_dataset": args.source_dataset,
                "split": args.split,
                "zone_region": zone_id,
            })
            next_idx += 1
            appended += 1

    print(f"  Appended {appended} new rows")
    print()
    print("Next steps:")
    print("  1. python3 scripts/build_diaphragm_bank.py")
    print("       → rebuild checkpoints/anatomy_bank_diaphragm.pt with new patches")
    print()
    print("  2. python3 -m src.train_realistic \\")
    print("       --resume checkpoints/realistic_v2_finetune/latest.pt \\")
    print("       --finetune --lora --lora-rank 16 --lora-alpha 32 \\")
    print("       --output-dir checkpoints/realistic_v2_diaphragm_lora \\")
    print("       --epochs 20 --batch-size 16 --lr 3e-4 \\")
    print("       --cfg-dropout 0.10 --sample-every 3 --save-every 5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
