"""
Class Imbalance Scanner
========================
Scans the real POCUS dataset to identify under-represented pathology classes,
with focus on high-severity consolidations (Severity 3/4).

Outputs:
  - Per-class counts and percentages
  - Imbalance ratio (max_class / min_class)
  - Recommended synthesis counts to reach target balance

Usage:
    python scripts/scan_class_balance.py
    python scripts/scan_class_balance.py --target-per-class 1500
"""

import csv
import sys
from collections import Counter
from pathlib import Path


# Pathology names (avoid importing src/ which pulls in keras/torch)
PATHOLOGY_NAMES = {
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

# B-mode classes only (0-9). Classes 10+ are M-mode variants.
BMODE_CLASSES = set(range(10))
MMODE_OFFSET = 10

# Severity mapping for consolidation-related classes
# Severity 3: diffuse B-lines (class 3), ARDS/white lung (class 6)
# Severity 4: consolidation (class 4)
SEVERITY_HIGH = {3, 4, 6}


def scan_metadata(metadata_path: str) -> tuple:
    """Read metadata.csv and return (bmode_counts, mmode_counts, name_map)."""
    bmode_counts = Counter()
    mmode_counts = Counter()
    name_map = {}

    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = int(row["pathology_class"])
            name_map[cls] = row["pathology_name"]
            if cls in BMODE_CLASSES:
                bmode_counts[cls] += 1
            else:
                mmode_counts[cls] += 1

    return bmode_counts, mmode_counts, name_map


def report(bmode_counts: Counter, mmode_counts: Counter, name_map: dict,
           target_per_class: int = 0) -> dict:
    """Print distribution report and return synthesis recommendations."""
    total_bmode = sum(bmode_counts.values())
    total_mmode = sum(mmode_counts.values())
    max_count = max(bmode_counts.values()) if bmode_counts else 0
    min_count = min(bmode_counts.get(c, 0) for c in BMODE_CLASSES)

    print("=" * 70)
    print("B-MODE CLASS DISTRIBUTION")
    print("=" * 70)
    print(f"{'Class':>5}  {'Pathology':<30}  {'Count':>6}  {'%':>6}  {'Sev':>4}")
    print("-" * 70)

    synthesis_needed = {}

    for cls in sorted(BMODE_CLASSES):
        name = PATHOLOGY_NAMES.get(cls, name_map.get(cls, f"unknown_{cls}"))
        count = bmode_counts.get(cls, 0)
        pct = (count / total_bmode * 100) if total_bmode > 0 else 0
        sev = "HIGH" if cls in SEVERITY_HIGH else ""
        marker = " <<<" if count < (max_count * 0.3) else ""
        print(f"{cls:>5}  {name:<30}  {count:>6}  {pct:>5.1f}%  {sev:>4}{marker}")

        if target_per_class > 0 and count < target_per_class:
            synthesis_needed[cls] = target_per_class - count

    print("-" * 70)
    print(f"{'TOTAL B-mode':<37}  {total_bmode:>6}")
    print(f"{'TOTAL M-mode':<37}  {total_mmode:>6}")
    print(f"\nImbalance ratio (max/min): {max_count}/{min_count} = {max_count/min_count:.1f}x" if min_count > 0 else "\nImbalance ratio: N/A (missing classes)")

    most = bmode_counts.most_common(1)[0]
    print(f"Most represented:  class {most[0]} ({PATHOLOGY_NAMES[most[0]]}) = {most[1]}")
    least_cls = min(BMODE_CLASSES, key=lambda c: bmode_counts.get(c, 0))
    print(f"Least represented: class {least_cls} ({PATHOLOGY_NAMES[least_cls]}) = {bmode_counts.get(least_cls, 0)}")

    # High-severity focus
    print(f"\nHIGH-SEVERITY CLASSES (consolidation / severe parenchymal):")
    for cls in sorted(SEVERITY_HIGH):
        name = PATHOLOGY_NAMES[cls]
        count = bmode_counts.get(cls, 0)
        deficit = max_count - count
        print(f"  Class {cls} ({name}): {count} samples, deficit vs max: {deficit}")

    if target_per_class > 0 and synthesis_needed:
        print(f"\nSYNTHESIS RECOMMENDATIONS (target={target_per_class}/class):")
        total_synth = 0
        for cls in sorted(synthesis_needed):
            name = PATHOLOGY_NAMES.get(cls, f"class_{cls}")
            n = synthesis_needed[cls]
            print(f"  Class {cls} ({name}): generate {n} frames")
            total_synth += n
        print(f"  Total frames to synthesize: {total_synth}")
    elif target_per_class > 0:
        print(f"\nAll classes meet target of {target_per_class}/class.")

    return synthesis_needed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scan LUS dataset class balance")
    parser.add_argument("--metadata", default="data/real_pocus/processed/metadata.csv",
                        help="Path to metadata.csv")
    parser.add_argument("--target-per-class", type=int, default=0,
                        help="Target samples per class (0 = report only)")
    args = parser.parse_args()

    metadata_path = Path(__file__).resolve().parent.parent / args.metadata
    if not metadata_path.exists():
        print(f"ERROR: metadata not found at {metadata_path}")
        sys.exit(1)

    bmode_counts, mmode_counts, name_map = scan_metadata(str(metadata_path))
    synthesis_needed = report(bmode_counts, mmode_counts, name_map, args.target_per_class)
    return synthesis_needed


if __name__ == "__main__":
    main()
