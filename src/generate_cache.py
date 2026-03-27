"""
Pre-generate frame cache for all scenarios × zones on GPU.
=========================================================
Generates diffusion-refined B-mode stacks and M-mode strips for every
(scenario, zone) combination and saves as a compact .npz file.

The Docker CPU image ships with this cache — no GPU needed at runtime.

Usage:
    python -m src.generate_cache                         # defaults
    python -m src.generate_cache --image-size 256        # smaller/faster
    python -m src.generate_cache --n-frames 32           # more frames
    python -m src.generate_cache --output data/frame_cache.npz
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import time
import numpy as np
from pathlib import Path

from .poc_image_stack import (
    POCImageStackGenerator, StackConfig, ProbeReading,
    LungZone, SCENARIOS,
)


def generate_cache(
    image_size: int = 256,
    n_frames: int = 32,
    output_path: str = "data/frame_cache.npz",
):
    """Generate and save pre-rendered frames for all scenarios × zones."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cfg = StackConfig(n_frames=n_frames, image_size=(image_size, image_size))
    cache = {}
    total = len(SCENARIOS) * len(LungZone)
    done = 0

    for scenario_key in SCENARIOS:
        print(f"\n=== Scenario: {scenario_key} ===")
        gen = POCImageStackGenerator(scenario=scenario_key, stack_config=cfg)

        for zone in LungZone:
            t0 = time.time()
            anchor = gen.zone_resolver.get_anchor(zone)
            probe = ProbeReading(x_m=anchor.x_m, y_m=anchor.y_m, pressure=1.0)
            result = gen.generate(probe, seed=hash(f"{scenario_key}_{zone.name}") % (2**31))

            key = f"{scenario_key}/{zone.name}"
            cache[f"{key}/bmode"] = result["bmode_stack"].astype(np.float16)
            cache[f"{key}/mmode"] = result["mmode"].astype(np.float16)
            cache[f"{key}/pathology"] = result["pathology"]
            cache[f"{key}/pathology_class"] = result["pathology_class"]
            cache[f"{key}/sliding"] = result["lung_sliding"]
            cache[f"{key}/mmode_pattern"] = result["mmode_pattern"]

            done += 1
            dt = time.time() - t0
            print(f"  [{done}/{total}] {zone.name}: {result['pathology']} ({dt:.1f}s)")

    # Save as compressed npz
    print(f"\nSaving cache to {output}...")
    np.savez_compressed(str(output), **cache)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Done! Cache size: {size_mb:.1f} MB ({total} zone stacks, {n_frames} frames each)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-generate frame cache for Docker deployment")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--n-frames", type=int, default=32)
    parser.add_argument("--output", type=str, default="data/frame_cache.npz")
    args = parser.parse_args()

    generate_cache(
        image_size=args.image_size,
        n_frames=args.n_frames,
        output_path=args.output,
    )
