#!/usr/bin/env python3
"""Reconstruct every built-in sample and write a contact sheet.

    python examples/generate_examples.py --out examples/output

Produces, per sample: the source image, the exported model files, and a
turntable render. Handy as a smoke test after changing the pipeline, and as a
way to see what the tool actually produces without supplying a photo.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from image_to_model.imaging import resize_array, save_image
from image_to_model.logging import configure
from image_to_model.pipeline import reconstruct
from image_to_model.preview import render_turntable
from image_to_model.samples import SAMPLES, make_sample

TILE = 256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="examples/output", help="Output directory.")
    parser.add_argument("--target", default="roblox", help="Target platform profile.")
    parser.add_argument("--size", type=int, default=384, help="Source image size.")
    parser.add_argument("--views", type=int, default=4, help="Turntable views per sample.")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    configure(args.verbose)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in sorted(SAMPLES):
        image = make_sample(name, args.size)
        source_path = out_dir / f"{name}.png"
        save_image(image, source_path)

        started = time.perf_counter()
        result = reconstruct(source_path, target=args.target)
        elapsed = time.perf_counter() - started

        result.save(out_dir / name)
        print(f"\n=== {name} ({elapsed:.1f}s) ===")
        print(result.summary())

        turntable = render_turntable(
            result.mesh, size=TILE, views=args.views, texture=result.texture
        )
        source_tile = resize_array(image[..., :3], (TILE, TILE))
        rows.append(np.concatenate([source_tile, turntable], axis=1))

    sheet_path = save_image(np.concatenate(rows, axis=0), out_dir / "contact_sheet.png")
    print(f"\nContact sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
