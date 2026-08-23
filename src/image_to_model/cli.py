"""Command-line interface.

    image-to-model photo.jpg -o out/            # Roblox-ready OBJ + GLB + texture
    image-to-model photo.jpg -o chair.glb --target game
    image-to-model demo --sample rocket -o out/
    image-to-model info
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import PRESETS, ReconstructionConfig, preset
from .errors import ImageToModelError
from .logging import configure, get_logger

log = get_logger("cli")


def _add_verbosity(parser: argparse.ArgumentParser) -> None:
    """Accept -v on a subcommand as well as before it.

    ``SUPPRESS`` as the default is what makes this safe: without it the
    subparser would write its own default over a ``-v`` given earlier.
    """
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=argparse.SUPPRESS,
        help="Show progress; repeat for debug detail.",
    )


def _add_reconstruction_options(parser: argparse.ArgumentParser) -> None:
    """Options shared by the commands that actually build a model."""
    _add_verbosity(parser)
    output = parser.add_argument_group("output")
    output.add_argument(
        "-o",
        "--output",
        default="out",
        help="Output file or directory. A directory writes the target's preferred formats.",
    )
    output.add_argument(
        "-f",
        "--format",
        dest="formats",
        action="append",
        choices=["obj", "glb", "gltf", "ply", "stl"],
        help="Export format; repeat for several. Defaults to the target's preferred set.",
    )
    output.add_argument(
        "--preview",
        nargs="?",
        const="preview.png",
        default=None,
        metavar="PATH",
        help="Render a turntable PNG of the result.",
    )
    output.add_argument(
        "--viewer",
        nargs="?",
        const="viewer.html",
        default=None,
        metavar="PATH",
        help="Write a self-contained HTML page for spinning the model in a browser.",
    )
    output.add_argument(
        "--debug-dir",
        default=None,
        metavar="DIR",
        help="Also write the intermediate mask, depth map and baked texture.",
    )

    target = parser.add_argument_group("target")
    target.add_argument(
        "-t",
        "--target",
        default=None,
        help="Destination platform: roblox, roblox-avatar, game, print, generic.",
    )
    target.add_argument(
        "--preset", default=None, choices=sorted(PRESETS), help="Start from a named preset."
    )
    target.add_argument(
        "--target-faces",
        type=int,
        default=None,
        help="Triangle budget. 0 uses the target's; negative disables decimation.",
    )
    target.add_argument(
        "--size-units",
        type=float,
        default=None,
        help="Longest axis in the target's units (studs for Roblox).",
    )
    target.add_argument(
        "--texture-size", type=int, default=None, help="Baked texture resolution in pixels."
    )

    quality = parser.add_argument_group("quality")
    quality.add_argument(
        "-r",
        "--working-resolution",
        type=int,
        default=None,
        help="Longest image edge used for depth and texture (default 512).",
    )
    quality.add_argument(
        "--depth-model",
        default=None,
        help="auto, heuristic, depth-anything, dpt, or a HuggingFace model id.",
    )
    quality.add_argument(
        "--segmentation",
        default=None,
        choices=["auto", "alpha", "rembg", "border", "none"],
        help="How to separate the subject from the background.",
    )
    quality.add_argument(
        "--smooth-iterations", type=int, default=None, help="Taubin smoothing passes."
    )

    shape = parser.add_argument_group("shape")
    shape.add_argument(
        "--relief-scale",
        type=float,
        default=None,
        help="Depth of the front surface as a fraction of subject width (default 0.35).",
    )
    shape.add_argument(
        "--thickness",
        type=float,
        default=None,
        help="Back depth relative to the front. 0 gives a flat back (default 0.55).",
    )
    shape.add_argument(
        "--open-back",
        dest="close_back",
        action="store_false",
        default=None,
        help="Leave the model open instead of closing it into a solid.",
    )
    shape.add_argument(
        "--fov-degrees", type=float, default=None, help="Assumed camera field of view."
    )
    shape.add_argument(
        "--no-texture",
        dest="texture",
        action="store_false",
        default=None,
        help="Skip colour entirely.",
    )
    shape.add_argument(
        "--bake-texture",
        default=None,
        choices=["auto", "always", "never"],
        help="Bake a UV texture image. Roblox needs one, so auto bakes for it.",
    )
    shape.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])


def _build_config(args: argparse.Namespace) -> ReconstructionConfig:
    base = preset(args.preset) if getattr(args, "preset", None) else ReconstructionConfig()
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in {"preset", "command", "output", "formats", "preview", "viewer", "debug_dir", "verbose", "image", "sample", "size", "func"}
    }
    return base.merged(**overrides)


def _run_reconstruction(args: argparse.Namespace, image) -> int:
    from .pipeline import reconstruct

    config = _build_config(args)
    result = reconstruct(image, config=config)

    written = result.save(args.output, formats=args.formats)

    if args.debug_dir:
        written.extend(result.save_debug(args.debug_dir))

    if args.preview:
        from .imaging import save_image
        from .preview import render_turntable

        preview_path = Path(args.preview)
        if not preview_path.is_absolute() and Path(args.output).suffix == "":
            preview_path = Path(args.output) / preview_path
        sheet = render_turntable(result.mesh, size=320, views=4, texture=result.texture)
        written.append(save_image(sheet, preview_path))

    if args.viewer:
        from .viewer import write_viewer

        viewer_path = Path(args.viewer)
        if not viewer_path.is_absolute() and Path(args.output).suffix == "":
            viewer_path = Path(args.output) / viewer_path
        written.append(write_viewer(result.mesh, viewer_path, texture=result.texture))

    print(result.summary())
    print()
    print("Wrote:")
    for path in written:
        print(f"  {path}")

    return 0 if result.validation.ok else 1


def _cmd_run(args: argparse.Namespace) -> int:
    return _run_reconstruction(args, args.image)


def _cmd_demo(args: argparse.Namespace) -> int:
    from .samples import make_sample

    image = make_sample(args.sample, args.size)
    log.info("Reconstructing the %r sample at %dx%d", args.sample, args.size, args.size)
    return _run_reconstruction(args, image)


def _cmd_info(_: argparse.Namespace) -> int:
    from .backends import available_backends
    from .depth import available_models
    from .export import SUPPORTED_FORMATS
    from .segmentation import available_methods
    from .targets import TARGETS, available_targets

    print(f"image-to-model {__version__}\n")

    print("Targets:")
    for name in available_targets():
        profile = TARGETS[name]
        limit = (
            f"hard limit {profile.triangle_hard_limit:,}"
            if profile.triangle_hard_limit
            else "no hard limit"
        )
        print(
            f"  {name:14} {profile.triangle_budget:>7,} tris ({limit}), "
            f"texture <={profile.max_texture_size}, {profile.up_axis.upper()}-up, "
            f"units in {profile.unit_name}s"
        )

    print("\nBackends      :", ", ".join(available_backends()))
    print("Depth models  :", ", ".join(available_models()))
    print("Segmentation  :", ", ".join(available_methods()))
    print("Export formats:", ", ".join(SUPPORTED_FORMATS))
    print("Presets       :", ", ".join(sorted(PRESETS)))

    try:
        import torch  # noqa: F401

        print("\nLearned depth : available")
    except ImportError:
        print("\nLearned depth : not installed. `pip install 'image-to-model[ai]'` enables it;")
        print("                without it the heuristic silhouette estimator is used.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-to-model",
        description="Turn a single photograph into a textured 3D model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"image-to-model {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Show progress; repeat for debug detail.",
    )

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Reconstruct an image file.")
    run_parser.add_argument("image", help="Input image (png, jpg, webp, ...).")
    _add_reconstruction_options(run_parser)
    run_parser.set_defaults(func=_cmd_run)

    demo_parser = subparsers.add_parser("demo", help="Reconstruct a built-in sample image.")
    demo_parser.add_argument(
        "--sample", default="ball", choices=["ball", "star", "rocket"], help="Which sample."
    )
    demo_parser.add_argument("--size", type=int, default=384, help="Sample image size.")
    _add_reconstruction_options(demo_parser)
    demo_parser.set_defaults(func=_cmd_demo)

    info_parser = subparsers.add_parser("info", help="Show targets and available components.")
    _add_verbosity(info_parser)
    info_parser.set_defaults(func=_cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Allow the common form `image-to-model photo.jpg -o out/` with no subcommand.
    known = {"run", "demo", "info"}
    if argv and argv[0] not in known and not argv[0].startswith("-"):
        argv.insert(0, "run")

    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        parser.print_help()
        return 1

    configure(args.verbose)

    try:
        return args.func(args)
    except ImageToModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
