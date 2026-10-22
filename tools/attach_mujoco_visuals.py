"""Prepare a visual scene without changing the canonical Wheelbipe physics XML."""

from __future__ import annotations

import argparse
from pathlib import Path

from unilab.assets import materialize_visual_scene


def attach(robot_xml: Path, manifest: Path, *, output_dir: Path | None = None) -> Path:
    return materialize_visual_scene(
        robot_xml.with_name("scene_flat.xml"), manifest, output_dir=output_dir
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_xml", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = attach(args.robot_xml, args.manifest, output_dir=args.output_dir)
    print(f"Prepared visual scene: {result}")


if __name__ == "__main__":
    main()
