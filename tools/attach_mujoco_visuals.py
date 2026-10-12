"""Attach generated Wheelbipe visual OBJ meshes to the MuJoCo robot XML."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

BEGIN = "    <!-- Wheelbipe USD visual meshes: begin -->"
END = "    <!-- Wheelbipe USD visual meshes: end -->"


def attach(robot_xml: Path, manifest: Path) -> None:
    data = json.loads(manifest.read_text(encoding="ascii"))
    text = robot_xml.read_text(encoding="utf-8")
    old = re.compile(
        r"\n?    <!-- Wheelbipe USD visual meshes: begin -->.*?    <!-- Wheelbipe USD visual meshes: end -->\n?",
        re.S,
    )
    text = old.sub("\n", text)
    text = re.sub(r"^\s*<geom name=\"[^\"]+_visual\" type=\"mesh\"[^\n]*\n", "", text, flags=re.M)

    asset_lines = [BEGIN]
    for link, item in data.items():
        asset_lines.append(
            f'    <mesh name="{link}_visual" '
            f'file="{os.path.relpath(manifest.parent / item["file"], robot_xml.parent)}" />'
        )
    asset_lines.append(END)
    asset_block = "\n".join(asset_lines)
    text = text.replace("  <asset>\n", f"  <asset>\n{asset_block}\n", 1)

    for link, item in data.items():
        rgba = " ".join(f"{float(value):.6f}" for value in item["rgba"])
        pattern = re.compile(rf'(?m)^(?P<indent>\s*)<body name="{re.escape(link)}"[^>]*>')
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"MuJoCo body not found for USD link: {link}")
        indent = match.group("indent") + "  "
        geom = (
            f'{indent}<geom name="{link}_visual" type="mesh" '
            f'mesh="{link}_visual" contype="0" conaffinity="0" '
            f'group="1" rgba="{rgba}" />\n'
        )
        text = text[: match.end()] + "\n" + geom + text[match.end() :]

    robot_xml.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_xml", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    attach(args.robot_xml, args.manifest)
    print(f"Attached visual meshes to {args.robot_xml}")


if __name__ == "__main__":
    main()
