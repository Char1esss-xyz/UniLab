"""Resolve packaged models and prepare optional visuals during construction."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ASSETS_ROOT_PATH = Path(__file__).resolve().parent
_WHEELBIPE_MODEL = ASSETS_ROOT_PATH / "robots/wheelbipe_v14/scene_flat.xml"
_VISUAL_MANIFEST = Path(".unilab_assets/wheelbipe_v14_visuals/manifest.json")


def resolve_model_file(model_file: str | Path, *, root_dir: str | Path | None = None) -> Path:
    """Resolve built-in config paths against the package in source and wheel installs."""
    path = Path(model_file).expanduser()
    package_prefix = ("src", "unilab", "assets")
    if not path.is_absolute() and path.parts[:3] == package_prefix:
        return (ASSETS_ROOT_PATH / Path(*path.parts[3:])).resolve()
    if not path.is_absolute():
        path = (Path(root_dir) if root_dir is not None else Path.cwd()) / path
    return path.resolve()


def wheelbipe_body_names() -> frozenset[str]:
    """Read the current task's body selection on the asset conversion cold path."""
    root = ET.parse(_WHEELBIPE_MODEL.with_name("robot.xml")).getroot()
    return frozenset(body.attrib["name"] for body in root.iter("body"))


def _write_atomic(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def materialize_visual_scene(
    model_file: str | Path,
    manifest: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Build a cached scene with available task bodies, preserving the physics source."""
    model_path = Path(model_file).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    scene = ET.parse(model_path).getroot()
    robot_include = next(
        (
            item
            for item in scene.findall("include")
            if Path(item.attrib["file"]).name == "robot.xml"
        ),
        None,
    )
    if robot_include is None:
        raise ValueError(f"Wheelbipe scene must include robot.xml: {model_path}")
    robot_path = (model_path.parent / robot_include.attrib["file"]).resolve()
    robot = ET.parse(robot_path).getroot()
    bodies = {body.attrib["name"]: body for body in robot.iter("body")}
    data: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Visual manifest must contain a link mapping: {manifest_path}")
    selected = {name: value for name, value in data.items() if name in bodies}
    if not selected:
        raise ValueError(f"Visual manifest has no bodies in {robot_path}")

    # The generated XML lives outside the package, so existing mesh references
    # must keep pointing at their original asset files.
    for element in robot.iter():
        if "file" in element.attrib:
            element.set("file", str((robot_path.parent / element.attrib["file"]).resolve()))
    asset = robot.find("asset")
    if asset is None:
        asset = ET.SubElement(robot, "asset")
    for name, item in selected.items():
        mesh_path = (manifest_path.parent / item["file"]).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Visual mesh not found: {mesh_path}")
        visual_name = f"{name}_visual"
        ET.SubElement(asset, "mesh", name=visual_name, file=str(mesh_path))
        ET.SubElement(
            bodies[name],
            "geom",
            name=visual_name,
            type="mesh",
            mesh=visual_name,
            contype="0",
            conaffinity="0",
            group="1",
            rgba=" ".join(str(float(value)) for value in item["rgba"]),
        )
    robot_include.set("file", "robot.xml")
    robot_text = ET.tostring(robot, encoding="unicode")
    scene_text = ET.tostring(scene, encoding="unicode")
    digest = hashlib.sha256((robot_text + scene_text).encode()).hexdigest()[:16]
    cache = Path(output_dir) if output_dir is not None else manifest_path.parent / "models"
    destination = cache / digest
    destination.mkdir(parents=True, exist_ok=True)
    result = destination / model_path.name
    if not (destination / "robot.xml").is_file() or not result.is_file():
        _write_atomic(destination / "robot.xml", robot_text)
        _write_atomic(result, scene_text)
    return result.resolve()


def prepare_model_file(
    model_file: str | Path,
    *,
    root_dir: str | Path | None = None,
    visual_manifest: str | Path | None = None,
) -> Path:
    """Use canonical physics unless a local optional visual cache is available."""
    model_path = resolve_model_file(model_file, root_dir=root_dir)
    if visual_manifest is not None:
        return materialize_visual_scene(model_path, visual_manifest)
    if model_path != _WHEELBIPE_MODEL:
        return model_path
    roots = [Path(root_dir)] if root_dir is not None else []
    source_root = ASSETS_ROOT_PATH.parents[2]
    if (source_root / "pyproject.toml").is_file():
        roots.append(source_root)
    roots.append(Path.cwd())
    for root in roots:
        manifest = root / _VISUAL_MANIFEST
        if manifest.is_file():
            try:
                return materialize_visual_scene(model_path, manifest)
            except FileNotFoundError:
                # A partial local cache must not prevent physics-only operation.
                continue
    return model_path
