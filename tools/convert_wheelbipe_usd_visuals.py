"""Export Wheelbipe USD visual meshes for MuJoCo playback.

This cold-path tool needs ``usd-core`` but does not add it to UniLab's runtime
dependencies.  The exported OBJ files are local generated assets and should
stay outside git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdShade


def _rgba_for_mesh(mesh: Usd.Prim) -> tuple[float, float, float, float]:
    material, _ = UsdShade.MaterialBindingAPI(mesh.GetParent()).ComputeBoundMaterial()
    if material:
        shader = material.GetPrim().GetChild("Shader")
        value = shader.GetAttribute("inputs:diffuse_color_constant").Get()
        if value is not None:
            return (float(value[0]), float(value[1]), float(value[2]), 1.0)
    return (0.7, 0.7, 0.7, 1.0)


def _write_obj(mesh: UsdGeom.Mesh, transform: Gf.Matrix4d, output: Path) -> None:
    points = mesh.GetPointsAttr().Get() or []
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii") as stream:
        stream.write("# Generated from Wheelbipe V14 USD visual mesh.\n")
        for point in points:
            value = transform.Transform(Gf.Vec3d(point))
            stream.write(f"v {value[0]:.9g} {value[1]:.9g} {value[2]:.9g}\n")
        cursor = 0
        for count in counts:
            face = [int(index) + 1 for index in indices[cursor : cursor + int(count)]]
            cursor += int(count)
            for index in range(1, len(face) - 1):
                stream.write(f"f {face[0]} {face[index]} {face[index + 1]}\n")


def export(source: Path, destination: Path) -> dict[str, object]:
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {source}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    exported: dict[str, object] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not str(prim.GetPath()).startswith("/visuals/"):
            continue
        link = str(prim.GetPath()).split("/")[2]
        link_prim = stage.GetPrimAtPath(f"/visuals/{link}")
        transform = cache.GetLocalToWorldTransform(prim)
        transform = cache.GetLocalToWorldTransform(link_prim).GetInverse() * transform
        name = f"{link}.obj"
        _write_obj(UsdGeom.Mesh(prim), transform, destination / name)
        exported[link] = {"file": name, "rgba": _rgba_for_mesh(prim)}
    if len(exported) != 37:
        raise RuntimeError(f"Expected 37 visual meshes, found {len(exported)}")
    manifest = destination / "manifest.json"
    manifest.write_text(json.dumps(exported, indent=2) + "\n", encoding="ascii")
    return exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    exported = export(args.source, args.destination)
    print(f"Exported {len(exported)} visual meshes to {args.destination}")


if __name__ == "__main__":
    main()
