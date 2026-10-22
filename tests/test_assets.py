import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from unilab.assets import ASSETS_ROOT_PATH, materialize_visual_scene, prepare_model_file
from unilab.assets import models as asset_models
from unilab.base.config_adapter import BackendAdapter
from unilab.tasks.locomotion.wheelbipe_v14.math import ClosedChainKinematics


@pytest.fixture
def clean_scene(tmp_path):
    destination = tmp_path / "checkout/src/unilab/assets/robots/wheelbipe_v14"
    shutil.copytree(ASSETS_ROOT_PATH / "robots/wheelbipe_v14", destination)
    return destination / "scene_flat.xml"


def _manifest(directory):
    directory.mkdir(parents=True)
    (directory / "base.obj").write_text(
        "v 0 0 0\nv 0.001 0 0\nv 0 0.001 0\nv 0 0 0.001\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
    )
    result = directory / "manifest.json"
    result.write_text(
        json.dumps(
            {
                "base_link": {"file": "base.obj", "rgba": [0.7, 0.7, 0.7, 1]},
                # Old full-robot manifests remain usable after removing the gimbal.
                "gimbal_yaw_link": {"file": "absent-gimbal.obj", "rgba": [1, 1, 1, 1]},
            }
        )
    )
    return result


def test_clean_checkout_model_and_closure_load_without_visual_cache(clean_scene):
    model = mujoco.MjModel.from_xml_path(str(clean_scene))
    names = [model.joint(index).name for index in range(1, model.njnt)]
    ClosedChainKinematics(clean_scene, names)
    assert model.nq == 41
    assert ".unilab_assets" not in clean_scene.with_name("robot.xml").read_text()


def test_wheel_layout_resolves_packaged_model_without_source_root(monkeypatch, tmp_path):
    package_assets = tmp_path / "venv/lib/python3.13/site-packages/unilab/assets"
    shutil.copytree(ASSETS_ROOT_PATH, package_assets)
    monkeypatch.setattr(asset_models, "ASSETS_ROOT_PATH", package_assets)
    monkeypatch.setattr(
        asset_models, "_WHEELBIPE_MODEL", package_assets / "robots/wheelbipe_v14/scene_flat.xml"
    )
    monkeypatch.chdir(tmp_path)
    config_dir = Path(__file__).resolve().parents[1] / "src/unilab/conf/ppo"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config")

    overrides = BackendAdapter(
        cfg, root_dir=tmp_path / "venv/lib/python3.13"
    ).build_task_env_cfg_override()

    selected = Path(overrides["scene"]["model_file"])
    assert selected == package_assets / "robots/wheelbipe_v14/scene_flat.xml"
    assert mujoco.MjModel.from_xml_path(str(selected)).nq == 41


def test_visual_materialization_filters_removed_bodies_and_preserves_physics(clean_scene, tmp_path):
    manifest = _manifest(tmp_path / "visuals")
    canonical_xml = clean_scene.with_name("robot.xml").read_bytes()
    physics = mujoco.MjModel.from_xml_path(str(clean_scene))

    result = materialize_visual_scene(clean_scene, manifest)
    visual = mujoco.MjModel.from_xml_path(str(result))

    assert clean_scene.with_name("robot.xml").read_bytes() == canonical_xml
    assert result != clean_scene
    assert visual.ngeom == physics.ngeom + 1
    np.testing.assert_array_equal(visual.body_mass, physics.body_mass)
    np.testing.assert_array_equal(visual.body_inertia, physics.body_inertia)
    np.testing.assert_array_equal(visual.qpos0, physics.qpos0)
    assert "gimbal" not in result.with_name("robot.xml").read_text()
    before = result.stat().st_mtime_ns
    assert materialize_visual_scene(clean_scene, manifest) == result
    assert result.stat().st_mtime_ns == before


def test_incomplete_optional_cache_falls_back_to_canonical_physics(monkeypatch, clean_scene):
    checkout = clean_scene.parents[5]
    manifest = _manifest(checkout / ".unilab_assets/wheelbipe_v14_visuals")
    manifest.with_name("base.obj").unlink()
    monkeypatch.setattr(asset_models, "ASSETS_ROOT_PATH", clean_scene.parents[2])
    monkeypatch.setattr(asset_models, "_WHEELBIPE_MODEL", clean_scene)
    monkeypatch.chdir(checkout)

    assert prepare_model_file(clean_scene, root_dir=checkout) == clean_scene
    with pytest.raises(FileNotFoundError, match="Visual mesh not found"):
        prepare_model_file(clean_scene, visual_manifest=manifest)


def test_custom_relative_model_paths_keep_the_explicit_root(tmp_path):
    assert asset_models.resolve_model_file("models/custom.xml", root_dir=tmp_path) == (
        tmp_path / "models/custom.xml"
    )
