import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.scripts import manual_policy_mujoco, play_rsl_rl
from unilab.utils.sim2sim import (
    CrossBackendIncompatibleError,
    extract_contract_snapshot,
    policy_load_dim_guard,
    resolve_sim2sim_config,
)


@pytest.fixture
def cfg():
    config_dir = Path(__file__).resolve().parents[1] / "src/unilab/conf/ppo"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="config")


def save_snapshot(directory, cfg):
    (directory / "run_config.json").write_text(
        json.dumps({"contract_snapshot": extract_contract_snapshot(cfg)})
    )


def test_backend_scene_and_run_settings_may_change_without_changing_policy(cfg, tmp_path):
    save_snapshot(tmp_path, cfg)
    target = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    target.training.sim_backend = "mjwarp"
    target.env.manager_torch_device = "cuda:1"
    target.env.scene.model_file = "/different/scene.xml"
    target.algo.num_envs = 2

    assert resolve_sim2sim_config(tmp_path, target) is target
    assert cfg.training.sim_backend == "mujoco"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("env.observations.policy.terms.state.scale", 2.0),
        ("env.actions.motors.entity_name", "another_robot"),
        ("algo.actor.hidden_dims", [32]),
        ("algo.actor.activation", "tanh"),
    ),
)
def test_policy_semantics_changes_fail_strict_validation(cfg, tmp_path, field, value):
    save_snapshot(tmp_path, cfg)
    OmegaConf.update(cfg, field, value, force_add=True)

    with pytest.raises(CrossBackendIncompatibleError, match="Policy contract mismatch"):
        resolve_sim2sim_config(tmp_path, cfg)


def test_old_run_config_yaml_still_protects_observation_semantics(cfg, tmp_path):
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    OmegaConf.update(cfg, "env.observations.policy.terms.state.scale", 2.0, force_add=True)

    with pytest.raises(CrossBackendIncompatibleError, match="env.observations"):
        resolve_sim2sim_config(tmp_path, cfg)


def test_partial_historical_snapshot_uses_saved_config_for_new_fields(cfg, tmp_path):
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "contract_snapshot": {
                    "env.observations": extract_contract_snapshot(cfg)["env.observations"]
                }
            }
        )
    )
    assert resolve_sim2sim_config(tmp_path, cfg) is cfg
    cfg.algo.actor.activation = "tanh"
    with pytest.raises(CrossBackendIncompatibleError, match="algo.actor.activation"):
        resolve_sim2sim_config(tmp_path, cfg)


def test_saved_config_cannot_override_existing_snapshot_fields(cfg, tmp_path):
    save_snapshot(tmp_path, cfg)
    cfg.algo.actor.activation = "tanh"
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    with pytest.raises(CrossBackendIncompatibleError, match="algo.actor.activation"):
        resolve_sim2sim_config(tmp_path, cfg)


def test_warning_fields_report_differences_without_rejecting(cfg, tmp_path):
    save_snapshot(tmp_path, cfg)
    cfg.env.ctrl_dt = 0.04

    with pytest.warns(UserWarning, match="env.ctrl_dt"):
        resolve_sim2sim_config(tmp_path, cfg)


def test_missing_source_contract_fails_closed_unless_explicitly_non_strict(cfg, tmp_path):
    with pytest.raises(CrossBackendIncompatibleError, match="No training contract"):
        resolve_sim2sim_config(tmp_path, cfg)
    with pytest.warns(UserWarning, match="No training contract"):
        assert resolve_sim2sim_config(tmp_path, cfg, strict=False) is cfg


def test_invalid_snapshot_is_not_silently_replaced_by_config_yaml(cfg, tmp_path):
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    (tmp_path / "run_config.json").write_text('{"contract_snapshot": []}')
    with pytest.raises(CrossBackendIncompatibleError, match="Invalid policy contract"):
        resolve_sim2sim_config(tmp_path, cfg)


@pytest.mark.parametrize("entrypoint", [play_rsl_rl.play, manual_policy_mujoco.run])
def test_play_entrypoints_reject_rescaled_observations_before_env_construction(
    cfg, tmp_path, monkeypatch, entrypoint
):
    save_snapshot(tmp_path, cfg)
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    OmegaConf.update(cfg, "env.observations.policy.terms.state.scale", 2.0, force_add=True)
    config_path = tmp_path / "play.yaml"
    OmegaConf.save(cfg, config_path)

    def unexpected_env(*args, **kwargs):
        pytest.fail("an incompatible environment must not be constructed")

    monkeypatch.setattr(play_rsl_rl, "create_env", unexpected_env)
    monkeypatch.setattr(manual_policy_mujoco, "create_env", unexpected_env)
    with pytest.raises(CrossBackendIncompatibleError, match="env.observations"):
        entrypoint(checkpoint, config_path=config_path, device="cpu")


def test_explicit_missing_play_config_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Playback config not found"):
        play_rsl_rl._load_config(tmp_path / "model.pt", tmp_path / "typo.yaml")


def test_checkpoint_dimension_guard_reports_environment_dimensions():
    with pytest.raises(CrossBackendIncompatibleError, match="observations=35, actions=6"):
        with policy_load_dim_guard(env_obs_dim=35, env_action_dim=6):
            raise RuntimeError("size mismatch for mlp.0.weight")
    with pytest.raises(RuntimeError, match="unrelated failure"):
        with policy_load_dim_guard(env_obs_dim=35, env_action_dim=6):
            raise RuntimeError("unrelated failure")
