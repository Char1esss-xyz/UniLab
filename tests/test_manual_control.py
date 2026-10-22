from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from unilab.scripts import manual_policy_mujoco as manual


@pytest.mark.parametrize(
    ("expired_key", "active_key", "stopped_field", "moving_field"),
    ((265, 263, "forward", "yaw"), (263, 265, "yaw", "forward")),
)
def test_expired_direction_stops_without_cancelling_other_direction(
    monkeypatch, expired_key, active_key, stopped_field, moving_field
):
    monkeypatch.setattr(manual.time, "monotonic", lambda: 10.0)
    command = manual.ManualCommand(
        forward=0.5,
        yaw=0.5,
        height=0.35,
        held_until={expired_key: 9.9, active_key: 10.1, ord("q"): 9.9},
    )

    manual._advance_manual_command(command, 0.02)

    assert getattr(command, stopped_field) == 0.0
    assert getattr(command, moving_field) > 0.5
    assert command.height == 0.35
    assert set(command.held_until) == {active_key}


def test_released_arrows_stop_both_velocity_commands(monkeypatch):
    monkeypatch.setattr(manual.time, "monotonic", lambda: 10.0)
    command = manual.ManualCommand(
        forward=1.5,
        yaw=-0.8,
        held_until={265: 9.9, 262: 9.9},
    )

    manual._advance_manual_command(command, 0.02)

    assert command.forward == 0.0
    assert command.yaw == 0.0


def test_policy_loop_waits_for_control_period_instead_of_physics_substep(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    cfg = OmegaConf.create(
        {
            "training": {"task_name": "WheelbipeV14Flat", "sim2sim_strict": True},
            "env": {},
            "algo": {},
        }
    )
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    model = SimpleNamespace(opt=SimpleNamespace(timestep=0.005))
    closed = []
    env = SimpleNamespace(
        cfg=SimpleNamespace(ctrl_dt=0.02),
        action_manager=SimpleNamespace(get_term=lambda name: object()),
        get_playback_model=lambda row: model,
        get_physics_state_snapshot=lambda: None,
        close=lambda: closed.append(True),
    )
    wrapped = SimpleNamespace(
        num_obs=35,
        num_actions=6,
        reset=lambda: ({}, {}),
        step=lambda actions: ({}, torch.zeros(1), torch.zeros(1, dtype=torch.bool), {}),
    )
    runner = SimpleNamespace(
        load=lambda *args, **kwargs: None,
        get_inference_policy=lambda **kwargs: lambda obs, **kwargs_: torch.zeros((1, 6)),
    )
    running = iter((True, False))
    viewer = SimpleNamespace(
        is_running=lambda: next(running),
        lock=nullcontext,
        sync=lambda: None,
    )
    clock = iter((0.0, 0.0, 0.004))
    sleeps = []
    monkeypatch.setattr(
        manual,
        "time",
        SimpleNamespace(
            perf_counter=lambda: next(clock),
            monotonic=lambda: 10.0,
            sleep=sleeps.append,
        ),
    )
    monkeypatch.setattr(manual, "_load_config", lambda *args: cfg)
    monkeypatch.setattr(manual, "_resolve_device", lambda *args: "cpu")
    monkeypatch.setattr(
        manual, "apply_configured_training_seed", lambda cfg: SimpleNamespace(effective_seed=1)
    )
    monkeypatch.setattr(
        manual,
        "BackendAdapter",
        lambda *args, **kwargs: SimpleNamespace(build_task_env_cfg_override=lambda: {}),
    )
    monkeypatch.setattr(manual, "create_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(manual, "RslRlVecEnvAdapter", lambda *args, **kwargs: wrapped)
    monkeypatch.setattr(manual, "patch_rsl_rl_resume_state", lambda: None)
    monkeypatch.setattr(manual, "OnPolicyRunner", lambda *args, **kwargs: runner)
    monkeypatch.setattr(manual, "_set_manual_command", lambda *args: None)
    monkeypatch.setattr(manual, "_sync_viewer", lambda *args: None)
    monkeypatch.setattr(manual, "mujoco", SimpleNamespace(MjData=lambda model: None))
    monkeypatch.setattr(
        manual,
        "mujoco_viewer",
        SimpleNamespace(launch_passive=lambda *args, **kwargs: nullcontext(viewer)),
    )

    manual.run(checkpoint, device="cpu")

    assert sleeps == [pytest.approx(0.016)]
    assert closed == [True]
