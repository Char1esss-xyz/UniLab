"""Interactive MuJoCo playback for a trained Wheelbipe policy."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, cast

import mujoco
import mujoco.viewer
import torch
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.rl.vec_env import RslRlVecEnvAdapter
from unilab.scripts.play_rsl_rl import (
    _load_config,
    _remove_gimbal_references,
    _resolve_device,
)
from unilab.training.experiment import patch_rsl_rl_resume_state
from unilab.utils.seed import apply_configured_training_seed


@dataclass
class ManualCommand:
    forward: float = 0.0
    yaw: float = 0.0
    height: float = 0.30
    speed_scale: float = 1.0
    reset_requested: bool = False
    stop_requested: bool = False
    held_until: dict[int, float] = field(default_factory=dict)


def _key_name(key: int) -> str:
    try:
        return chr(key).lower()
    except (TypeError, ValueError):
        return ""


def _key_callback(command: ManualCommand, lock: Lock):
    def on_key(key: int) -> None:
        name = _key_name(key)
        with lock:
            # Arrow keys avoid MuJoCo's built-in W/A/S/D camera shortcuts.
            if key in (262, 263, 264, 265) or name in ("q", "e"):
                held_key = key if key in (262, 263, 264, 265) else ord(name)
                command.held_until[held_key] = time.monotonic() + 0.30
                opposites = {265: 264, 264: 265, 263: 262, 262: 263, ord("q"): ord("e"), ord("e"): ord("q")}
                command.held_until.pop(opposites[held_key], None)
            elif name == " ":
                command.forward = 0.0
                command.yaw = 0.0
                command.held_until.clear()
            elif name == "1":
                command.speed_scale = 0.5
            elif name == "2":
                command.speed_scale = 1.0
            elif name == "3":
                command.speed_scale = 1.5
            elif name == "r":
                command.reset_requested = True
            elif key in (256, 27):
                command.stop_requested = True

    return on_key


def _read_command(command: ManualCommand, lock: Lock) -> ManualCommand:
    with lock:
        return ManualCommand(
            forward=command.forward,
            yaw=command.yaw,
            height=command.height,
            speed_scale=command.speed_scale,
            reset_requested=command.reset_requested,
            stop_requested=command.stop_requested,
            held_until=dict(command.held_until),
        )


def _advance_manual_command(command: ManualCommand, dt: float) -> None:
    """Integrate keys that are still receiving repeat events."""
    now = time.monotonic()
    command.held_until = {
        key: expiry for key, expiry in command.held_until.items() if expiry > now
    }
    active = command.held_until
    forward_direction = int(265 in active) - int(264 in active)
    yaw_direction = int(263 in active) - int(262 in active)
    height_direction = int(ord("q") in active) - int(ord("e") in active)
    command.forward = max(-2.7, min(2.7, command.forward + 1.5 * forward_direction * dt))
    command.yaw = max(
        -2 * 3.1415926535,
        min(2 * 3.1415926535, command.yaw + 3.0 * yaw_direction * dt),
    )
    command.height = max(0.20, min(0.42, command.height + 0.12 * height_direction * dt))


def _set_manual_command(motor: Any, command: ManualCommand) -> None:
    motor.command[0] = torch.tensor(
        [command.forward * command.speed_scale, 0.0, command.yaw * command.speed_scale],
        dtype=motor.command.dtype,
    )
    motor.height_command[0] = command.height
    motor.command_remaining[0] = 1.0e9
    motor.heading_env[0] = False
    motor.standing_env[0] = False
    motor.special[0] = 0
    motor.ground_start[0] = False
    motor.age[0] = max(float(motor.age[0]), 2.0)
    motor.observation_cache = None


def _sync_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    snapshot: torch.Tensor | Any,
) -> None:
    state = snapshot[0] if isinstance(snapshot, torch.Tensor) else snapshot[0]
    state = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else state
    nq = int(model.nq)
    data.time = float(state[0])
    data.qpos[:] = state[1 : 1 + nq]
    data.qvel[:] = state[1 + nq : 1 + nq + int(model.nv)]
    data.qacc[:] = 0
    mujoco.mj_forward(model, data)


def run(
    checkpoint: Path,
    *,
    config_path: Path | None = None,
    device: str | None = None,
) -> None:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    cfg = _load_config(checkpoint, config_path)
    _remove_gimbal_references(cfg)
    if str(cfg.training.task_name) != "WheelbipeV14Flat":
        raise ValueError("This checkout supports only WheelbipeV14Flat")
    OmegaConf.update(cfg, "training.sim_backend", "mujoco", force_add=True)
    learner_device = _resolve_device(cfg, device)
    OmegaConf.update(cfg, "training.device", learner_device, force_add=True)
    OmegaConf.update(cfg, "algo.num_envs", 1, force_add=True)
    OmegaConf.update(cfg, "env.manager_torch_device", "cpu", force_add=True)
    seed_info = apply_configured_training_seed(cfg)
    env_overrides = BackendAdapter(
        cfg, root_dir=Path(__file__).resolve().parents[3]
    ).build_task_env_cfg_override()
    env_overrides["manager_torch_device"] = "cpu"
    env_overrides["seed"] = seed_info.effective_seed
    env = create_env(cfg, num_envs=1, env_cfg_override=env_overrides)
    try:
        wrapped_env = RslRlVecEnvAdapter(env, device=learner_device)
        raw_train_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
        if not isinstance(raw_train_cfg, dict):
            raise TypeError("algo must resolve to a mapping")
        train_cfg = cast(dict[str, Any], raw_train_cfg)
        train_cfg["logger"] = "tensorboard"
        patch_rsl_rl_resume_state()
        runner = OnPolicyRunner(
            cast(Any, wrapped_env), train_cfg, log_dir=None, device=learner_device
        )
        runner.load(str(checkpoint), map_location=learner_device)
        policy = runner.get_inference_policy(device=learner_device)
        obs, _ = wrapped_env.reset()
        motor = env.action_manager.get_term("motors")
        visual_model = env.get_playback_model(0)
        visual_data = mujoco.MjData(visual_model)
        command = ManualCommand()
        command_lock = Lock()
        _set_manual_command(motor, command)
        _sync_viewer(visual_model, visual_data, env.get_physics_state_snapshot())
        print(
            "Loaded policy: "
            f"{checkpoint}\n"
            "Hold arrow keys to move, release to stop, Q/E height, Space stop, R reset, "
            "Esc quit; 1/2/3 speed"
        )
        with mujoco.viewer.launch_passive(
            visual_model,
            visual_data,
            key_callback=_key_callback(command, command_lock),
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            previous_loop = time.perf_counter()
            while viewer.is_running():
                loop_start = time.perf_counter()
                with command_lock:
                    _advance_manual_command(command, loop_start - previous_loop)
                previous_loop = loop_start
                current = _read_command(command, command_lock)
                if current.stop_requested:
                    break
                if current.reset_requested:
                    obs, _ = wrapped_env.reset()
                    with command_lock:
                        command.reset_requested = False
                    current = _read_command(command, command_lock)
                _set_manual_command(motor, current)
                with torch.inference_mode():
                    actions = policy(obs, stochastic_output=False)
                # Keep environment-owned observation buffers as regular tensors.
                obs, _, dones, _ = wrapped_env.step(actions.detach().clone())
                if bool(dones[0]):
                    obs, _ = wrapped_env.reset()
                with viewer.lock():
                    _sync_viewer(visual_model, visual_data, env.get_physics_state_snapshot())
                viewer.sync()
                time.sleep(
                    max(0.0, float(visual_model.opt.timestep) - (time.perf_counter() - loop_start))
                )
    finally:
        env.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Manually control a trained Wheelbipe policy in MuJoCo."
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path, help="Path to model_*.pt checkpoint"
    )
    parser.add_argument(
        "--config", type=Path, help="Optional config.yaml saved with the checkpoint"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args(argv)
    run(args.checkpoint, config_path=args.config, device=args.device)


if __name__ == "__main__":
    main()
