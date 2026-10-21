"""Play a trained Wheelbipe V14 Flat PPO policy in MuJoCo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from rsl_rl.runners import OnPolicyRunner

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.rl.vec_env import RslRlVecEnvAdapter
from unilab.training.experiment import patch_rsl_rl_resume_state
from unilab.utils.device import get_default_device
from unilab.utils.seed import apply_configured_training_seed


def _load_config(checkpoint: Path, config_path: Path | None) -> DictConfig:
    selected = config_path or checkpoint.parent / "config.yaml"
    if selected.is_file():
        loaded = OmegaConf.load(selected)
        if not OmegaConf.is_config(loaded):
            raise TypeError(f"Playback config must be a mapping: {selected}")
        return loaded

    config_dir = Path(__file__).resolve().parents[1] / "conf" / "ppo"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="config")


def _remove_gimbal_references(cfg: DictConfig) -> None:
    """Keep older checkpoint configs usable after the optional gimbal was removed."""
    robot = OmegaConf.select(cfg, "env.scene.entities.robot")
    if robot is None:
        return
    for field in ("body_names", "joint_names", "actuator_names", "geom_names"):
        values = OmegaConf.select(robot, field)
        if values is not None:
            OmegaConf.update(
                cfg,
                f"env.scene.entities.robot.{field}",
                [value for value in values if "gimbal" not in str(value)],
                force_add=True,
            )


def _resolve_device(cfg: DictConfig, requested: str | None) -> str:
    device = requested or str(OmegaConf.select(cfg, "training.device") or get_default_device())
    if torch.device(device).type not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported playback device: {device}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def play(
    checkpoint: Path,
    *,
    config_path: Path | None = None,
    device: str | None = None,
    num_envs: int = 1,
    steps: int = 200,
    output: Path = Path("playback.mp4"),
    render_mode: str = "record",
    cam_distance: float | None = None,
    cam_elevation: float | None = None,
    cam_azimuth: float | None = None,
    sim_backend: str | None = None,
) -> None:
    """Run a deterministic policy and optionally render the resulting rollout."""
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if num_envs <= 0 or steps <= 0:
        raise ValueError("num_envs and steps must be positive")
    if render_mode not in {"record", "none"}:
        raise ValueError("render_mode must be 'record' or 'none'")

    cfg = _load_config(checkpoint, config_path)
    _remove_gimbal_references(cfg)
    if sim_backend is not None:
        if sim_backend not in {"mujoco", "mjwarp"}:
            raise ValueError(f"Unsupported simulation backend: {sim_backend}")
        OmegaConf.update(cfg, "training.sim_backend", sim_backend, force_add=True)
    if str(cfg.training.task_name) != "WheelbipeV14Flat" or str(cfg.training.sim_backend) not in {
        "mujoco",
        "mjwarp",
    }:
        raise ValueError("This checkout supports only WheelbipeV14Flat with MuJoCo or MJWarp")
    learner_device = _resolve_device(cfg, device)
    OmegaConf.update(cfg, "training.device", learner_device, force_add=True)
    OmegaConf.update(cfg, "training.play_steps", steps, force_add=True)
    OmegaConf.update(cfg, "training.play_env_num", num_envs, force_add=True)
    OmegaConf.update(cfg, "training.play_render_mode", render_mode, force_add=True)
    OmegaConf.update(cfg, "algo.num_envs", num_envs, force_add=True)
    manager_device = "cuda" if str(cfg.training.sim_backend) == "mjwarp" else "cpu"
    OmegaConf.update(cfg, "env.manager_torch_device", manager_device, force_add=True)
    if cam_distance is not None:
        OmegaConf.update(cfg, "training.cam_distance", cam_distance, force_add=True)
    if cam_elevation is not None:
        OmegaConf.update(cfg, "training.cam_elevation", cam_elevation, force_add=True)
    if cam_azimuth is not None:
        OmegaConf.update(cfg, "training.cam_azimuth", cam_azimuth, force_add=True)

    seed_info = apply_configured_training_seed(cfg)
    env_overrides = BackendAdapter(
        cfg, root_dir=Path(__file__).resolve().parents[3]
    ).build_task_env_cfg_override()
    env_overrides["manager_torch_device"] = manager_device
    env_overrides["seed"] = seed_info.effective_seed
    env = create_env(cfg, num_envs=num_envs, env_cfg_override=env_overrides)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

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

        episode_returns = torch.zeros(num_envs, device=learner_device)
        completed_returns: list[float] = []

        def initialize() -> Any:
            obs, _ = wrapped_env.reset()
            episode_returns.zero_()
            completed_returns.clear()
            return obs

        def step(obs: Any) -> Any:
            with torch.inference_mode():
                actions = policy(obs, stochastic_output=False)
                next_obs, rewards, dones, _ = wrapped_env.step(actions)
            episode_returns.add_(rewards)
            done_indices = torch.nonzero(dones, as_tuple=False).flatten()
            if len(done_indices) > 0:
                completed_returns.extend(episode_returns[done_indices].detach().cpu().tolist())
                episode_returns[done_indices] = 0
            return next_obs

        if render_mode == "record":
            print(f"Rendering {steps} steps to {output}")
            rendered = env.run_playback_mode(
                play_render_mode="record",
                play_steps=steps,
                output_video=str(output),
                initialize=initialize,
                step=step,
                render_spacing=float(cfg.training.render_spacing),
                camera_kwargs={
                    "cam_distance": float(cfg.training.cam_distance),
                    "cam_elevation": float(cfg.training.cam_elevation),
                    "cam_azimuth": float(cfg.training.cam_azimuth),
                    "cam_lookat": cfg.training.cam_lookat,
                    "cam_tracking": bool(cfg.training.cam_tracking),
                    "cam_tracking_env_idx": int(cfg.training.cam_tracking_env_idx),
                    "cam_tracking_extra_envs": int(cfg.training.cam_tracking_extra_envs),
                },
            )
            if rendered is not None:
                print(f"Playback video: {rendered}")
        else:
            obs = initialize()
            for _ in range(steps):
                obs = step(obs)

        if completed_returns:
            values = torch.tensor(completed_returns)
            print(
                f"Completed episodes: {len(completed_returns)}; "
                f"mean return: {values.mean().item():.3f}; "
                f"std: {values.std(unbiased=False).item():.3f}"
            )
        else:
            print("No episode completed during playback; increase --steps for return statistics.")
    finally:
        env.close()


def main(
    checkpoint: str,
    *,
    config_path: str | None = None,
    device: str | None = None,
    num_envs: int = 1,
    steps: int = 200,
    output: str = "playback.mp4",
    render_mode: str = "record",
    cam_distance: float | None = None,
    cam_elevation: float | None = None,
    cam_azimuth: float | None = None,
    sim_backend: str | None = None,
) -> None:
    play(
        Path(checkpoint),
        config_path=Path(config_path) if config_path else None,
        device=device,
        num_envs=num_envs,
        steps=steps,
        output=Path(output),
        render_mode=render_mode,
        cam_distance=cam_distance,
        cam_elevation=cam_elevation,
        cam_azimuth=cam_azimuth,
        sim_backend=sim_backend,
    )
