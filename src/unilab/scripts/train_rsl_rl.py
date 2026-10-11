"""Direct RSL-RL PPO training for the Wheelbipe V14 Flat task."""

from __future__ import annotations

from datetime import datetime, timezone
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


def _learner_device(cfg: DictConfig) -> str:
    configured = OmegaConf.select(cfg, "training.device")
    device = str(configured) if configured else get_default_device()
    if torch.device(device).type not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported learner device: {device}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def train(cfg: DictConfig) -> Path:
    if (
        str(cfg.training.task_name) != "WheelbipeV14Flat"
        or str(cfg.training.sim_backend) != "mujoco"
    ):
        raise ValueError("This checkout supports only WheelbipeV14Flat with MuJoCo")
    if int(cfg.algo.num_envs) <= 0 or int(cfg.algo.max_iterations) <= 0:
        raise ValueError("algo.num_envs and algo.max_iterations must be positive")
    device = _learner_device(cfg)
    seed_info = apply_configured_training_seed(cfg)
    env_overrides = BackendAdapter(
        cfg, root_dir=Path(__file__).resolve().parents[3]
    ).build_task_env_cfg_override()
    env_overrides["manager_torch_device"] = "cpu"
    env_overrides["seed"] = seed_info.effective_seed
    OmegaConf.update(cfg, "training.device", device, force_add=True)
    OmegaConf.update(cfg, "env.manager_torch_device", "cpu", force_add=True)
    OmegaConf.update(cfg, "env.seed", seed_info.effective_seed, force_add=True)
    resume_value = OmegaConf.select(cfg, "algo.resume_path")
    resume_path = Path(str(resume_value)).expanduser().resolve() if resume_value else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    log_root_value = OmegaConf.select(cfg, "training.log_root")
    log_root = Path(str(log_root_value)).expanduser() if log_root_value else Path("logs/rsl_rl_ppo")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
    log_dir = (log_root / "WheelbipeV14Flat" / f"{timestamp}_mujoco").resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, log_dir / "config.yaml", resolve=True)
    env = create_env(cfg, num_envs=int(cfg.algo.num_envs), env_cfg_override=env_overrides)
    try:
        wrapped_env = RslRlVecEnvAdapter(env, device=device)
        raw_train_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
        if not isinstance(raw_train_cfg, dict):
            raise TypeError("algo must resolve to a mapping")
        train_cfg = cast(dict[str, Any], raw_train_cfg)
        train_cfg["logger"] = "tensorboard"
        patch_rsl_rl_resume_state()
        runner = OnPolicyRunner(
            cast(Any, wrapped_env), train_cfg, log_dir=str(log_dir), device=device
        )
        if resume_path is not None:
            print(f"Resuming from {resume_path}")
            runner.load(str(resume_path), map_location=device)
        print(f"Training WheelbipeV14Flat; learner={device}; simulation=cpu; logs={log_dir}")
        runner.learn(
            num_learning_iterations=int(cfg.algo.max_iterations), init_at_random_ep_len=True
        )
        runner.save(str(log_dir / f"model_{runner.current_learning_iteration}.pt"))
    finally:
        env.close()
    return log_dir


def main(overrides: list[str] | None = None) -> None:
    config_dir = Path(__file__).resolve().parents[1] / "conf" / "ppo"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides or [])
    train(cfg)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
