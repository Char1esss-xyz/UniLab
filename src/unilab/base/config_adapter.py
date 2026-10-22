from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from unilab.assets import prepare_model_file
from unilab.base import registry
from unilab.utils.reward import extract_reward_config


class BackendAdapter:
    def __init__(self, cfg: DictConfig, *, root_dir: str | Path | None = None) -> None:
        self.cfg = cfg
        self.root_dir = Path(root_dir) if root_dir is not None else None

    def build_task_env_cfg_override(self) -> dict[str, Any]:
        registry.ensure_registries()
        reward_target = registry.resolve_reward_override_field(str(self.cfg.training.task_name))
        env = OmegaConf.to_container(self.cfg.env, resolve=True)
        if not isinstance(env, dict):
            raise TypeError("env must be a mapping")
        if reward_target in env:
            raise ValueError("Declare reward terms only in the root reward mapping")
        overrides: dict[str, Any] = dict(
            extract_reward_config(self.cfg, target_field=reward_target)
        )
        overrides.update({str(key): value for key, value in env.items()})
        scene = overrides["scene"]
        scene["model_file"] = str(prepare_model_file(scene["model_file"], root_dir=self.root_dir))
        return overrides


def create_env(cfg: DictConfig, *, num_envs: int, env_cfg_override: dict[str, Any] | None = None):
    return registry.make(
        str(cfg.training.task_name),
        num_envs=num_envs,
        sim_backend=str(cfg.training.sim_backend),
        env_cfg_override=env_cfg_override,
    )
