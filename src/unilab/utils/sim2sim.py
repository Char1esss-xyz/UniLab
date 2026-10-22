"""Policy contract snapshots and validation shared by training and playback."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


class CrossBackendIncompatibleError(RuntimeError):
    """The target environment does not preserve the trained policy contract."""


ALLOWLIST = ["training.sim_backend", "env.scene", "training.play_steps"]
WARNING_LIST = ["reward", "env.ctrl_dt", "env.sim_dt"]
DENYLIST = [
    "training.task_name",
    "algo.obs_groups",
    "env.observations",
    "env.actions",
    "env.policy_observation_group",
    "env.critic_observation_group",
    "algo.actor.class_name",
    "algo.actor.hidden_dims",
    "algo.actor.activation",
    "algo.actor.obs_normalization",
    "algo.actor.distribution_cfg",
    "algo.critic.class_name",
    "algo.critic.hidden_dims",
    "algo.critic.activation",
    "algo.critic.obs_normalization",
]
SNAPSHOT_FIELDS = DENYLIST + WARNING_LIST


def extract_contract_snapshot(cfg: DictConfig) -> dict[str, Any]:
    """Capture resolved policy semantics without devices, paths or run counters."""
    snapshot: dict[str, Any] = {}
    for path in SNAPSHOT_FIELDS:
        value = OmegaConf.select(cfg, path)
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        snapshot[path] = value
    return snapshot


def _read_snapshot(run_dir: Path) -> dict[str, Any]:
    snapshot_path = run_dir / "run_config.json"
    snapshot: dict[str, Any] | None = None
    if snapshot_path.is_file():
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        raw_snapshot = data.get("contract_snapshot") if isinstance(data, dict) else None
        if not isinstance(raw_snapshot, dict):
            raise CrossBackendIncompatibleError(
                f"Invalid policy contract snapshot: {snapshot_path}"
            )
        snapshot = dict(raw_snapshot)

    # Runs produced before snapshots were restored still have their resolved
    # training config. Validate that source instead of trusting the play config.
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        cfg = OmegaConf.load(config_path)
        if not isinstance(cfg, DictConfig):
            raise CrossBackendIncompatibleError(f"Training config must be a mapping: {config_path}")
        config_snapshot = extract_contract_snapshot(cfg)
        if snapshot is None:
            return config_snapshot
        # Older snapshots covered fewer fields. Recover only absent fields from
        # their saved training config; the original snapshot stays authoritative.
        for path, value in config_snapshot.items():
            snapshot.setdefault(path, value)
    if snapshot is not None:
        return snapshot
    raise CrossBackendIncompatibleError(
        f"No training contract found in {run_dir}; keep run_config.json or the original "
        "config.yaml beside the checkpoint"
    )


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def resolve_sim2sim_config(
    source_run_dir: str | Path,
    target_cfg: DictConfig,
    *,
    strict: bool = True,
    algo_name: str | None = None,
) -> DictConfig:
    """Validate before allocating an environment, without modifying its config."""
    del algo_name
    try:
        source = _read_snapshot(Path(source_run_dir))
    except CrossBackendIncompatibleError as exc:
        if strict:
            raise
        warnings.warn(str(exc), stacklevel=2)
        return target_cfg
    target = extract_contract_snapshot(target_cfg)
    denials: list[str] = []
    for path in SNAPSHOT_FIELDS:
        if _normalize(source.get(path)) == _normalize(target.get(path)):
            continue
        difference = f"{path}: trained={source.get(path)!r}, requested={target.get(path)!r}"
        if path in DENYLIST:
            denials.append(difference)
        else:
            warnings.warn(f"Sim2Sim configuration differs: {difference}", stacklevel=2)
    if denials:
        message = "Policy contract mismatch:\n  " + "\n  ".join(denials)
        if strict:
            raise CrossBackendIncompatibleError(message)
        warnings.warn(message, stacklevel=2)
    return target_cfg


@contextmanager
def policy_load_dim_guard(
    *,
    env_obs_dim: int,
    env_action_dim: int,
    algo_name: str = "ppo",
) -> Iterator[None]:
    """Explain checkpoint shape errors while preserving unrelated load errors."""
    try:
        yield
    except (RuntimeError, ValueError) as exc:
        if "size mismatch for " not in str(exc).lower():
            raise
        raise CrossBackendIncompatibleError(
            f"Checkpoint dimensions do not fit {algo_name}: policy observations="
            f"{env_obs_dim}, actions={env_action_dim}. Original error: {exc}"
        ) from exc
