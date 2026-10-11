from __future__ import annotations

from typing import TYPE_CHECKING, Any

import unisim
from unisim.backend.base import SimBackend

if TYPE_CHECKING:
    from unilab.base.base import EnvCfg
    from unilab.base.scene import SceneCfg


def env_backend_kwargs(cfg: EnvCfg) -> dict[str, Any]:
    return {"cpu_ids": cfg.cpu_ids, "post_step_forward_sensor": True}


def create_backend(
    backend_type: str,
    scene: SceneCfg,
    num_envs: int,
    sim_dt: float,
    *,
    body_state_required: bool = False,
    **kwargs: Any,
) -> SimBackend:
    if backend_type != "mujoco":
        raise ValueError("This Wheelbipe checkout supports only MuJoCo")
    return unisim.create_backend(
        backend_type, scene, num_envs, sim_dt, body_state_required=body_state_required, **kwargs
    )
