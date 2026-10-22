from __future__ import annotations

from typing import TYPE_CHECKING, Any

import unisim
from unisim.backend.base import SimBackend

if TYPE_CHECKING:
    from unilab.base.base import EnvCfg
    from unilab.base.scene import SceneCfg


def env_backend_kwargs(cfg: EnvCfg) -> dict[str, Any]:
    return {
        "cpu_ids": cfg.cpu_ids,
        "post_step_forward_sensor": True,
        "mjwarp_nconmax": cfg.mjwarp_nconmax,
        "mjwarp_njmax": cfg.mjwarp_njmax,
    }


def create_backend(
    backend_type: str,
    scene: SceneCfg,
    num_envs: int,
    sim_dt: float,
    *,
    body_state_required: bool = False,
    tensor_device: str | None = None,
    **kwargs: Any,
) -> SimBackend:
    if backend_type not in {"mujoco", "mjwarp"}:
        raise ValueError("This Wheelbipe checkout supports MuJoCo and MJWarp")
    if backend_type == "mjwarp" and tensor_device is not None:
        import torch
        import warp

        device = torch.device(tensor_device)
        if device.type != "cuda":
            raise ValueError("MJWarp requires a CUDA tensor device")
        # Warp and Torch have independent process-default devices. Bind both
        # before the backend allocates tensors, including for cuda:<index>.
        if device.index is not None:
            torch.cuda.set_device(device)
        warp.set_device(f"cuda:{torch.cuda.current_device()}")
    return unisim.create_backend(
        backend_type, scene, num_envs, sim_dt, body_state_required=body_state_required, **kwargs
    )
