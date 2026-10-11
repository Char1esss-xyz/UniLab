"""Sensor-driven reward terms shared by Manager-Based locomotion families.

Biped and quadruped locomotion families port legacy reward equations that read
IMU-style XML sensors by name.  This module owns those equations once, on top
of the ``SensorTermBase`` cold-path binding contract from ``manager_terms.py``:
the robot's sensor name is a cfg parameter (``params={"sensor_name": ...}``),
so each family binds its own sensors from its owner YAML instead of
subclassing.  Plain action/state penalties and the shared term bases live in
``manager_terms.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from unilab.managers.manager_base import ManagerTermBaseCfg

from .manager_terms import SensorTermBase, _command, _real, _state

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


def _torch_values(values: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(dtype=torch.float32)
    return torch.from_numpy(np.ascontiguousarray(values)).to(dtype=torch.float32)


class _Vec3SensorTerm(SensorTermBase):
    """Bind one named vec3 sensor on the cold path and read it per step."""

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"sensor_name"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        sensor_name = cfg.params.get("sensor_name")
        if not isinstance(sensor_name, str) or not sensor_name:
            raise ValueError(f"{self.name} sensor_name must be a non-empty string")
        self._sensor_name = sensor_name
        self._sensor = self._bind((sensor_name,))
        if self._sensor.dimensions != (3,):
            raise ValueError(
                f"{self.name} sensor '{sensor_name}' must expose 3 values; received "
                f"{self._sensor.dimensions} on backend '{self._sensor.backend_type}'"
            )

    def _read_sensor(self, env: ManagerBasedRlEnv) -> np.ndarray:
        return _state(
            self.name,
            f"sensor '{self._sensor_name}'",
            self._read(self._sensor, self.name),
            (env.num_envs, 3),
        )

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return (self._sensor_name,)

    def _read_sensor_tensor(self, env: ManagerBasedRlEnv) -> np.ndarray | torch.Tensor:
        read_plan = getattr(env.scene, "_tensor_read_plan", None)
        if read_plan is None or not set(self.tensor_sensor_names).issubset(
            set(getattr(read_plan, "sensor_names", {}).get("robot", ()))
        ):
            return self._read_sensor(env)
        if not hasattr(read_plan, "sensor_tensor_views"):
            return self._read_sensor(env)
        entity = env.scene["robot"]
        values = read_plan.sensor_tensor_views(entity, self.tensor_sensor_names).values[
            self._sensor_name
        ]
        if tuple(values.shape) != (env.num_envs, 3):
            raise ValueError(
                f"{self.name} sensor '{self._sensor_name}' must have shape "
                f"(num_envs, 3); got {tuple(values.shape)}"
            )
        return values


class _CommandTrackingTerm(_Vec3SensorTerm):
    """``exp(-error / tracking_sigma)`` tracking against a command channel."""

    _allowed_params = frozenset({"sensor_name", "tracking_sigma", "command_name"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._sigma = _real(
            self.name,
            "tracking_sigma",
            cfg.params.get("tracking_sigma", 0.25),
            minimum=0.0,
            strict_minimum=True,
        )
        self._command_name = cfg.params.get("command_name", "twist")
        _command(env, self.name, self._command_name)  # Fail closed at construction.


class track_lin_vel(_CommandTrackingTerm):
    """Exponential reward for tracking commanded xy linear velocity."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray | torch.Tensor:
        del params
        linvel = self._read_sensor_tensor(env)
        command = _command(env, self.name, self._command_name)
        linvel_tensor = _torch_values(linvel)
        command_tensor = (
            _torch_values(command).to(device=linvel_tensor.device)
            if isinstance(command, np.ndarray)
            else command.to(device=linvel_tensor.device, dtype=linvel_tensor.dtype)
        )
        error = (command_tensor[:, :2] - linvel_tensor[:, :2]).square().sum(dim=1)
        return torch.exp(-error / self._sigma)


class track_ang_vel(_CommandTrackingTerm):
    """Exponential reward for tracking commanded yaw angular velocity."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray | torch.Tensor:
        del params
        gyro = self._read_sensor_tensor(env)
        command = _command(env, self.name, self._command_name)
        gyro_tensor = _torch_values(gyro)
        command_tensor = (
            _torch_values(command).to(device=gyro_tensor.device)
            if isinstance(command, np.ndarray)
            else command.to(device=gyro_tensor.device, dtype=gyro_tensor.dtype)
        )
        error = (command_tensor[:, 2] - gyro_tensor[:, 2]).square()
        return torch.exp(-error / self._sigma)


class lin_vel_z(_Vec3SensorTerm):
    """Penalty for vertical (z) linear velocity."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray | torch.Tensor:
        del params
        values = self._read_sensor_tensor(env)[:, 2]
        return _torch_values(values).square()


class ang_vel_xy(_Vec3SensorTerm):
    """Penalty for roll/pitch angular velocity."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray | torch.Tensor:
        del params
        return _torch_values(self._read_sensor_tensor(env)[:, :2]).square().sum(dim=1)


class orientation(_Vec3SensorTerm):
    """Penalty for deviation from upright (horizontal upvector components)."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray | torch.Tensor:
        del params
        upvector = self._read_sensor_tensor(env)
        return _torch_values(upvector[:, :2]).square().sum(dim=1)


__all__ = [
    "ang_vel_xy",
    "lin_vel_z",
    "orientation",
    "track_ang_vel",
    "track_lin_vel",
]
