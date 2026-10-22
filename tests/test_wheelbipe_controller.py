from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.locomotion.wheelbipe_v14.controller import WheelbipeMotor
from unilab.tasks.locomotion.wheelbipe_v14.math import (
    quat_apply,
    quat_conjugate,
    quat_from_euler,
)


def _make_env(sim_backend="mujoco", num_envs=2):
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "src/unilab/conf/ppo")):
        cfg = compose(
            config_name="config",
            overrides=[
                f"task=wheelbipe_v14_flat/{sim_backend}",
                "training.device=cpu",
                "env.actions.motors.observation_noise=false",
                "env.actions.motors.delays=false",
            ],
        )
    overrides = BackendAdapter(cfg, root_dir=root).build_task_env_cfg_override()
    overrides.update(manager_torch_device="cuda" if sim_backend == "mjwarp" else "cpu", seed=17)
    return create_env(cfg, num_envs=num_envs, env_cfg_override=overrides)


@pytest.fixture
def cpu_env():
    env = _make_env()
    try:
        env.reset()
        yield env
    finally:
        env.close()


@pytest.mark.parametrize("use_read_plan", [True, False])
def test_mjwarp_gyro_matches_mujoco_body_frame(cpu_env, use_read_plan):
    backend = cpu_env.backend
    state = backend.get_state()
    orientations = torch.tensor([[0.3, -0.2, np.pi / 2], [-0.4, 0.1, -np.pi / 2]])
    state["qpos"][:, 3:7] = quat_from_euler(orientations).numpy()
    state["qvel"][:] = 0
    state["qvel"][:, 3:6] = [[1, 2, 3], [-2, 1, 0.5]]
    backend.set_state(np.arange(2), state["qpos"], state["qvel"])
    expected_gyro = torch.as_tensor(backend.get_base_ang_vel()).clone().float()
    body_ids = backend.get_body_ids(["base_link"])
    view = SimpleNamespace(
        pos_w=torch.as_tensor(backend.get_body_pos_w(body_ids)).clone().float(),
        quat_w=torch.as_tensor(backend.get_body_quat_w(body_ids)).clone().float(),
        lin_vel_w=torch.as_tensor(backend.get_body_lin_vel_w(body_ids)).clone().float(),
        ang_vel_w=torch.as_tensor(backend.get_body_ang_vel_w(body_ids)).clone().float(),
    )
    assert not torch.allclose(view.ang_vel_w[:, 0], expected_gyro)

    # Use real backend state with a CPU stand-in for MJWarp's CUDA view boundary.
    motor = object.__new__(WheelbipeMotor)
    motor._device_resident = True
    motor._base_state_cache = None
    motor.device = torch.device("cpu")
    motor._entity = SimpleNamespace(body_tensor_view=lambda device, names: view)
    plan = SimpleNamespace(ready=True, body_tensor_view=lambda entity, names: view)
    motor._env = SimpleNamespace(
        scene=SimpleNamespace(_tensor_read_plan=plan if use_read_plan else None)
    )

    _, _, _, actual_gyro = motor._base_state()
    torch.testing.assert_close(actual_gyro, expected_gyro, atol=1e-6, rtol=1e-6)


def test_push_publishes_current_critic_velocity(cpu_env):
    backend = cpu_env.backend
    motor = cpu_env.action_manager.get_term("motors")
    state = backend.get_state()
    state["qvel"][:, :2] = [[2, 1], [-1, 2]]
    backend.set_state(np.arange(2), state["qpos"], state["qvel"])
    motor._clear_state_cache()
    motor.next_push.zero_()

    transition = cpu_env.step(torch.zeros((2, 6)))

    quaternion = torch.as_tensor(backend.get_base_quat()).clone()
    world_velocity = torch.as_tensor(backend.get_base_lin_vel()).clone()
    assert bool((world_velocity[:, :2].abs() <= 0.250001).all())
    current_velocity = quat_apply(quat_conjugate(quaternion), world_velocity)
    # The critic's first 28 values are the common command/state/action block.
    torch.testing.assert_close(
        transition.obs["critic"][:, 28:31], current_velocity.float(), atol=1e-6, rtol=1e-6
    )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MJWarp requires a CUDA device")
def test_mjwarp_reset_step_gyro_and_push_on_cuda():
    pytest.importorskip("mujoco_warp")
    env = _make_env("mjwarp", num_envs=4)
    try:
        env.reset()
        backend = env.backend
        motor = env.action_manager.get_term("motors")
        state = backend.get_state()
        orientations = torch.tensor([[0.3, -0.2, np.pi / 2], [-0.4, 0.1, -np.pi / 2]] * 2)
        state["qpos"][:, 3:7] = quat_from_euler(orientations).numpy()
        state["qvel"][:] = 0
        state["qvel"][:, :2] = [[2, 1], [-1, 2]] * 2
        state["qvel"][:, 3:6] = [[1, 2, 3], [-2, 1, 0.5]] * 2
        backend.set_state(np.arange(4), state["qpos"], state["qvel"])
        env.scene._invalidate_state_reads()
        motor._clear_state_cache()
        expected_gyro = torch.as_tensor(state["qvel"][:, 3:6], device=env.device)
        torch.testing.assert_close(motor.raw_observation()[0], expected_gyro, atol=1e-5, rtol=1e-5)

        motor.next_push.zero_()
        transition = env.step(torch.zeros((4, 6), device=env.device))

        assert all(bool(torch.isfinite(values).all()) for values in transition.obs.values())
        assert bool(torch.isfinite(transition.reward).all())
        quaternion = torch.as_tensor(backend.get_base_quat(), device=env.device)
        world_velocity = torch.as_tensor(backend.get_base_lin_vel(), device=env.device)
        assert bool((world_velocity[:, :2].abs() <= 0.250001).all())
        current_velocity = quat_apply(quat_conjugate(quaternion), world_velocity)
        torch.testing.assert_close(
            transition.obs["critic"][:, 28:31], current_velocity, atol=1e-5, rtol=1e-5
        )
    finally:
        env.close()
