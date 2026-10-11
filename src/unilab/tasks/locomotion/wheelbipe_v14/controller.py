# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch

from unilab.assets import ASSETS_ROOT_PATH
from unilab.managers.action_manager import ActionTerm, ActionTermCfg

from .math import (
    ClosedChainKinematics,
    Delay,
    inverse_kinematics,
    quat_apply,
    quat_conjugate,
    quat_from_euler,
    reward_terms,
    spring_force,
    yaw_from_quat,
)

POLICY_JOINTS = (
    "left_front1_joint",
    "left_rear1_joint",
    "right_front1_joint",
    "right_rear1_joint",
    "left_wheel_joint",
    "right_wheel_joint",
)


def tensor(values):
    if isinstance(values, torch.Tensor):
        return values.to(dtype=torch.float64).clone()
    return torch.tensor(values, dtype=torch.float64)


@dataclass(kw_only=True)
class WheelbipeMotorCfg(ActionTermCfg):
    domain_randomization: bool = True
    observation_noise: bool = True
    delays: bool = True
    steps_per_iteration: int = 24

    def build(self, env):
        return WheelbipeMotor(self, env)


class WheelbipeMotor(ActionTerm):
    requires_substep_state_feedback = True

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if env.device.type != "cpu":
            raise ValueError("Wheelbipe MuJoCo control requires manager_torch_device=cpu")
        self.backend = env.backend
        self.count = env.num_envs
        self.generator = env.torch_rng.generator
        asset = ASSETS_ROOT_PATH / "robots/wheelbipe_v14"
        self.names = json.loads((asset / "conversion.json").read_text())["joints"]
        self.joints = len(self.names)
        self.policy_ids = [self.names.index(name) for name in POLICY_JOINTS]
        self.reset_ids = [
            self.names.index(f"{side}_{joint}_joint")
            for joint in ("front1", "rear1", "front2", "front3", "front4", "rear2")
            for side in ("left", "right")
        ]
        self.spring_ids = [self.names.index(f"{side}_spring2_joint") for side in ("left", "right")]
        self.pitch_id = self.names.index("gimbal_pitch_joint")
        self.yaw_id = self.names.index("gimbal_yaw_joint")
        self.body_names = ["base_link"] + [name.replace("_joint", "_link") for name in self.names]
        self.body_ids = self.backend.get_body_ids(self.body_names)
        self.base_id = self.backend.get_body_id("base_link")
        self.wheel_ids = self.backend.get_body_ids(["left_wheel_link", "right_wheel_link"])
        self.contact_names = [f"contact_{name}" for name in self.body_names]
        self.bad_contacts = [
            index
            for index, name in enumerate(self.body_names)
            if name == "base_link" or "gimbal" in name or "guide" in name
        ]
        self.wheel_contacts = [
            self.body_names.index(f"{side}_wheel_link") for side in ("left", "right")
        ]
        self.home = tensor(self.backend.get_keyframe_qpos("home"))
        self.home_velocity = tensor(self.backend.get_init_qvel())
        self.kinematics = ClosedChainKinematics(asset / "scene_flat.xml", self.names)
        self.kp = self.zeros(self.joints)
        self.kd = self.zeros(self.joints) + 0.01
        self.kp[:, self.policy_ids[:4]] = 60
        self.kd[:, self.policy_ids[:4]] = 2
        self.kd[:, self.policy_ids[4:]] = 0.2
        self.kd[:, self.spring_ids] = 50
        self.kp[:, self.pitch_id] = 20
        self.kd[:, [self.pitch_id, self.yaw_id]] = 0.5
        self.nominal_kp = self.kp[0].clone()
        self.nominal_kd = self.kd[0].clone()
        self.limits = tensor(self.backend.get_actuator_ctrl_range())
        self.effort_scale = self.zeros(self.joints) + 1
        self.coulomb = self.zeros(self.joints)
        self.viscous = self.zeros(self.joints)
        self.actions = self.zeros(6)
        self.previous = self.zeros(6)
        self.before_previous = self.zeros(6)
        self.motor_torque = self.zeros(self.joints)
        self.last_gain_reset = torch.full((self.count,), -720, dtype=torch.int64)
        self.last_velocity = self.zeros(6)
        self.acceleration = self.zeros(6)
        self.spring_offset = self.zeros(2)
        self.yaw_target = self.zeros()
        self.command = self.zeros(3)
        self.height_command = self.zeros() + 0.3
        self.heading_target = self.zeros()
        self.heading_env = torch.zeros(self.count, dtype=torch.bool)
        self.standing_env = self.heading_env.clone()
        self.command_remaining = self.zeros()
        self.special = torch.zeros(self.count, dtype=torch.int64)
        self.age = self.zeros()
        self.zero_torque_until = self.zeros()
        self.temporary_command = self.zeros(3)
        self.ground_start = self.heading_env.clone()
        self.termination_count = torch.zeros(self.count, dtype=torch.int64)
        self.contacts = torch.zeros((3, self.count, len(self.body_names)), dtype=torch.float64)
        self.obs_delays = [
            Delay(self.count, width, 4, dtype=torch.float64) for width in (3, 3, 6, 6)
        ]
        self.action_delays = [Delay(self.count, width, 3, dtype=torch.float64) for width in (4, 2)]
        self.observed = [self.zeros(width) for width in (3, 3, 6, 6)]
        self.next_push = self.uniform(5, 10, (self.count,))
        self.next_force = self.uniform(5, 10, (self.count,))
        self.external_force = self.zeros(3)
        self.external_torque = self.zeros(3)
        self.observation_cache = None
        self.step_cache = -1
        self.terms = {}
        self.terminated = self.heading_env.clone()
        self._initialize_randomization()
        if cfg.domain_randomization:
            self._entity.bind_body_mass_write(term_name="wheelbipe_mass")
            self._entity.bind_body_ipos_write(term_name="wheelbipe_com")
            self._entity.bind_body_inertia_write(term_name="wheelbipe_inertia")
            self._entity.bind_geom_friction_write(term_name="wheelbipe_friction")
            self._entity.bind_root_linear_velocity_delta(term_name="wheelbipe_push")
            _, self.wrench_body_ids = self._entity.bind_body_wrench(
                [0], torque=True, term_name="wheelbipe_force"
            )

    def zeros(self, *shape):
        return torch.zeros((self.count, *shape), dtype=torch.float64)

    def uniform(self, low, high, shape):
        low, high = tensor(low), tensor(high)
        return low + (high - low) * torch.rand(shape, generator=self.generator, dtype=torch.float64)

    @property
    def action_dim(self):
        return 6

    @property
    def raw_action(self):
        return self.actions.float()

    def _initialize_randomization(self):
        mass = tensor(self.backend.get_reset_term_default("body_mass"))
        self.mass = mass.expand(self.count, -1).clone()
        self.inertia = (
            tensor(self.backend.get_reset_term_default("body_inertia"))
            .expand(self.count, -1, -1)
            .clone()
        )
        self.com = (
            tensor(self.backend.get_reset_term_default("body_ipos"))
            .expand(self.count, -1, -1)
            .clone()
        )
        self.friction = (
            tensor(self.backend.get_reset_term_default("geom_friction"))
            .expand(self.count, -1, -1)
            .clone()
        )
        self.geom_bodies = torch.as_tensor(self.backend.get_geom_body_ids())
        self.geom_ids = [
            self.backend.get_geom_id(name)
            for name in self._env.cfg.scene.entities[self.cfg.entity_name].geom_names
        ]
        self.wheel_material = tensor([1, 1, 0]).expand(self.count, 2, 3).clone()
        self.base_mass_scale = self.zeros(1) + 1
        if not self.cfg.domain_randomization:
            return
        ids = self.backend.get_body_ids([name for name in self.body_names if "guide" not in name])
        self.mass[:, ids] *= self.uniform(0.9, 1.1, (self.count, len(ids)))
        self.mass[:, self.base_id] = mass[self.base_id] * self.uniform(0.9, 1.3, (self.count,))
        self.inertia[:, 1:] *= (self.mass[:, 1:] / mass[None, 1:])[:, :, None]
        self.base_mass_scale = self.mass[:, self.base_id : self.base_id + 1] / mass[self.base_id]
        self.com[:, self.base_id] += self.uniform(
            [-0.04, -0.02, -0.02], [0.04, 0.02, 0.02], (self.count, 3)
        )
        for name, body_id in zip(self.body_names, self.body_ids, strict=True):
            if "wheel" in name:
                buckets = self.uniform([0.5, 0.4, 0.02], [1.2, 1, 0.2], (64, 3))
                buckets[:, 1] = torch.minimum(buckets[:, 1], buckets[:, 0])
                material = buckets[torch.randint(64, (self.count,), generator=self.generator)]
                self.wheel_material[:, int(name.startswith("right"))] = material
            elif name == "base_link":
                material = self.uniform([0.01, 0.01, 0], [0.1, 0.1, 0], (self.count, 3))
            elif "guide" in name:
                material = self.uniform([0.1, 0.1, 0], [0.7, 0.7, 0], (self.count, 3))
            else:
                continue
            material[:, 1] = torch.minimum(material[:, 1], material[:, 0])
            self.friction[:, self.geom_bodies == int(body_id), 0] = material[:, 1, None]
        for index, name in enumerate(self.names):
            if name.endswith(("front1_joint", "rear1_joint")):
                friction, viscous = (0.25, 1), (0.05, 0.2)
            elif name.endswith("wheel_joint"):
                friction, viscous = (0.05, 0.25), (0, 0.01)
            elif "gimbal" in name:
                friction, viscous = (0.002, 0.01), (0.002, 0.01)
            elif "guide" in name or name.endswith("spring2_joint"):
                continue
            else:
                friction, viscous = (0.05, 0.1), (0.01, 0.025)
            self.coulomb[:, index] = self.uniform(*friction, (self.count,))
            self.viscous[:, index] = self.uniform(*viscous, (self.count,))

    def reset_simulation(self, ids):
        count = len(ids)
        qpos = self.home.expand(count, -1).clone()
        qvel = self.home_velocity.expand(count, -1).clone()
        qpos[:, 3:7] = quat_from_euler(
            self.uniform([-0.15, -0.15, -3.14], [0.15, 0.15, 3.14], (count, 3))
        )
        length = self.uniform(0.13, 0.32, (count, 2))
        angle = self.uniform(-math.pi / 2, 3 * math.pi / 4, (count, 2))
        ground = self.uniform(0, 1, (count,)) < 0.5
        ground_count = int(ground.sum())
        if ground_count:
            negative = self.uniform(0, 1, (ground_count, 2)) >= 0.6
            height = self.uniform(-0.06, 0.12, (ground_count, 2))
            height[negative] = self.uniform(-0.06, 0, (int(negative.sum()),))
            length[ground] = self.uniform(0.14, 0.36, (ground_count, 2))
            angle[ground] = torch.acos(
                (height / length[ground]).clamp(-1 + 1e-6, 1 - 1e-6)
            ) * torch.where(negative, -1, 1)
            qpos[ground, :3] = tensor([0, 0, 0.25])
            qpos[ground, 3:7] = tensor([1, 0, 0, 0])
        qpos[:, [7 + index for index in self.reset_ids]] = (
            inverse_kinematics(length, angle).transpose(1, 2).reshape(count, 12)
        )
        self.kinematics.close_springs(qpos[:, 7:])
        qpos[:, [7 + index for index in self.policy_ids[4:]]] = self.uniform(
            -2 * math.pi, 2 * math.pi, (count, 2)
        )
        qvel[:, [6 + index for index in self.policy_ids[:4]]] = self.uniform(
            -math.pi / 2, math.pi / 2, (count, 4)
        )
        qvel[:, [6 + index for index in self.policy_ids[4:]]] = self.uniform(-50, 50, (count, 2))
        gain_ids = ids[self._env.step_counter - self.last_gain_reset[ids] >= 720]
        gain_count = len(gain_ids)
        self.kp[gain_ids] = self.nominal_kp
        self.kd[gain_ids] = self.nominal_kd
        if self.cfg.domain_randomization:
            self.kp[gain_ids] *= self.uniform(0.75, 1.25, (gain_count, self.joints))
            self.kd[gain_ids] *= self.uniform(0.75, 1.25, (gain_count, self.joints))
            self.kd[gain_ids[:, None], self.spring_ids] = 50 * self.uniform(
                0.5, 1.5, (gain_count, 2)
            )
            self.effort_scale[gain_ids[:, None], self.policy_ids[:4]] = self.uniform(
                0.8, 1.1, (gain_count, 4)
            )
            self.effort_scale[gain_ids[:, None], self.policy_ids[4:]] = self.uniform(
                0.9, 1.1, (gain_count, 2)
            )
            self._entity.write_body_mass_to_sim(self.mass[ids][:, self.body_ids], env_ids=ids)
            self._entity.write_body_ipos_to_sim(self.com[ids][:, self.body_ids], env_ids=ids)
            self._entity.write_body_inertia_to_sim(self.inertia[ids][:, self.body_ids], env_ids=ids)
            self._entity.write_geom_friction_to_sim(
                self.friction[ids][:, self.geom_ids], env_ids=ids
            )
        self.last_gain_reset[gain_ids] = self._env.step_counter
        self.spring_offset[ids] = self.uniform(-50, 50, (count, 2))
        self.yaw_target[ids] = self.uniform(-math.pi, math.pi, (count,))
        qvel[:, 6 + self.yaw_id] = self.yaw_target[ids]
        self.age[ids] = 0
        self.ground_start[ids] = ground
        self.zero_torque_until[ids] = ground.double() * 0.2
        self.temporary_command[ids] = self.uniform([-1, 0, -1], [1, 0, 1], (count, 3))
        for values in (
            self.actions,
            self.previous,
            self.before_previous,
            self.motor_torque,
            self.termination_count,
            self.external_force,
            self.external_torque,
        ):
            values[ids] = 0
        self.contacts[:, ids] = 0
        self._entity.write_root_link_pose_to_sim(qpos[:, :7], env_ids=ids)
        self._entity.write_root_link_velocity_to_sim(qvel[:, :6], env_ids=ids)
        self._entity.write_joint_state_to_sim(qpos[:, 7:], qvel[:, 6:], env_ids=ids)
        self.resample_commands(ids)
        self.observation_cache = None

    def reset(self, env_ids=None):
        ids = torch.arange(self.count) if env_ids is None else env_ids
        if isinstance(ids, slice):
            ids = torch.arange(self.count)[ids]
        self.update_heading()
        raw = self.raw_observation()
        for delay, values in zip(self.obs_delays, raw, strict=True):
            lag = (
                torch.randint(1, 4, (len(ids),), generator=self.generator)
                if self.cfg.delays
                else torch.zeros(len(ids), dtype=torch.int64)
            )
            delay.reset(ids, values[ids], lag)
        for delay, width in zip(self.action_delays, (4, 2), strict=True):
            lag = (
                torch.randint(1, 3, (len(ids),), generator=self.generator)
                if self.cfg.delays
                else torch.zeros(len(ids), dtype=torch.int64)
            )
            delay.reset(ids, torch.zeros((len(ids), width), dtype=torch.float64), lag)
        for observed, values in zip(self.observed, raw, strict=True):
            observed[ids] = values[ids]
        self.last_velocity[ids] = raw[-1][ids]
        self.acceleration[ids] = 0
        self.observation_cache = None

    def resample_commands(self, ids):
        count = len(ids)
        self.command[ids] = self.uniform([-2.7, 0, -2 * math.pi], [2.7, 0, 2 * math.pi], (count, 3))
        self.standing_env[ids] = self.uniform(0, 1, (count,)) < 0.1
        self.heading_env[ids] = self.uniform(0, 1, (count,)) < 0.5
        self.heading_target[ids] = self.uniform(-math.pi, math.pi, (count,))
        self.height_command[ids] = self.uniform(0.2, 0.42, (count,))
        self.command_remaining[ids] = self.uniform(5, 15, (count,))
        self.special[ids] = 0
        iteration = self._env.step_counter // self.cfg.steps_per_iteration
        draw = self.uniform(0, 1, (count,))
        cursor = 0.0
        for mode, threshold, probability, velocity_range, yaw_range in (
            (1, 3000, 0.15, (-0.1, 0.1), (2 * math.pi, 3.25 * math.pi)),
            (2, 4000, 0.15, (-0.1, 0.1), (3.25 * math.pi, 4.5 * math.pi)),
            (3, 2000, 0.3, (2, 3), (0, 2 * math.pi)),
        ):
            if iteration < threshold:
                continue
            chosen = ids[
                (draw >= cursor)
                & (draw < cursor + probability)
                & (self.age[ids] >= 5)
                & ~self.standing_env[ids]
            ]
            cursor += probability
            self.special[chosen] = mode
            self.command[chosen, 0] = self.uniform(*velocity_range, (len(chosen),))
            if mode == 3:
                self.command[chosen, 0] *= (
                    torch.randint(2, (len(chosen),), generator=self.generator) * 2 - 1
                )
            self.command[chosen, 2] = self.uniform(*yaw_range, (len(chosen),)) * (
                torch.randint(2, (len(chosen),), generator=self.generator) * 2 - 1
            )
        self.update_heading()

    def update_heading(self):
        normal = self.special == 0
        heading = self.heading_env & normal
        error = (
            self.heading_target - yaw_from_quat(tensor(self.backend.get_base_quat())) + math.pi
        ) % (2 * math.pi) - math.pi
        self.command[heading, 2] = (5 * error[heading]).clamp(-2 * math.pi, 2 * math.pi)
        self.command[self.standing_env & normal] = 0

    def effective_command(self):
        command = self.command.clone()
        temporary = self.ground_start & (self.age < 1.5)
        command[temporary] = self.temporary_command[temporary]
        return command

    def raw_observation(self):
        quaternion = tensor(self.backend.get_base_quat())
        gravity = quat_apply(quat_conjugate(quaternion), tensor([0, 0, -1]).expand(self.count, 3))
        position = tensor(self.backend.get_dof_pos())[:, self.policy_ids]
        position[:, 4:] = 0
        velocity = tensor(self.backend.get_dof_vel())[:, self.policy_ids]
        return tensor(self.backend.get_base_ang_vel()), gravity, position, velocity

    def record_contacts(self):
        forces = tensor(self.backend.get_sensor_data_batch(self.contact_names)).reshape(
            self.count, len(self.body_names), 3
        )
        self.contacts[:-1] = self.contacts[1:].clone()
        self.contacts[-1] = torch.linalg.vector_norm(forces, dim=-1)

    def process_actions(self, actions):
        self.before_previous.copy_(self.previous)
        self.previous.copy_(self.actions)
        self.actions.copy_(actions)
        self.observation_cache = None

    def apply_actions(self):
        position = tensor(self.backend.get_dof_pos())
        velocity = tensor(self.backend.get_dof_vel())
        self.record_contacts()
        self.acceleration.copy_(
            (velocity[:, self.policy_ids] - self.last_velocity) / self._env.cfg.sim_dt
        )
        self.last_velocity.copy_(velocity[:, self.policy_ids])
        leg_target = self.action_delays[0].compute((0.5 * self.actions[:, :4]).clamp(-3.14, 3.14))
        wheel_target = self.action_delays[1].compute((10 * self.actions[:, 4:]).clamp(-150, 150))
        ctrl = -self.kd * velocity
        legs, wheels = self.policy_ids[:4], self.policy_ids[4:]
        ctrl[:, legs] += self.kp[:, legs] * (leg_target - position[:, legs])
        ctrl[:, wheels] += self.kd[:, wheels] * wheel_target
        disabled = (self.age < self.zero_torque_until).nonzero().flatten()
        ctrl[disabled[:, None], self.policy_ids] = 0
        ctrl[:, self.spring_ids] += spring_force(position[:, self.spring_ids], self.spring_offset)
        ctrl[:, self.pitch_id] += self.kp[:, self.pitch_id] * (-0.5 - position[:, self.pitch_id])
        ctrl[:, self.yaw_id] += self.kd[:, self.yaw_id] * self.yaw_target
        ctrl *= self.effort_scale
        self.motor_torque.copy_(ctrl.clamp(self.limits[:, 0], self.limits[:, 1]))
        ctrl -= self.coulomb * torch.tanh(velocity / 0.01) + self.viscous * velocity
        self._entity.set_joint_effort_target(
            ctrl.clamp(self.limits[:, 0], self.limits[:, 1]).float()
        )
        for observed, delay, values in zip(
            self.observed, self.obs_delays, self.raw_observation(), strict=True
        ):
            observed.copy_(delay.compute(values))
        if self.cfg.domain_randomization:
            quaternion = tensor(self.backend.get_base_quat())
            self._entity.apply_body_wrench_to_sim(
                quat_apply(quaternion, self.external_force)[:, None],
                quat_apply(quaternion, self.external_torque)[:, None],
                self.wrench_body_ids,
                term_name="wheelbipe_force",
            )

    def evaluate(self):
        if self.step_cache == self._env.common_step_counter:
            return
        self.step_cache = self._env.common_step_counter
        self.record_contacts()
        gyro, gravity, _, velocity = self.raw_observation()
        self.acceleration.copy_((velocity - self.last_velocity) / self._env.cfg.sim_dt)
        quaternion = tensor(self.backend.get_base_quat())
        roll = torch.atan2(
            2 * (quaternion[:, 0] * quaternion[:, 1] + quaternion[:, 2] * quaternion[:, 3]),
            1 - 2 * (quaternion[:, 1].square() + quaternion[:, 2].square()),
        )
        pitch = torch.asin(
            (2 * (quaternion[:, 0] * quaternion[:, 2] - quaternion[:, 3] * quaternion[:, 1])).clamp(
                -1, 1
            )
        )
        bad = (
            (self.contacts[-1][:, self.bad_contacts] > 1).any(-1)
            | (roll.abs() > math.radians(40))
            | (pitch.abs() > math.radians(40))
        )
        self.termination_count.copy_(torch.where(bad, self.termination_count + 1, 0))
        linvel = quat_apply(quat_conjugate(quaternion), tensor(self.backend.get_base_lin_vel()))
        finite = (
            torch.isfinite(tensor(self.backend.get_dof_pos())).all(-1)
            & torch.isfinite(tensor(self.backend.get_dof_vel())).all(-1)
            & torch.isfinite(gyro).all(-1)
            & torch.isfinite(linvel).all(-1)
        )
        self.terminated = (
            (self.termination_count >= 20)
            | ~finite
            | (velocity.abs().amax(-1) > 500)
            | (gyro.abs().amax(-1) > 200)
            | (linvel.abs().amax(-1) > 100)
        )
        self.terms = reward_terms(
            self.effective_command(),
            linvel,
            gyro,
            gravity,
            tensor(self.backend.get_base_pos())[:, 2],
            self.height_command,
            tensor(self.backend.get_body_pos_b(self.wheel_ids)),
            velocity,
            self.acceleration,
            self.motor_torque[:, self.policy_ids],
            self.actions,
            self.previous,
            self.before_previous,
            (self.contacts.amax(0)[:, self.bad_contacts] > 3).any(-1),
            self.terminated,
        )

    def advance_commands(self):
        self.age += self._env.cfg.ctrl_dt
        self.command_remaining -= self._env.cfg.ctrl_dt
        ids = (self.command_remaining <= 0).nonzero().flatten()
        if len(ids):
            self.resample_commands(ids)
        self.update_heading()
        if self.cfg.domain_randomization:
            time = self._env.common_step_counter * self._env.cfg.ctrl_dt
            ids = (time >= self.next_push).nonzero().flatten()
            if len(ids):
                delta = torch.zeros((len(ids), 3), dtype=torch.float64)
                delta[:, :2] = (
                    self.uniform(-0.25, 0.25, (len(ids), 2))
                    - tensor(self.backend.get_base_lin_vel())[ids, :2]
                )
                self._entity.apply_root_velocity_delta_to_sim(
                    delta, None, env_ids=ids, term_name="wheelbipe_push"
                )
                self.next_push[ids] = time + self.uniform(5, 10, (len(ids),))
            ids = (time >= self.next_force).nonzero().flatten()
            if len(ids):
                self.external_force[ids] = self.uniform(-10, 10, (len(ids), 3))
                self.external_torque[ids] = self.uniform(-1, 1, (len(ids), 3))
                self.next_force[ids] = time + self.uniform(5, 10, (len(ids),))
        self.observation_cache = None

    def observations(self):
        if self.observation_cache is not None:
            return self.observation_cache
        gyro, gravity, position, velocity = self.raw_observation()
        delayed = [values.clone() for values in self.observed]
        if self.cfg.observation_noise:
            for values, noise in zip(
                delayed, (0.25, 0.05, 0.025, tensor([0.5] * 4 + [1] * 2)), strict=True
            ):
                values += self.uniform(-1, 1, values.shape) * noise
        mode = self.zeros(7)
        mode[:, 0] = 1
        command = self.effective_command().clamp(-100, 100)
        height = self.height_command[:, None].clamp(0, 1) * 5

        def basic(angular, projected, joint_position, joint_velocity):
            return torch.cat(
                [
                    command,
                    height,
                    angular.clamp(-100, 100) * 0.5,
                    projected.clamp(-100, 100),
                    joint_position.clamp(-100, 100),
                    joint_velocity.clamp(-200, 200) * 0.1,
                    self.actions.clamp(-100, 100),
                ],
                dim=-1,
            )

        policy = torch.cat([basic(*delayed), mode], dim=-1)
        quaternion = tensor(self.backend.get_base_quat())
        linvel = quat_apply(quat_conjugate(quaternion), tensor(self.backend.get_base_lin_vel()))
        relative = (
            tensor(self.backend.get_body_lin_vel_w(self.wheel_ids))
            - tensor(self.backend.get_base_lin_vel())[:, None]
        )
        relative = quat_apply(quat_conjugate(quaternion[:, None]), relative)
        relative[:, :, 1] = 0
        privileged = torch.cat(
            [
                self.kp[:, self.policy_ids].clamp(-100, 100),
                self.kd[:, self.policy_ids].clamp(-100, 100),
                self.motor_torque[:, self.policy_ids].clamp(-100, 100) * 0.05,
                torch.stack([delay.lag for delay in self.obs_delays], dim=-1),
                torch.stack([delay.lag for delay in self.action_delays], dim=-1),
                relative.reshape(self.count, 6).clamp(-100, 100),
                (self.contacts.amax(0)[:, self.wheel_contacts] > 5).double(),
                self.base_mass_scale,
                self.wheel_material.reshape(self.count, 6),
            ],
            dim=-1,
        )
        critic = torch.cat(
            [
                basic(gyro, gravity, position, velocity),
                linvel.clamp(-100, 100),
                tensor(self.backend.get_base_pos())[:, 2:3].clamp(-10, 10) * 5,
                mode,
                privileged,
            ],
            dim=-1,
        )
        self.observation_cache = {"policy": policy.float(), "critic": critic.float()}
        return self.observation_cache


def motor(env):
    return env.action_manager.get_term("motors")


def reset_robot(env, env_ids):
    motor(env).reset_simulation(env_ids)


def advance_commands(env, env_ids=None, dt=None):
    del env_ids, dt
    motor(env).advance_commands()


def observe(env, group):
    return motor(env).observations()[group]


def reward(env, name):
    controller = motor(env)
    controller.evaluate()
    return controller.terms[name].float()


def terminated(env):
    controller = motor(env)
    controller.evaluate()
    return controller.terminated


def time_out(env):
    return env.episode_length_buf >= env.max_episode_length
