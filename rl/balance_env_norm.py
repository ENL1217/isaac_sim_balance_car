"""Same as balance_env.py but with explicit observation normalisation and a
simpler reward function. Use this if the original env's policy fails to
learn velocity tracking due to differently-scaled observation features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

WHEEL_JOINT_NAMES = ["left_wheel_joint", "right_wheel_joint"]
BASE_PATH = "/World/BalanceCar/base_link"
WHEEL_RADIUS_M = 0.034

# Per-feature normalisation scales (each obs is divided by its scale).
# Picked so each normalised feature sits roughly in [-1, +1] in healthy states.
OBS_SCALES = np.array(
    [
        0.6,    # pitch (rad). Fall threshold is 0.61 rad = 35 deg.
        5.0,    # pitch_rate (rad/s). ±5 covers normal balance dynamics.
        0.6,    # roll (rad)
        math.pi,  # yaw (rad)
        5.0,    # yaw_rate (rad/s)
        5.0,    # x_w (m) — course length is ~14 m, but most early-stage activity within ±5
        0.5,    # x_w_dot (m/s)
        0.2,    # target_velocity (m/s)
    ],
    dtype=np.float32,
)


@dataclass
class BalanceEnvNormConfig:
    asset_path: Path
    world_asset_path: Optional[Path] = None
    world_prim_path: str = "/World/CombinedCourse"
    use_default_ground: bool = True
    physics_hz: float = 240.0
    effort_limit_nm: float = 1.2
    max_episode_steps: int = 1200
    init_pitch_range_deg: tuple[float, float] = (-1.0, 1.0)
    init_x_range_m: tuple[float, float] = (-0.05, 0.05)
    target_velocity_range_m_s: tuple[float, float] = (-0.15, 0.15)
    fall_pitch_deg: float = 35.0
    fall_roll_deg: float = 35.0


class BalanceCarEnvNorm(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: BalanceEnvNormConfig):
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import UsdGeom

        self.cfg = cfg
        self._dt = 1.0 / cfg.physics_hz
        self._UsdGeom = UsdGeom

        self._world = World(stage_units_in_meters=1.0, physics_dt=self._dt, rendering_dt=self._dt)
        if cfg.use_default_ground:
            self._world.scene.add_default_ground_plane()
        if cfg.world_asset_path is not None:
            add_reference_to_stage(usd_path=str(cfg.world_asset_path.resolve()), prim_path=cfg.world_prim_path)
        add_reference_to_stage(usd_path=str(cfg.asset_path.resolve()), prim_path="/World/BalanceCar")

        stage = self._world.stage
        for joint_path in (
            "/World/BalanceCar/joints/left_wheel_joint",
            "/World/BalanceCar/joints/right_wheel_joint",
        ):
            joint = stage.GetPrimAtPath(joint_path)
            for attr_name in (
                "drive:angular:physics:stiffness",
                "drive:angular:physics:damping",
                "drive:angular:physics:maxForce",
            ):
                attr = joint.GetAttribute(attr_name)
                if attr:
                    attr.Set(0.0)

        self._robot = self._world.scene.add(
            Articulation(prim_paths_expr=BASE_PATH, name="balance_car_rl_norm")
        )
        self._world.reset()

        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._step_idx = 0
        self._target_velocity = 0.0
        self._prev_pitch = 0.0
        self._prev_yaw = 0.0
        self._prev_x_w = 0.0
        self._rng = np.random.default_rng()
        self._root_xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/BalanceCar"))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        init_pitch_deg = self._rng.uniform(*self.cfg.init_pitch_range_deg)
        init_x_m = self._rng.uniform(*self.cfg.init_x_range_m)
        self._target_velocity = self._rng.uniform(*self.cfg.target_velocity_range_m_s)

        xform_ops = self._root_xform.GetOrderedXformOps()
        translate_op = None
        rotate_y_op = None
        for op in xform_ops:
            if op.GetOpType() == self._UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif op.GetOpType() == self._UsdGeom.XformOp.TypeRotateY:
                rotate_y_op = op
        if translate_op is None:
            translate_op = self._root_xform.AddTranslateOp()
        if rotate_y_op is None:
            rotate_y_op = self._root_xform.AddRotateYOp()
        from pxr import Gf
        translate_op.Set(Gf.Vec3d(float(init_x_m), 0.0, 0.0))
        rotate_y_op.Set(float(init_pitch_deg))

        self._world.reset()
        self._step_idx = 0
        obs_raw = self._compute_raw_obs()
        self._prev_x_w = float(obs_raw[5])
        return obs_raw / OBS_SCALES, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        left_effort = float(action[0]) * self.cfg.effort_limit_nm
        right_effort = float(action[1]) * self.cfg.effort_limit_nm
        self._robot.set_joint_efforts(
            np.array([[left_effort, right_effort]], dtype=float), joint_names=WHEEL_JOINT_NAMES
        )
        self._world.step(render=False)
        self._step_idx += 1

        obs_raw = self._compute_raw_obs()
        obs = obs_raw / OBS_SCALES

        pitch_deg = math.degrees(obs_raw[0])
        roll_deg = math.degrees(obs_raw[2])
        x_w = float(obs_raw[5])
        x_w_dot = float(obs_raw[6])

        fell = abs(pitch_deg) > self.cfg.fall_pitch_deg or abs(roll_deg) > self.cfg.fall_roll_deg
        truncated = self._step_idx >= self.cfg.max_episode_steps
        terminated = fell

        # Simple, single-objective reward:
        # - Heavy reward for hitting target velocity exactly.
        # - Constant survival bonus to discourage falling.
        # - Tiny pitch/roll penalties.
        # - Big fall penalty.
        vel_err = x_w_dot - self._target_velocity
        vel_reward = 2.0 * math.exp(-(vel_err / 0.06) ** 2)
        pitch_pen = (pitch_deg / 35.0) ** 2
        roll_pen = 3.0 * (roll_deg / 35.0) ** 2
        survive = 1.0
        fall_pen = 100.0 if fell else 0.0
        reward = vel_reward - pitch_pen - roll_pen + survive - fall_pen

        self._prev_x_w = x_w
        info = {
            "x_w": x_w,
            "x_w_dot": x_w_dot,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "fell": fell,
            "vel_err": vel_err,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        return None

    def _body_angles(self) -> tuple[float, float, float]:
        from pxr import Usd
        matrix = self._UsdGeom.Xformable(
            self._world.stage.GetPrimAtPath(BASE_PATH)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rotation = matrix.ExtractRotationMatrix()
        local_x_world = rotation.GetRow(0)
        local_z_world = rotation.GetRow(2)
        pitch = math.atan2(-float(local_z_world[0]), float(local_z_world[2]))
        yaw = math.atan2(float(local_x_world[1]), float(local_x_world[0]))
        roll = math.atan2(float(local_z_world[1]), float(local_z_world[2]))
        return pitch, yaw, roll

    def _compute_raw_obs(self) -> np.ndarray:
        pitch, yaw, roll = self._body_angles()
        jp = np.asarray(
            self._robot.get_joint_positions(joint_names=WHEEL_JOINT_NAMES), dtype=float
        ).reshape(-1)
        jv = np.asarray(
            self._robot.get_joint_velocities(joint_names=WHEEL_JOINT_NAMES), dtype=float
        ).reshape(-1)
        x_w = 0.5 * float(jp[0] + jp[1]) * WHEEL_RADIUS_M
        x_w_dot = 0.5 * float(jv[0] + jv[1]) * WHEEL_RADIUS_M
        pitch_rate = (pitch - self._prev_pitch) / self._dt
        yaw_delta = math.atan2(math.sin(yaw - self._prev_yaw), math.cos(yaw - self._prev_yaw))
        yaw_rate = yaw_delta / self._dt
        self._prev_pitch = pitch
        self._prev_yaw = yaw
        return np.array(
            [pitch, pitch_rate, roll, yaw, yaw_rate, x_w, x_w_dot, self._target_velocity],
            dtype=np.float32,
        )
