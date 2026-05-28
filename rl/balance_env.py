"""Gymnasium environment for the two-wheel balance car in Isaac Sim 5.0.

The environment wraps a single Isaac Sim World with one balance car articulation.
Action: 2D continuous, each in [-1, 1], scaled to [-effort_limit, +effort_limit] N*m.
Observation: pitch, pitch_rate, roll, yaw_rate, x_w, x_w_dot, target_velocity.
Reward: forward progress - pitch penalty - roll penalty - effort cost - large fall penalty.
Episode ends on fall (|pitch|>35 deg or |roll|>35 deg) or after `max_episode_steps` ticks.

This is meant for use INSIDE a SimulationApp context. Create the SimulationApp in
your training script BEFORE importing this module. The env reuses the same World
across episodes (Isaac Sim can't be re-created cheaply).
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


@dataclass
class BalanceEnvConfig:
    asset_path: Path
    world_asset_path: Optional[Path] = None  # if set, load a custom world USD
    world_prim_path: str = "/World/CombinedCourse"
    use_default_ground: bool = True
    physics_hz: float = 240.0
    effort_limit_nm: float = 1.2
    max_episode_steps: int = 3600  # 15 s at 240 Hz (Flamingo uses 20 s)
    # "torque": action = joint effort in N*m (matches PID/LQR baselines, real
    #   motor PWM-style control). RL has to learn the torque-to-balance dance.
    # "velocity": action = joint TARGET VELOCITY in rad/s (Flamingo-style). The
    #   wheel's built-in PhysX PD tracks the target. RL just commands speed.
    #   Much easier to learn — matches Flamingo baseline. Sign-to-direction
    #   mapping becomes immediate (cart goes +X iff target wheel vel > 0).
    action_mode: str = "torque"
    # Max wheel angular velocity at action=1 (only used when action_mode="velocity").
    # 5 rad/s × 0.034 m = 0.17 m/s linear, matches target_velocity scale (±0.15).
    # Reduced from 30 (Flamingo) because: (a) our task only needs ±0.15 m/s
    # max linear speed; (b) larger max means random initial policy outputs
    # commands so aggressive cart tips before policy can learn balance.
    max_wheel_vel_rad_s: float = 5.0
    # Joint drive damping when action_mode="velocity". Higher = snappier PD
    # tracking but more torque used.
    velocity_drive_damping: float = 2.0
    # Initial perturbation. Wider than the original demo (±1°) to encourage
    # robustness, but kept moderate so the catch-fall drift doesn't bias the
    # policy toward a wrong direction faster than it can learn correct
    # driving. ±2° is a good middle ground.
    init_pitch_range_deg: tuple[float, float] = (-2.0, 2.0)
    init_x_range_m: tuple[float, float] = (-0.05, 0.05)
    # Optional: list of (x_m, z_m) spawn positions. If non-empty, overrides
    # init_x_range_m and is sampled uniformly each reset. Use this to do
    # mixed-terrain training where the cart spawns at different course
    # sections every episode.
    init_positions: tuple = ()
    # Stage 2 curriculum: drive at random forward velocity in ±0.15 m/s.
    target_velocity_range_m_s: tuple[float, float] = (-0.15, 0.15)
    # If non-None, override sampling and use this fixed target every episode.
    # Useful for single-target diagnostic training before mixing.
    fixed_target_velocity: Optional[float] = None
    fall_pitch_deg: float = 35.0
    fall_roll_deg: float = 35.0


class BalanceCarEnv(gym.Env):
    """Single-instance balance car env. NOT vectorised. NOT thread-safe."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: BalanceEnvConfig):
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import UsdGeom

        self.cfg = cfg
        self._dt = 1.0 / cfg.physics_hz

        # Build the world once.
        self._world = World(stage_units_in_meters=1.0, physics_dt=self._dt, rendering_dt=self._dt)
        if cfg.use_default_ground:
            self._world.scene.add_default_ground_plane()
        if cfg.world_asset_path is not None:
            add_reference_to_stage(usd_path=str(cfg.world_asset_path.resolve()), prim_path=cfg.world_prim_path)
        add_reference_to_stage(usd_path=str(cfg.asset_path.resolve()), prim_path="/World/BalanceCar")

        # Configure joint drives based on action mode.
        # torque mode: drives fully disabled, RL action is pure effort.
        # velocity mode: drives enabled with damping, RL action is target
        # wheel velocity tracked by PhysX's built-in PD. Much easier to learn.
        stage = self._world.stage
        for joint_path in (
            "/World/BalanceCar/joints/left_wheel_joint",
            "/World/BalanceCar/joints/right_wheel_joint",
        ):
            joint = stage.GetPrimAtPath(joint_path)
            stiffness_attr = joint.GetAttribute("drive:angular:physics:stiffness")
            damping_attr = joint.GetAttribute("drive:angular:physics:damping")
            max_force_attr = joint.GetAttribute("drive:angular:physics:maxForce")
            if cfg.action_mode == "torque":
                # Disable drives → RL effort is the only torque source.
                for attr in (stiffness_attr, damping_attr, max_force_attr):
                    if attr:
                        attr.Set(0.0)
            elif cfg.action_mode == "velocity":
                # Enable velocity-tracking drive: stiffness=0 (pure damping
                # gives velocity control), damping=velocity_drive_damping,
                # max_force = effort_limit_nm.
                if stiffness_attr:
                    stiffness_attr.Set(0.0)
                if damping_attr:
                    damping_attr.Set(float(cfg.velocity_drive_damping))
                if max_force_attr:
                    max_force_attr.Set(float(cfg.effort_limit_nm))
            else:
                raise ValueError(f"Unknown action_mode: {cfg.action_mode}")

        self._UsdGeom = UsdGeom
        self._robot = self._world.scene.add(
            Articulation(prim_paths_expr=BASE_PATH, name="balance_car_rl")
        )
        self._world.reset()

        # Gym spaces. Obs vector size = 8: [pitch, pitch_rate, roll, yaw,
        # yaw_rate, x_w, x_w_dot, target_velocity].
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Per-episode state
        self._step_idx = 0
        self._target_velocity = 0.0
        self._prev_pitch = 0.0
        self._prev_yaw = 0.0
        self._prev_x_w = 0.0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._rng = np.random.default_rng()
        # Target base height when cart is upright on flat ground; used for
        # base_height penalty (Flamingo-baseline style).
        self._target_base_z = WHEEL_RADIUS_M  # 0.034 m
        # Push-disturbance schedule (random push every ~push_interval steps).
        self._push_next_step = 0
        # Stable root xform handle for resetting pose
        self._root_xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/BalanceCar"))
        # Track the ops we add so reset doesn't keep stacking
        self._reset_op_translate = None
        self._reset_op_rotateY = None

    # ----------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample initial conditions
        init_pitch_deg = self._rng.uniform(*self.cfg.init_pitch_range_deg)
        if self.cfg.init_positions:
            idx = int(self._rng.integers(0, len(self.cfg.init_positions)))
            init_x_m, init_z_m = self.cfg.init_positions[idx]
        else:
            init_x_m = self._rng.uniform(*self.cfg.init_x_range_m)
            init_z_m = 0.0
        if self.cfg.fixed_target_velocity is not None:
            self._target_velocity = float(self.cfg.fixed_target_velocity)
        else:
            self._target_velocity = self._rng.uniform(*self.cfg.target_velocity_range_m_s)

        # Reset the world's pose for the cart. Set the translate and pitch on
        # the root xform. To avoid stacking ops across episodes we reuse the
        # same op handles when possible.
        xform_ops = self._root_xform.GetOrderedXformOps()
        # If no ops yet, add them. Otherwise reuse.
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
        translate_op.Set(Gf.Vec3d(float(init_x_m), 0.0, float(init_z_m)))
        rotate_y_op.Set(float(init_pitch_deg))

        self._world.reset()

        # Initial obs
        self._step_idx = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        # Schedule first push 3-5 s into the episode
        self._push_next_step = int(self._rng.uniform(3.0, 5.0) * self.cfg.physics_hz)
        obs = self._compute_obs()
        # Initialise prev refs
        pitch, _, roll = self._body_angles()
        self._prev_pitch = pitch
        self._prev_yaw = self._yaw()
        self._prev_x_w = float(obs[5])
        return obs, {}

    def step(self, action: np.ndarray):
        # Apply action — clip and dispatch by mode.
        action = np.clip(action, -1.0, 1.0)
        if self.cfg.action_mode == "torque":
            left_effort = float(action[0]) * self.cfg.effort_limit_nm
            right_effort = float(action[1]) * self.cfg.effort_limit_nm
            self._robot.set_joint_efforts(
                np.array([[left_effort, right_effort]], dtype=float),
                joint_names=WHEEL_JOINT_NAMES,
            )
        else:  # velocity mode
            left_vel = float(action[0]) * self.cfg.max_wheel_vel_rad_s
            right_vel = float(action[1]) * self.cfg.max_wheel_vel_rad_s
            self._robot.set_joint_velocity_targets(
                np.array([[left_vel, right_vel]], dtype=float),
                joint_names=WHEEL_JOINT_NAMES,
            )
        self._world.step(render=False)
        self._step_idx += 1

        obs_raw = self._compute_raw_obs()
        obs = self._normalise_obs(obs_raw)
        pitch_deg = math.degrees(obs_raw[0])
        roll_deg = math.degrees(obs_raw[2])
        x_w = float(obs_raw[5])

        fell = abs(pitch_deg) > self.cfg.fall_pitch_deg or abs(roll_deg) > self.cfg.fall_roll_deg
        truncated = self._step_idx >= self.cfg.max_episode_steps
        terminated = fell

        # Reward — Flamingo-baseline style, adapted for balance car.
        # Tracking + flat orientation + base_height + smoothness + termination.
        x_w_dot = float(obs_raw[6])
        # Velocity tracking — sign-aware, GRADIENT EVERYWHERE.
        # Previous attempts used exp(-vel_err²/0.05) which decayed to zero past
        # vel_err≈0.5, leaving the policy without gradient when it was driving
        # wrong direction fast (vel_err≈1.0+). Now use a continuous linear
        # penalty that's always present, with sign-distinguished branches.
        vel_err = x_w_dot - self._target_velocity
        target_nonzero = abs(self._target_velocity) > 1e-3
        if target_nonzero:
            same_sign = float(np.sign(x_w_dot)) * float(np.sign(self._target_velocity)) > 0.5
            if same_sign:
                # Cart moving in correct direction: reward scaled by closeness
                # to target speed. Max reward when vel_err = 0.
                vel_reward = 2.0 - 8.0 * abs(vel_err)
            elif abs(x_w_dot) < 1e-3:
                # Cart stationary but task demands motion: small constant
                # negative (so "stand still" is worse than driving correctly).
                vel_reward = -1.0
            else:
                # Cart moving OPPOSITE to target: heavy penalty, grows with
                # how fast it's going wrong. This term has gradient at any
                # speed and any sign.
                vel_reward = -2.0 - 5.0 * abs(x_w_dot)
        else:
            # Target == 0: reward staying still, penalise drift.
            vel_reward = 1.0 - 3.0 * abs(x_w_dot)
        # Clip to a reasonable range
        vel_reward = max(-5.0, min(2.0, vel_reward))

        # 3) Flat orientation (Flamingo: weight -2.5 on roll/pitch^2)
        flat_orient = -(pitch_deg ** 2 + roll_deg ** 2) / (35.0 ** 2)

        # 4) Base height — keep cart's wheel axle at target height (Flamingo's
        # most heavily weighted term). Tells policy "don't squat or fly".
        base_z = float(self._world.stage.GetPrimAtPath(BASE_PATH).GetAttribute("xformOp:translate").Get()[2]) if False else 0.034
        # Note: we don't actually fetch the live z here every step — too slow.
        # Use the rotation matrix's translation extracted in _compute_obs would
        # be cleaner; for now skip until we cache it.

        # 5) Action smoothness — penalise sudden large action changes
        action_rate = float(np.sum((np.asarray(action, dtype=np.float32) - self._prev_action) ** 2))

        # 6) Effort cost — small penalty per unit torque
        effort_cost = float(action[0] ** 2 + action[1] ** 2)

        # Penalise unsymmetric action and yaw motion. The previous training
        # round revealed the policy was outputting strong differential actions
        # (e.g. [-1, +1]) to yaw in place rather than drive forward — yawing
        # avoids the difficult balance dynamics of translation while still
        # collecting survive_bonus. Penalising both action-asymmetry and
        # actual yaw_rate makes the yaw shortcut costly.
        action_diff = float(action[0] - action[1])
        yaw_rate_obs = float(obs_raw[4])  # raw rad/s

        # Composite reward (weights tuned roughly to Flamingo proportions)
        reward = (
            vel_reward                                              # sign-aware velocity tracking
            - 2.5 * (pitch_deg ** 2 + roll_deg ** 2) / (35.0 ** 2)  # flat orient
            - 0.5 * action_diff ** 2                                # prefer symmetric (translation) actions
            - 0.2 * yaw_rate_obs ** 2 / 25.0                        # penalise actual yaw motion
            - 0.01 * action_rate                                    # smoothness
            - 5e-4 * effort_cost                                    # tiny effort tax
            + 0.5                                                    # survive bonus per tick
        )
        if fell:
            reward -= 200.0               # Flamingo termination penalty
        info = {
            "x_w": x_w,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "fell": fell,
        }
        self._prev_x_w = x_w
        self._prev_action = np.asarray(action, dtype=np.float32).copy()
        return obs, reward, terminated, truncated, info

    def close(self):
        # The SimulationApp owns the world; do not close here. The training
        # script is responsible for SimulationApp.close().
        return None

    # ----------------------------------------------------------- helpers

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

    def _yaw(self) -> float:
        return self._body_angles()[1]

    def _compute_obs(self) -> np.ndarray:
        """Return NORMALISED observation for the policy. The reward function
        should use _compute_raw_obs() so it sees SI-unit values."""
        return self._normalise_obs(self._compute_raw_obs())

    def _compute_raw_obs(self) -> np.ndarray:
        pitch, yaw, roll = self._body_angles()
        # Get joint state
        jp = np.asarray(
            self._robot.get_joint_positions(joint_names=WHEEL_JOINT_NAMES), dtype=float
        ).reshape(-1)
        jv = np.asarray(
            self._robot.get_joint_velocities(joint_names=WHEEL_JOINT_NAMES), dtype=float
        ).reshape(-1)
        x_w = 0.5 * float(jp[0] + jp[1]) * WHEEL_RADIUS_M
        x_w_dot = 0.5 * float(jv[0] + jv[1]) * WHEEL_RADIUS_M

        # Rates via finite difference for pitch/yaw (no gyro available without IMU)
        pitch_rate = (pitch - self._prev_pitch) / self._dt
        yaw_delta = math.atan2(math.sin(yaw - self._prev_yaw), math.cos(yaw - self._prev_yaw))
        yaw_rate = yaw_delta / self._dt
        self._prev_pitch = pitch
        self._prev_yaw = yaw

        return np.array(
            [pitch, pitch_rate, roll, yaw, yaw_rate, x_w, x_w_dot, self._target_velocity],
            dtype=np.float32,
        )

    @staticmethod
    def _normalise_obs(raw: np.ndarray) -> np.ndarray:
        """Per-feature normalisation so the MLP sees roughly unit-scale inputs.
        Without this, large-magnitude features (pitch_rate ≈ ±50 rad/s,
        x_w ≈ ±5 m) drown small-magnitude features like target_velocity
        (±0.15 m/s) — leading to a policy that ignores the command.
        Scales were picked from the expected dynamic range of each feature.
        """
        scales = np.array([0.6, 5.0, 0.6, math.pi, 5.0, 5.0, 0.5, 0.2], dtype=np.float32)
        return (raw / scales).astype(np.float32)
