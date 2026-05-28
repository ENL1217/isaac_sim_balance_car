"""Direct-workflow Isaac Lab env for the two-wheel balance car.

Single articulation per env (chassis + 2 wheels). Action = torque on each
wheel. Reward — Flamingo-baseline style: linear velocity tracking +
flat-orientation penalty + termination penalty.

Parallel-env training target: 4096 envs.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from .terrain_config import BALANCE_TERRAINS_CFG

# Absolute path to the car USD asset in this repo. Lab task discovery imports
# this module from outside the repo (via PYTHONPATH); resolve relative to
# THIS file's location.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAR_USD = _REPO_ROOT / "sim" / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda"

WHEEL_RADIUS_M = 0.034
WHEEL_TRACK_M = 0.168


BALANCE_CAR_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_CAR_USD).replace("\\", "/"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Spawn slightly above terrain so cart drops + settles on whatever
        # sub-terrain it's on (varies in height across the procedural grid).
        pos=(0.0, 0.0, 0.05),
        joint_pos={"left_wheel_joint": 0.0, "right_wheel_joint": 0.0},
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=1.2,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


@configclass
class BalanceCarEnvCfg(DirectRLEnvCfg):
    decimation = 2  # 240 Hz physics, 120 Hz control
    episode_length_s = 15.0
    action_scale = 1.2  # N*m per wheel at action=1
    action_space = 2
    observation_space = 8
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    robot_cfg: ArticulationCfg = BALANCE_CAR_CFG

    # Procedural rough terrain (slopes / stairs / boxes / random rough),
    # scaled for our small wheels. Each env is placed on a different
    # sub-terrain so the policy sees varied geometry during training.
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=BALANCE_TERRAINS_CFG,
        max_init_terrain_level=4,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # Reset / termination
    max_pitch_rad = math.radians(35.0)
    max_roll_rad = math.radians(35.0)
    init_pitch_range_rad = (-math.radians(2.0), math.radians(2.0))

    # Command sampling
    target_vel_x_range = (-0.15, 0.15)  # m/s
    target_vel_yaw_range = (0.0, 0.0)  # rad/s (yaw command); start with 0

    # Reward scales (Flamingo-baseline style)
    # Boosted to make moving clearly more rewarding than standing still on
    # procedural terrain (where the conservative policy converged to perfect
    # stillness).
    rew_track_lin_vel = 5.0
    rew_dir_match = 2.0
    rew_flat_orient = -2.0
    rew_action_rate = -0.01
    rew_alive = 0.1
    rew_terminated = -200.0
    rew_no_move_penalty = -1.0  # explicit penalty when target != 0 and cart still
    tracking_std = 0.08

    # Sensor model
    # When True, the policy observation has only IMU + target (6 dims).
    # When False, observations also include wheel-position-derived x_w and
    # wheel-velocity-derived x_w_dot (8 dims, default).
    no_encoder: bool = False


@configclass
class BalanceCarNoEncoderEnvCfg(BalanceCarEnvCfg):
    """Variant with no wheel encoder feedback.

    The cart's PhysX dynamics are unchanged — only the OBSERVATION exposed to
    the policy changes. Expect significantly worse velocity tracking and
    stationary stability because the policy can no longer observe its own
    position/velocity error.
    """
    observation_space = 6  # drop x_w and x_w_dot from the 8-dim default
    no_encoder = True


class BalanceCarEnv(DirectRLEnv):
    cfg: BalanceCarEnvCfg

    def __init__(self, cfg: BalanceCarEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._left_joint_idx, _ = self.robot.find_joints("left_wheel_joint")
        self._right_joint_idx, _ = self.robot.find_joints("right_wheel_joint")
        self._action_scale = self.cfg.action_scale
        self._prev_action = torch.zeros(self.num_envs, 2, device=self.device)
        # Per-env target velocity (resampled on reset)
        self._target_vel_x = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        # Terrain is configured via cfg.terrain (TerrainImporterCfg) and
        # auto-spawned by DirectRLEnv before _setup_scene runs, so we don't
        # need spawn_ground_plane here.
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions shape (num_envs, 2) in [-1, 1]; scale to torque.
        self._actions_scaled = actions.clamp(-1.0, 1.0) * self._action_scale
        self._raw_actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # left / right wheel torques
        left = self._actions_scaled[:, 0:1]
        right = self._actions_scaled[:, 1:2]
        self.robot.set_joint_effort_target(left, joint_ids=self._left_joint_idx)
        self.robot.set_joint_effort_target(right, joint_ids=self._right_joint_idx)

    def _get_observations(self) -> dict:
        # Body frame quantities for pitch / roll / yaw rate from articulation
        root_quat = self.robot.data.root_quat_w  # (N, 4) wxyz
        ang_vel_b = self.robot.data.root_ang_vel_b  # (N, 3) body frame
        lin_vel_b = self.robot.data.root_lin_vel_b  # (N, 3) body frame
        # Extract pitch / roll from quaternion
        w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        # roll = atan2(2*(w*x + y*z), 1 - 2*(x^2 + y^2))
        # pitch = asin(2*(w*y - z*x))
        sinp = 2.0 * (w * y - z * x)
        sinp = sinp.clamp(-1.0, 1.0)
        pitch = torch.asin(sinp)
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))

        # Wheel state
        joint_vel = self.robot.data.joint_vel  # (N, num_joints)
        left_wheel_vel = joint_vel[:, self._left_joint_idx[0]]
        right_wheel_vel = joint_vel[:, self._right_joint_idx[0]]
        wheel_vel_avg = 0.5 * (left_wheel_vel + right_wheel_vel)
        x_w_dot = wheel_vel_avg * WHEEL_RADIUS_M  # linear velocity in m/s

        # Root x position (in env-local frame is more useful; subtract env_origins x)
        root_pos = self.robot.data.root_pos_w  # (N, 3)
        env_origins = self.scene.env_origins  # (N, 3)
        x_w = root_pos[:, 0] - env_origins[:, 0]

        if self.cfg.no_encoder:
            # IMU-only observation: pitch / roll + 3 gyro axes + target.
            # The cart's internal dynamics still depend on x_w and x_w_dot,
            # but the policy doesn't see them. Equivalent to running the
            # classical controller with --no-encoder.
            obs = torch.stack(
                [
                    pitch,                      # rad
                    ang_vel_b[:, 1],            # body pitch rate (gyro_y)
                    roll,                       # rad
                    ang_vel_b[:, 2],            # yaw rate (gyro_z)
                    ang_vel_b[:, 0],            # roll rate (gyro_x)
                    self._target_vel_x,         # m/s target (still commanded)
                ],
                dim=-1,
            )
        else:
            obs = torch.stack(
                [
                    pitch,                      # rad
                    ang_vel_b[:, 1],            # body pitch rate
                    roll,                       # rad
                    ang_vel_b[:, 2],            # yaw rate
                    ang_vel_b[:, 0],            # roll rate
                    x_w,                        # m, local — encoder-derived
                    x_w_dot,                    # m/s — encoder-derived
                    self._target_vel_x,         # m/s target
                ],
                dim=-1,
            )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Linear velocity tracking
        ang_vel_b = self.robot.data.root_ang_vel_b
        lin_vel_b = self.robot.data.root_lin_vel_b
        root_quat = self.robot.data.root_quat_w
        w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        sinp = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
        pitch = torch.asin(sinp)
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))

        # x velocity in body frame
        vel_x = lin_vel_b[:, 0]
        vel_err = vel_x - self._target_vel_x
        track_lin_vel = torch.exp(-(vel_err ** 2) / (self.cfg.tracking_std ** 2))

        # Direction match bonus — only credit when cart moves in target direction.
        # When target ≈ 0 give bonus iff cart is also still. When target != 0
        # require sign match. This prevents the "stand still" local optimum.
        target_nonzero = self._target_vel_x.abs() > 1e-3
        cart_moving = vel_x.abs() > 1e-3
        sign_match = torch.where(
            target_nonzero & cart_moving,
            torch.sign(vel_x) * torch.sign(self._target_vel_x),
            torch.where(
                ~target_nonzero & ~cart_moving,
                torch.ones_like(vel_x),  # target=0 and cart still: full bonus
                torch.zeros_like(vel_x),  # target=0 but cart moving, OR target!=0 cart=0: 0
            ),
        )

        flat_orient = -(pitch ** 2 + roll ** 2)

        action_rate = torch.sum((self._raw_actions - self._prev_action) ** 2, dim=-1)

        # Explicit penalty for standing still when a non-zero velocity is
        # commanded. Without this the policy collapses to "balance perfectly
        # and ignore command" on procedural terrain.
        no_move = (self._target_vel_x.abs() > 1e-3) & (vel_x.abs() < 0.02)
        rew = (
            self.cfg.rew_track_lin_vel * track_lin_vel
            + self.cfg.rew_dir_match * sign_match
            + self.cfg.rew_flat_orient * flat_orient
            + self.cfg.rew_action_rate * action_rate
            + self.cfg.rew_alive * (1.0 - self.reset_terminated.float())
            + self.cfg.rew_terminated * self.reset_terminated.float()
            + self.cfg.rew_no_move_penalty * no_move.float()
        )
        self._prev_action = self._raw_actions.clone()
        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract pitch / roll again here (Direct workflow recomputes per step)
        root_quat = self.robot.data.root_quat_w
        w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        sinp = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
        pitch = torch.asin(sinp).abs()
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)).abs()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fell = (pitch > self.cfg.max_pitch_rad) | (roll > self.cfg.max_roll_rad)
        return fell, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Resample target velocity
        self._target_vel_x[env_ids] = sample_uniform(
            self.cfg.target_vel_x_range[0],
            self.cfg.target_vel_x_range[1],
            (len(env_ids),),
            self.device,
        )

        # Reset robot pose with random initial pitch
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Apply small random pitch perturbation via quaternion (axis-angle around Y)
        pitch_perturbation = sample_uniform(
            self.cfg.init_pitch_range_rad[0],
            self.cfg.init_pitch_range_rad[1],
            (len(env_ids),),
            self.device,
        )
        half = pitch_perturbation * 0.5
        # quat (w, x, y, z) = (cos(half), 0, sin(half), 0)
        cos_h = torch.cos(half)
        sin_h = torch.sin(half)
        default_root_state[:, 3] = cos_h
        default_root_state[:, 4] = 0.0
        default_root_state[:, 5] = sin_h
        default_root_state[:, 6] = 0.0

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._prev_action[env_ids] = 0.0
