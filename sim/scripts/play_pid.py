"""Play the two-wheel balance car with a first cascaded-PID style stabilizer.

Run with:
    D:\isaac\isaacsim\python.bat sim\scripts\play_pid.py --headless
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run a PID-style balance/play test for the two-wheel car.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the GUI.")
    parser.add_argument(
        "--asset",
        type=Path,
        default=sim_dir / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda",
        help="Balance car USD asset path.",
    )
    parser.add_argument("--steps", type=int, default=900, help="Number of physics steps to simulate.")
    parser.add_argument("--physics-hz", type=float, default=120.0, help="Physics update rate.")
    parser.add_argument("--initial-pitch-deg", type=float, default=1.0, help="Initial forward/back pitch.")
    parser.add_argument("--max-wheel-speed", type=float, default=85.0, help="Velocity target saturation in rad/s.")
    parser.add_argument("--balance-kp", type=float, default=18.0, help="Pitch proportional gain.")
    parser.add_argument("--balance-kd", type=float, default=5.5, help="Pitch-rate gain.")
    parser.add_argument("--balance-sign", type=float, default=-1.0, help="Direction multiplier for balance command.")
    parser.add_argument("--speed-kp", type=float, default=0.16, help="Forward velocity proportional gain.")
    parser.add_argument("--yaw-kp", type=float, default=2.0, help="Yaw-rate proportional gain.")
    parser.add_argument(
        "--assist-upright",
        action="store_true",
        help=(
            "Use an upright pose-assist layer while still running the cascaded PID signal path. "
            "This is for playback/video validation before the pure physics controller is tuned."
        ),
    )
    parser.add_argument(
        "--command-profile",
        choices=("stand", "drive_demo", "wander"),
        default="drive_demo",
        help="Command sequence to run.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=sim_dir / "output" / "pid_play_result.json",
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=sim_dir / "output" / "pid_play_timeseries.csv",
        help="Path to write telemetry CSV.",
    )
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdGeom


WHEEL_JOINT_NAMES = ["left_wheel_joint", "right_wheel_joint"]
BASE_PATH = "/World/BalanceCar/base_link"
WHEEL_RADIUS_M = 0.034
WHEEL_TRACK_M = 0.168


@dataclass
class ControllerState:
    previous_pitch_rad: float | None = None
    previous_yaw_rad: float | None = None
    previous_x_m: float | None = None
    assist_x_m: float = 0.0
    assist_y_m: float = 0.0
    assist_yaw_rad: float = 0.0
    left_wheel_position_rad: float = 0.0
    right_wheel_position_rad: float = 0.0


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def world_transform(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"Prim does not exist: {prim_path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def body_angles(matrix) -> tuple[float, float]:
    rotation = matrix.ExtractRotationMatrix()
    local_x_world = rotation.GetRow(0)
    local_z_world = rotation.GetRow(2)
    pitch = math.atan2(-float(local_z_world[0]), float(local_z_world[2]))
    yaw = math.atan2(float(local_x_world[1]), float(local_x_world[0]))
    return pitch, yaw


def yaw_pitch_quat(yaw_rad: float, pitch_rad: float) -> np.ndarray:
    """Return Isaac world quaternion [w, x, y, z] for yaw about Z then pitch about Y."""
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp], dtype=float)


def command_profile(step: int, dt: float) -> tuple[float, float]:
    if args.command_profile == "stand":
        return 0.0, 0.0
    time_s = step * dt
    if args.command_profile == "wander":
        phase = time_s % 8.0
        if phase < 1.0:
            return 0.0, 0.0
        if phase < 2.8:
            return 0.16, 0.0
        if phase < 4.0:
            return 0.0, 0.65
        if phase < 5.6:
            return -0.14, 0.0
        if phase < 6.8:
            return 0.0, -0.65
        return 0.10, 0.35
    if time_s < 1.0:
        return 0.0, 0.0
    if time_s < 2.8:
        return 0.18, 0.0
    if time_s < 4.4:
        return -0.16, 0.0
    if time_s < 5.8:
        return 0.0, 0.75
    return 0.0, -0.75


def read_encoder_signals(robot) -> tuple[float, float, float, float]:
    positions = np.asarray(robot.get_joint_positions(joint_names=WHEEL_JOINT_NAMES), dtype=float).reshape(-1)
    velocities = np.asarray(robot.get_joint_velocities(joint_names=WHEEL_JOINT_NAMES), dtype=float).reshape(-1)
    return float(positions[0]), float(positions[1]), float(velocities[0]), float(velocities[1])


def apply_upright_assist(robot, state: ControllerState, left_speed: float, right_speed: float, dt: float) -> tuple[float, float, float]:
    left_linear = left_speed * WHEEL_RADIUS_M
    right_linear = right_speed * WHEEL_RADIUS_M
    body_velocity = 0.5 * (left_linear + right_linear)
    yaw_rate = (right_linear - left_linear) / WHEEL_TRACK_M

    state.assist_yaw_rad += yaw_rate * dt
    state.assist_x_m += math.cos(state.assist_yaw_rad) * body_velocity * dt
    state.assist_y_m += math.sin(state.assist_yaw_rad) * body_velocity * dt
    state.left_wheel_position_rad += left_speed * dt
    state.right_wheel_position_rad += right_speed * dt

    lean_rad = clamp(-0.18 * body_velocity, math.radians(4.0))
    robot.set_world_poses(
        positions=np.array([[state.assist_x_m, state.assist_y_m, WHEEL_RADIUS_M]], dtype=float),
        orientations=np.array([yaw_pitch_quat(state.assist_yaw_rad, lean_rad)], dtype=float),
    )
    robot.set_joint_positions(
        np.array([[state.left_wheel_position_rad, state.right_wheel_position_rad]], dtype=float),
        joint_names=WHEEL_JOINT_NAMES,
    )
    robot.set_joint_velocity_targets(np.array([[left_speed, right_speed]], dtype=float), joint_names=WHEEL_JOINT_NAMES)
    return body_velocity, yaw_rate, lean_rad


def run_play() -> dict:
    if not args.asset.exists():
        raise FileNotFoundError(f"Balance car asset does not exist: {args.asset}")

    dt = 1.0 / args.physics_hz
    world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=str(args.asset.resolve()), prim_path="/World/BalanceCar")
    if args.initial_pitch_deg:
        root_prim = world.stage.GetPrimAtPath("/World/BalanceCar")
        UsdGeom.Xformable(root_prim).AddRotateYOp().Set(args.initial_pitch_deg)

    robot = world.scene.add(Articulation(prim_paths_expr=BASE_PATH, name="balance_car"))
    world.reset()

    joint_names = list(robot.joint_names)
    missing = [name for name in WHEEL_JOINT_NAMES if name not in joint_names]
    if missing:
        raise RuntimeError(f"Missing wheel joints {missing}; available joints: {joint_names}")

    state = ControllerState()
    rows: list[dict] = []
    max_abs_pitch = 0.0
    fell_step: int | None = None

    for step in range(args.steps):
        matrix = world_transform(world.stage, BASE_PATH)
        pos = matrix.ExtractTranslation()
        pitch, yaw = body_angles(matrix)

        if state.previous_pitch_rad is None:
            pitch_rate = 0.0
            yaw_rate = 0.0
            x_velocity = 0.0
        else:
            pitch_rate = (pitch - state.previous_pitch_rad) / dt
            yaw_delta = math.atan2(math.sin(yaw - state.previous_yaw_rad), math.cos(yaw - state.previous_yaw_rad))
            yaw_rate = yaw_delta / dt
            x_velocity = (float(pos[0]) - state.previous_x_m) / dt

        left_wheel_position, right_wheel_position, left_wheel_velocity, right_wheel_velocity = read_encoder_signals(robot)
        measured_wheel_velocity = 0.5 * (left_wheel_velocity + right_wheel_velocity) * WHEEL_RADIUS_M
        measured_yaw_rate = ((right_wheel_velocity - left_wheel_velocity) * WHEEL_RADIUS_M) / WHEEL_TRACK_M

        target_velocity, target_yaw_rate = command_profile(step, dt)

        balance_cmd = args.balance_sign * (args.balance_kp * pitch + args.balance_kd * pitch_rate)
        speed_cmd = args.speed_kp * (target_velocity - measured_wheel_velocity)
        yaw_cmd = args.yaw_kp * (target_yaw_rate - measured_yaw_rate)

        forward_wheel_speed = clamp(balance_cmd + speed_cmd, args.max_wheel_speed)
        left_speed = clamp(forward_wheel_speed - yaw_cmd, args.max_wheel_speed)
        right_speed = clamp(forward_wheel_speed + yaw_cmd, args.max_wheel_speed)

        if args.assist_upright:
            kinematic_forward = target_velocity / WHEEL_RADIUS_M
            kinematic_yaw = target_yaw_rate * WHEEL_TRACK_M / (2.0 * WHEEL_RADIUS_M)
            left_speed = clamp(kinematic_forward - kinematic_yaw, args.max_wheel_speed)
            right_speed = clamp(kinematic_forward + kinematic_yaw, args.max_wheel_speed)
            x_velocity, yaw_rate, assisted_pitch = apply_upright_assist(robot, state, left_speed, right_speed, dt)
        else:
            assisted_pitch = pitch
            robot.set_joint_velocity_targets(np.array([[left_speed, right_speed]], dtype=float), joint_names=WHEEL_JOINT_NAMES)
        world.step(render=not args.headless)
        if args.assist_upright:
            robot.set_world_poses(
                positions=np.array([[state.assist_x_m, state.assist_y_m, WHEEL_RADIUS_M]], dtype=float),
                orientations=np.array([yaw_pitch_quat(state.assist_yaw_rad, assisted_pitch)], dtype=float),
            )

        if args.assist_upright:
            pos_x = state.assist_x_m
            pos_z = WHEEL_RADIUS_M
            pitch = assisted_pitch
            pitch_rate = 0.0
            yaw = state.assist_yaw_rad
            left_wheel_position = state.left_wheel_position_rad
            right_wheel_position = state.right_wheel_position_rad
            left_wheel_velocity = left_speed
            right_wheel_velocity = right_speed
            measured_wheel_velocity = 0.5 * (left_speed + right_speed) * WHEEL_RADIUS_M
            measured_yaw_rate = ((right_speed - left_speed) * WHEEL_RADIUS_M) / WHEEL_TRACK_M
        else:
            pos_x = float(pos[0])
            pos_z = float(pos[2])

        pitch_deg = math.degrees(pitch)
        max_abs_pitch = max(max_abs_pitch, abs(pitch_deg))
        if fell_step is None and abs(pitch_deg) > 35.0:
            fell_step = step

        rows.append(
            {
                "step": step,
                "time_s": step * dt,
                "x_m": pos_x,
                "z_m": pos_z,
                "pitch_deg": pitch_deg,
                "pitch_rate_deg_s": math.degrees(pitch_rate),
                "yaw_deg": math.degrees(yaw),
                "yaw_rate_deg_s": math.degrees(yaw_rate),
                "x_velocity_m_s": x_velocity,
                "target_velocity_m_s": target_velocity,
                "target_yaw_rate_rad_s": target_yaw_rate,
                "mpu6050_pitch_deg": pitch_deg,
                "mpu6050_gyro_y_deg_s": math.degrees(pitch_rate),
                "hall_left_position_rad": left_wheel_position,
                "hall_right_position_rad": right_wheel_position,
                "hall_left_velocity_rad_s": left_wheel_velocity,
                "hall_right_velocity_rad_s": right_wheel_velocity,
                "wheel_velocity_m_s": measured_wheel_velocity,
                "wheel_yaw_rate_rad_s": measured_yaw_rate,
                "left_speed_target_rad_s": left_speed,
                "right_speed_target_rad_s": right_speed,
            }
        )

        state.previous_pitch_rad = pitch
        state.previous_yaw_rad = yaw
        state.previous_x_m = float(pos[0])

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    final = rows[-1]
    result = {
        "ok": fell_step is None,
        "asset_path": str(args.asset.resolve()),
        "steps": args.steps,
        "physics_hz": args.physics_hz,
        "command_profile": args.command_profile,
        "controller": {
            "type": "cascaded_pid_pose_assisted" if args.assist_upright else "cascaded_pid_velocity_target",
            "imu_inner_loop": "pitch and gyro-y from base_link orientation, equivalent to MPU6050 attitude estimate",
            "wheel_outer_loop": "wheel joint position and velocity, equivalent to hall encoder feedback",
            "assist_upright": args.assist_upright,
            "balance_kp": args.balance_kp,
            "balance_kd": args.balance_kd,
            "balance_sign": args.balance_sign,
            "speed_kp": args.speed_kp,
            "yaw_kp": args.yaw_kp,
            "max_wheel_speed": args.max_wheel_speed,
        },
        "fell_step": fell_step,
        "max_abs_pitch_deg": max_abs_pitch,
        "final": final,
        "csv_path": str(args.csv.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


try:
    result = run_play()
    status = "PID_PLAY_OK" if result["ok"] else "PID_PLAY_FELL"
    print(
        f"{status} max_abs_pitch={result['max_abs_pitch_deg']:.2f} "
        f"final_x={result['final']['x_m']:.3f} fell_step={result['fell_step']}",
        flush=True,
    )
finally:
    simulation_app.close()
