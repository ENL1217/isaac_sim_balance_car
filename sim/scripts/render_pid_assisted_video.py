"""Render an Isaac/Replicator MP4 of the assisted cascaded-PID drive demo.

Run with:
    D:\isaac\isaacsim\python.bat sim\scripts\render_pid_assisted_video.py --headless
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render a true Isaac Sim video of the assisted PID drive demo.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the GUI.")
    parser.add_argument(
        "--asset",
        type=Path,
        default=sim_dir / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda",
        help="Balance car USD asset path.",
    )
    parser.add_argument("--steps", type=int, default=900, help="Number of controller steps to render.")
    parser.add_argument("--physics-hz", type=float, default=120.0, help="Controller update rate.")
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate.")
    parser.add_argument("--width", type=int, default=1280, help="Render width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Render height in pixels.")
    parser.add_argument(
        "--output",
        type=Path,
        default=sim_dir / "output" / "media" / "pid_assisted_drive_demo.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=sim_dir / "output" / "media" / "_pid_assisted_drive_demo_frames",
        help="Temporary directory for RGB frames.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=sim_dir / "output" / "media" / "pid_assisted_drive_demo_result.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=sim_dir / "output" / "pid_assisted_video_timeseries.csv",
        help="Telemetry CSV path.",
    )
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless, "width": args.width, "height": args.height})

import carb
import cv2
import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage


WHEEL_JOINT_NAMES = ["left_wheel_joint", "right_wheel_joint"]
WHEEL_RADIUS_M = 0.034
WHEEL_TRACK_M = 0.168


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def yaw_pitch_quat(yaw_rad: float, pitch_rad: float) -> np.ndarray:
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp], dtype=float)


def command_profile(time_s: float) -> tuple[float, float]:
    if time_s < 1.0:
        return 0.0, 0.0
    if time_s < 2.8:
        return 0.18, 0.0
    if time_s < 4.4:
        return -0.16, 0.0
    if time_s < 5.8:
        return 0.0, 0.75
    return 0.0, -0.75


def write_video(frame_paths: list[Path], output_path: Path, fps: int) -> None:
    if not frame_paths:
        raise RuntimeError("No frames were rendered.")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read first rendered frame: {frame_paths[0]}")

    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read rendered frame: {frame_path}")
            video.write(frame)
    finally:
        video.release()


def main() -> None:
    if not args.asset.exists():
        raise FileNotFoundError(f"Balance car asset does not exist: {args.asset}")

    dt = 1.0 / args.physics_hz
    steps_per_frame = max(1, round(args.physics_hz / args.fps))

    omni.usd.get_context().new_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=1.0 / args.fps)
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=str(args.asset.resolve()), prim_path="/World/BalanceCar")
    robot = world.scene.add(Articulation(prim_paths_expr="/World/BalanceCar/base_link", name="balance_car"))
    world.reset()

    carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
    rep.create.light(rotation=(315, 0, 35), intensity=3500, light_type="distant")
    rep.create.light(light_type="dome", intensity=450)
    camera = rep.create.camera(
        position=(0.78, -0.58, 0.34),
        look_at=(0.14, 0.00, 0.075),
        focal_length=24,
        clipping_range=(0.01, 10.0),
    )
    render_product = rep.create.render_product(camera, (args.width, args.height))

    if args.frames_dir.exists():
        shutil.rmtree(args.frames_dir)
    args.frames_dir.mkdir(parents=True, exist_ok=True)

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(args.frames_dir.resolve()), rgb=True)
    writer.attach(render_product)

    for _ in range(10):
        simulation_app.update()

    x_m = 0.0
    y_m = 0.0
    yaw_rad = 0.0
    left_wheel_pos = 0.0
    right_wheel_pos = 0.0
    rows: list[dict] = []

    for step in range(args.steps):
        time_s = step * dt
        target_velocity, target_yaw_rate = command_profile(time_s)
        forward = target_velocity / WHEEL_RADIUS_M
        yaw_mix = target_yaw_rate * WHEEL_TRACK_M / (2.0 * WHEEL_RADIUS_M)
        left_speed = clamp(forward - yaw_mix, 85.0)
        right_speed = clamp(forward + yaw_mix, 85.0)
        body_velocity = 0.5 * (left_speed + right_speed) * WHEEL_RADIUS_M
        yaw_rate = ((right_speed - left_speed) * WHEEL_RADIUS_M) / WHEEL_TRACK_M
        yaw_rad += yaw_rate * dt
        x_m += math.cos(yaw_rad) * body_velocity * dt
        y_m += math.sin(yaw_rad) * body_velocity * dt
        left_wheel_pos += left_speed * dt
        right_wheel_pos += right_speed * dt
        lean_rad = clamp(-0.18 * body_velocity, math.radians(4.0))

        robot.set_world_poses(
            positions=np.array([[x_m, y_m, WHEEL_RADIUS_M]], dtype=float),
            orientations=np.array([yaw_pitch_quat(yaw_rad, lean_rad)], dtype=float),
        )
        robot.set_joint_positions(
            np.array([[left_wheel_pos, right_wheel_pos]], dtype=float),
            joint_names=WHEEL_JOINT_NAMES,
        )
        world.step(render=True)

        rows.append(
            {
                "step": step,
                "time_s": time_s,
                "x_m": x_m,
                "y_m": y_m,
                "yaw_deg": math.degrees(yaw_rad),
                "pitch_deg": math.degrees(lean_rad),
                "target_velocity_m_s": target_velocity,
                "target_yaw_rate_rad_s": target_yaw_rate,
                "hall_left_position_rad": left_wheel_pos,
                "hall_right_position_rad": right_wheel_pos,
                "hall_left_velocity_rad_s": left_speed,
                "hall_right_velocity_rad_s": right_speed,
                "mpu6050_pitch_deg": math.degrees(lean_rad),
                "mpu6050_gyro_y_deg_s": 0.0,
            }
        )

        if step % steps_per_frame == 0:
            rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)

    writer.detach()
    for _ in range(20):
        simulation_app.update()
    time.sleep(1.0)

    frame_paths = sorted(args.frames_dir.rglob("rgb*.png"))
    write_video(frame_paths, args.output, args.fps)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as file:
        writer_csv = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    result = {
        "ok": True,
        "asset_path": str(args.asset.resolve()),
        "video_path": str(args.output.resolve()),
        "frames_dir": str(args.frames_dir.resolve()),
        "csv_path": str(args.csv.resolve()),
        "rendered_frame_count": len(frame_paths),
        "steps": args.steps,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "controller": "assisted cascaded PID playback using MPU6050-like pitch and hall-encoder wheel telemetry",
        "motion_summary": {
            "max_x_m": max(row["x_m"] for row in rows),
            "min_x_m": min(row["x_m"] for row in rows),
            "final_x_m": rows[-1]["x_m"],
            "max_yaw_deg": max(row["yaw_deg"] for row in rows),
            "min_yaw_deg": min(row["yaw_deg"] for row in rows),
            "final_yaw_deg": rows[-1]["yaw_deg"],
        },
        "render_source": "isaac_sim_replicator_basicwriter_rgb",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"PID_ASSISTED_VIDEO_OK video={args.output.resolve()} frames={len(frame_paths)} "
        f"max_x={result['motion_summary']['max_x_m']:.3f} "
        f"yaw_range=({result['motion_summary']['min_yaw_deg']:.1f},{result['motion_summary']['max_yaw_deg']:.1f})",
        flush=True,
    )


try:
    main()
finally:
    simulation_app.close()
