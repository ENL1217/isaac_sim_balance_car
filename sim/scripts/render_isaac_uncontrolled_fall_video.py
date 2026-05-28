"""Render a real Isaac/Omniverse video of the uncontrolled fall test.

Run with:
    D:\isaac\isaacsim\python.bat sim\scripts\render_isaac_uncontrolled_fall_video.py --headless
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render a true Isaac Sim video of the uncontrolled balance-car fall.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the GUI.")
    parser.add_argument(
        "--asset",
        type=Path,
        default=sim_dir / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda",
        help="Balance car USD asset path.",
    )
    parser.add_argument("--steps", type=int, default=360, help="Number of physics steps to simulate.")
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate.")
    parser.add_argument("--width", type=int, default=1280, help="Render width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Render height in pixels.")
    parser.add_argument(
        "--initial-pitch-deg",
        type=float,
        default=5.0,
        help="Initial forward/back pitch perturbation around the wheel axle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=sim_dir / "output" / "media" / "isaac_uncontrolled_fall_test.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=sim_dir / "output" / "media" / "_isaac_uncontrolled_fall_frames",
        help="Temporary directory for RGB frames.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=sim_dir / "output" / "media" / "isaac_uncontrolled_fall_video_result.json",
        help="JSON report path.",
    )
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless, "width": args.width, "height": args.height})

import carb
import cv2
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdGeom


PHYSICS_HZ = 120


def up_tilt_deg(matrix) -> float:
    local_z_world = matrix.ExtractRotationMatrix().GetRow(2)
    z = max(-1.0, min(1.0, float(local_z_world[2])))
    return math.degrees(math.acos(abs(z)))


def world_transform(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"Prim does not exist: {prim_path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


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

    omni.usd.get_context().new_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / args.fps)
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=str(args.asset.resolve()), prim_path="/World/BalanceCar")
    if args.initial_pitch_deg:
        root_prim = world.stage.GetPrimAtPath("/World/BalanceCar")
        UsdGeom.Xformable(root_prim).AddRotateYOp().Set(args.initial_pitch_deg)
    world.scene.add(Articulation(prim_paths_expr="/World/BalanceCar/base_link", name="balance_car"))
    world.reset()

    carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)

    rep.create.light(rotation=(315, 0, 35), intensity=3500, light_type="distant")
    rep.create.light(light_type="dome", intensity=450)
    camera = rep.create.camera(
        position=(0.10, -0.68, 0.20),
        look_at=(0.0, 0.0, 0.070),
        focal_length=22,
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

    base_path = "/World/BalanceCar/base_link"
    start_matrix = world_transform(world.stage, base_path)
    start_tilt = up_tilt_deg(start_matrix)
    max_tilt = start_tilt

    steps_per_frame = max(1, round(PHYSICS_HZ / args.fps))
    frame_count = max(1, math.ceil(args.steps / steps_per_frame))
    simulated_steps = 0

    for _ in range(frame_count):
        for _ in range(steps_per_frame):
            if simulated_steps >= args.steps:
                break
            world.step(render=True)
            simulated_steps += 1
            matrix = world_transform(world.stage, base_path)
            max_tilt = max(max_tilt, up_tilt_deg(matrix))
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)

    writer.detach()
    for _ in range(20):
        simulation_app.update()
    time.sleep(1.0)

    frame_paths = sorted(args.frames_dir.rglob("rgb*.png"))
    write_video(frame_paths, args.output, args.fps)

    end_matrix = world_transform(world.stage, base_path)
    end_tilt = up_tilt_deg(end_matrix)
    result = {
        "ok": True,
        "asset_path": str(args.asset.resolve()),
        "video_path": str(args.output.resolve()),
        "frames_dir": str(args.frames_dir.resolve()),
        "rendered_frame_count": len(frame_paths),
        "steps": args.steps,
        "simulated_steps": simulated_steps,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "initial_pitch_deg": args.initial_pitch_deg,
        "start_tilt_deg": start_tilt,
        "end_tilt_deg": end_tilt,
        "max_tilt_deg": max_tilt,
        "fell_without_controller": end_tilt > 25.0 or max_tilt > 35.0,
        "render_source": "isaac_sim_replicator_basicwriter_rgb_during_world_step",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"ISAAC_FALL_VIDEO_OK video={args.output.resolve()} frames={len(frame_paths)} "
        f"start_tilt={start_tilt:.2f} end_tilt={end_tilt:.2f} max_tilt={max_tilt:.2f}",
        flush=True,
    )


try:
    main()
finally:
    simulation_app.close()
