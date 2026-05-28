"""Verify that the generated balance car articulation has spinning wheel joints.

Run with:
    D:\isaac\isaacsim\python.bat sim\scripts\verify_wheel_spin.py --headless
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Load the balance car USD and verify wheel joint motion.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the GUI.")
    parser.add_argument(
        "--asset",
        type=Path,
        default=sim_dir / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda",
        help="Balance car USD asset path.",
    )
    parser.add_argument("--steps", type=int, default=120, help="Number of simulation steps to run.")
    parser.add_argument("--target-velocity", type=float, default=20.0, help="Wheel target velocity in rad/s.")
    parser.add_argument(
        "--report",
        type=Path,
        default=sim_dir / "output" / "wheel_spin_result.json",
        help="Path to write a JSON result report.",
    )
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdPhysics


def as_flat_list(values) -> list[float]:
    array = np.asarray(values, dtype=float)
    return array.reshape(-1).tolist()


def main() -> None:
    if not args.asset.exists():
        raise FileNotFoundError(f"Balance car asset does not exist: {args.asset}")

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=str(args.asset.resolve()), prim_path="/World/BalanceCar")

    articulation_root = "/World/BalanceCar/base_link"
    robot = world.scene.add(Articulation(prim_paths_expr=articulation_root, name="balance_car"))
    world.reset()

    joint_names = list(robot.joint_names)
    wheel_joint_names = ["left_wheel_joint", "right_wheel_joint"]
    missing_joints = [name for name in wheel_joint_names if name not in joint_names]
    if missing_joints:
        raise RuntimeError(f"Missing wheel joints {missing_joints}; available joints: {joint_names}")

    start_positions = as_flat_list(robot.get_joint_positions(joint_names=wheel_joint_names))

    # Same-sign wheel velocity is enough for this smoke test; PID/yaw sign conventions come later.
    velocity_targets = np.array([[args.target_velocity, args.target_velocity]], dtype=float)
    for _ in range(args.steps):
        robot.set_joint_velocity_targets(velocity_targets, joint_names=wheel_joint_names)
        world.step(render=not args.headless)

    end_positions = as_flat_list(robot.get_joint_positions(joint_names=wheel_joint_names))
    deltas = [end - start for start, end in zip(start_positions, end_positions)]

    stage = world.stage
    articulation_prim = stage.GetPrimAtPath(articulation_root)
    has_articulation_root = bool(articulation_prim.HasAPI(UsdPhysics.ArticulationRootAPI))

    ok = has_articulation_root and all(abs(delta) > 0.25 for delta in deltas)
    result = {
        "ok": ok,
        "asset_path": str(args.asset.resolve()),
        "articulation_root": articulation_root,
        "has_articulation_root": has_articulation_root,
        "joint_names": joint_names,
        "wheel_joint_names": wheel_joint_names,
        "start_positions_rad": start_positions,
        "end_positions_rad": end_positions,
        "delta_positions_rad": deltas,
        "steps": args.steps,
        "target_velocity_rad_s": args.target_velocity,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")

    status = "WHEEL_SPIN_TEST_OK" if ok else "WHEEL_SPIN_TEST_FAIL"
    print(
        f"{status} left_delta={deltas[0]:.3f} right_delta={deltas[1]:.3f} steps={args.steps}",
        flush=True,
    )
    if not ok:
        raise RuntimeError(f"Wheel spin verification failed: {result}")


try:
    main()
finally:
    simulation_app.close()
