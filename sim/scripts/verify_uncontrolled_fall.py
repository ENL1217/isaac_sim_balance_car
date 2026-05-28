"""Verify what the balance car does with no balancing controller.

Run with:
    D:\isaac\isaacsim\python.bat sim\scripts\verify_uncontrolled_fall.py --headless
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run an uncontrolled balance-car fall test.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the GUI.")
    parser.add_argument(
        "--asset",
        type=Path,
        default=sim_dir / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda",
        help="Balance car USD asset path.",
    )
    parser.add_argument("--steps", type=int, default=360, help="Number of simulation steps to run.")
    parser.add_argument(
        "--initial-pitch-deg",
        type=float,
        default=5.0,
        help="Initial forward/back pitch perturbation around the wheel axle in degrees; use 0 for upright placement.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=sim_dir / "output" / "uncontrolled_fall_result.json",
        help="Path to write a JSON result report.",
    )
    return parser.parse_args()


args = parse_args()
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdGeom


def up_tilt_deg(matrix) -> float:
    # Third row is the local Z axis expressed in world coordinates for USD row-vector matrices.
    local_z_world = matrix.ExtractRotationMatrix().GetRow(2)
    z = max(-1.0, min(1.0, float(local_z_world[2])))
    return math.degrees(math.acos(abs(z)))


def world_transform(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"Prim does not exist: {prim_path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def main() -> None:
    if not args.asset.exists():
        raise FileNotFoundError(f"Balance car asset does not exist: {args.asset}")

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=str(args.asset.resolve()), prim_path="/World/BalanceCar")
    if args.initial_pitch_deg:
        root_prim = world.stage.GetPrimAtPath("/World/BalanceCar")
        UsdGeom.Xformable(root_prim).AddRotateYOp().Set(args.initial_pitch_deg)
    world.scene.add(Articulation(prim_paths_expr="/World/BalanceCar/base_link", name="balance_car"))
    world.reset()

    base_path = "/World/BalanceCar/base_link"
    start_matrix = world_transform(world.stage, base_path)
    start_pos = start_matrix.ExtractTranslation()
    start_tilt = up_tilt_deg(start_matrix)

    max_tilt = start_tilt
    for _ in range(args.steps):
        world.step(render=not args.headless)
        matrix = world_transform(world.stage, base_path)
        max_tilt = max(max_tilt, up_tilt_deg(matrix))

    end_matrix = world_transform(world.stage, base_path)
    end_pos = end_matrix.ExtractTranslation()
    end_tilt = up_tilt_deg(end_matrix)
    fell = end_tilt > 25.0 or max_tilt > 35.0

    result = {
        "ok": True,
        "asset_path": str(args.asset.resolve()),
        "steps": args.steps,
        "initial_pitch_deg": args.initial_pitch_deg,
        "start_z_m": float(start_pos[2]),
        "end_z_m": float(end_pos[2]),
        "start_tilt_deg": start_tilt,
        "end_tilt_deg": end_tilt,
        "max_tilt_deg": max_tilt,
        "fell_without_controller": fell,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")

    status = "UNCONTROLLED_FALL_EXPECTED" if fell else "UNCONTROLLED_STAYED_UP"
    print(
        f"{status} start_tilt={start_tilt:.2f} end_tilt={end_tilt:.2f} "
        f"max_tilt={max_tilt:.2f} steps={args.steps}",
        flush=True,
    )


try:
    main()
except BaseException as exc:
    print(f"UNCONTROLLED_FALL_ERROR {exc!r}", file=sys.stderr, flush=True)
    raise
finally:
    simulation_app.close()
