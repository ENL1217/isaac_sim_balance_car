"""Minimal Isaac Sim 5.0 standalone smoke test.

Run with:
    D:\isaac\isaacsim\python.bat sim\scripts\smoke_test.py --headless
"""

import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal Isaac Sim gravity smoke test.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the GUI.")
    parser.add_argument("--steps", type=int, default=180, help="Number of simulation steps to run.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "smoke_test_result.json",
        help="Path to write a small JSON result report.",
    )
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid


def main() -> None:
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()

    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/SmokeTestCube",
            name="smoke_test_cube",
            position=np.array([0.0, 0.0, 1.0]),
            size=0.25,
            mass=1.0,
            color=np.array([0.1, 0.6, 1.0]),
        )
    )

    world.reset()
    start_z = float(cube.get_world_pose()[0][2])

    for _ in range(args.steps):
        world.step(render=not args.headless)

    end_z = float(cube.get_world_pose()[0][2])
    result = {
        "ok": end_z < start_z,
        "start_z": start_z,
        "end_z": end_z,
        "steps": args.steps,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if result["ok"]:
        print(f"SMOKE_TEST_OK start_z={start_z:.3f} end_z={end_z:.3f} steps={args.steps}", flush=True)
    else:
        print(f"SMOKE_TEST_FAIL start_z={start_z:.3f} end_z={end_z:.3f} steps={args.steps}", flush=True)
        raise RuntimeError("Cube did not fall under gravity.")


try:
    main()
finally:
    simulation_app.close()
