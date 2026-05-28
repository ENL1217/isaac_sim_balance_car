"""Headless test for balance_board.usda — load, step physics, log plank pitch.

Runs simulation for `--steps` ticks at 1 kHz, reading the plank's live
world transform each tick. Outputs CSV with (time, plank_pitch_deg, plank_z,
load_x, load_z) so we can verify the plank actually tilts when the load
sits on its exit side.

Usage:
    python.bat sim/scripts/test_balance_board.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from isaacsim import SimulationApp  # noqa: E402


def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Headless test of balance_board.usda")
    parser.add_argument(
        "--asset",
        type=Path,
        default=sim_dir / "assets" / "balance_board" / "balance_board.usda",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=sim_dir / "output" / "balance_board_test.csv",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": True})

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

world = World(physics_dt=args.dt, rendering_dt=args.dt)
add_reference_to_stage(
    usd_path=str(args.asset.resolve()),
    prim_path="/World/BalanceBoard",
)
world.reset()

plank_prim = world.stage.GetPrimAtPath("/World/BalanceBoard/plank")
load_prim = world.stage.GetPrimAtPath("/World/BalanceBoard/load")
if not plank_prim.IsValid() or not load_prim.IsValid():
    print("MISSING_PRIM plank or load not found", flush=True)
    simulation_app.close()
    raise SystemExit(1)

args.csv.parent.mkdir(parents=True, exist_ok=True)
csv_file = args.csv.open("w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow(
    ["step", "time_s", "plank_pitch_deg", "plank_z", "load_x", "load_y", "load_z"]
)


def read_xform(prim) -> tuple[float, float, float, float]:
    """Return (pitch_deg_around_y, world_x, world_y, world_z) of the prim."""
    xform = UsdGeom.Xformable(prim)
    m = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    wx, wy, wz = float(m[3][0]), float(m[3][1]), float(m[3][2])
    # USD row-vector RotateY(theta) sends +X to (cos, 0, -sin). Pitch around Y:
    m00 = float(m[0][0])
    m20 = float(m[2][0])
    pitch = math.degrees(math.atan2(m20, m00))
    return pitch, wx, wy, wz


# Initial state
plank_pitch, _, _, plank_z = read_xform(plank_prim)
load_pitch, load_x, load_y, load_z = read_xform(load_prim)
print(
    f"INIT plank_pitch={plank_pitch:.3f} plank_z={plank_z:.4f} "
    f"load_pos=({load_x:.3f},{load_y:.3f},{load_z:.3f})",
    flush=True,
)

for step in range(args.steps):
    world.step(render=False)
    plank_pitch, _, _, plank_z = read_xform(plank_prim)
    _, load_x, load_y, load_z = read_xform(load_prim)
    writer.writerow(
        [step, step * args.dt, plank_pitch, plank_z, load_x, load_y, load_z]
    )
    if step % 200 == 0:
        print(
            f"t={step*args.dt:.3f}s plank_pitch={plank_pitch:+7.3f} "
            f"plank_z={plank_z:.4f} load=({load_x:.3f},{load_z:.3f})",
            flush=True,
        )

csv_file.close()
print(f"DONE csv={args.csv}", flush=True)
simulation_app.close()
