"""Render a real Isaac/Omniverse snapshot of the balance-car USD.

Run with:
    D:\isaac\isaacsim\python.bat sim\scripts\render_balance_car_snapshot.py --headless
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render a true 3D snapshot of the balance-car USD.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the GUI.")
    parser.add_argument(
        "--asset",
        type=Path,
        default=sim_dir / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda",
        help="Balance car USD asset path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=sim_dir / "output" / "media" / "balance_car_isaac_3d_snapshot.png",
        help="PNG snapshot path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=sim_dir / "output" / "media" / "balance_car_isaac_3d_snapshot_result.json",
        help="JSON report path.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Render width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Render height in pixels.")
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless, "width": args.width, "height": args.height})

import carb
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage


def render_snapshot() -> Path:
    if not args.asset.exists():
        raise FileNotFoundError(f"Balance car asset does not exist: {args.asset}")

    omni.usd.get_context().new_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=str(args.asset.resolve()), prim_path="/World/BalanceCar")
    world.reset()

    carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)

    rep.create.light(rotation=(315, 0, 35), intensity=3500, light_type="distant")
    rep.create.light(light_type="dome", intensity=450)
    camera = rep.create.camera(
        position=(0.62, -0.30, 0.30),
        look_at=(0.0, 0.0, 0.088),
        focal_length=28,
        clipping_range=(0.01, 10.0),
    )
    render_product = rep.create.render_product(camera, (args.width, args.height))

    temp_dir = args.output.parent / "_replicator_snapshot"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(temp_dir.resolve()), rgb=True)
    writer.attach(render_product)

    for _ in range(10):
        simulation_app.update()
    rep.orchestrator.step(rt_subframes=16, delta_time=0.0, pause_timeline=True)
    writer.detach()
    for _ in range(20):
        simulation_app.update()
    time.sleep(1.0)

    rgb_images = sorted(temp_dir.rglob("rgb*.png"))
    if not rgb_images:
        raise RuntimeError(f"No RGB snapshot was written under {temp_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(rgb_images[-1], args.output)
    return rgb_images[-1]


def main() -> None:
    source_png = render_snapshot()
    result = {
        "ok": True,
        "asset_path": str(args.asset.resolve()),
        "snapshot_path": str(args.output.resolve()),
        "replicator_source_path": str(source_png.resolve()),
        "width": args.width,
        "height": args.height,
        "render_source": "isaac_sim_replicator_basicwriter_rgb",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"RENDER_BALANCE_CAR_SNAPSHOT_OK snapshot={args.output.resolve()}", flush=True)


try:
    main()
finally:
    simulation_app.close()
