"""Create a stairs world for descending tests (downstairs).

Layout: a long flat top platform, then N descending steps, then a flat ground at
the bottom. Step height defaults to 2 * wheel_radius = 0.068 m.

Run with:
    D:\\isaac\\isaacsim\\python.bat sim\\scripts\\create_stairs_world.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from isaacsim import SimulationApp


@dataclass(frozen=True)
class StairsWorldSpec:
    step_count: int = 4
    step_height_m: float = 0.068
    step_depth_m: float = 0.25
    width_y_m: float = 1.5
    top_x_m: float = 1.5
    bottom_x_m: float = 2.0
    plate_thickness_m: float = 0.02
    static_friction: float = 1.2
    dynamic_friction: float = 1.0
    restitution: float = 0.0


def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create a stairs USD world.")
    parser.add_argument(
        "--output",
        type=Path,
        default=sim_dir / "assets" / "stairs_world" / "stairs_world.usda",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=sim_dir / "output" / "create_stairs_world_result.json",
    )
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


def add_collidable_box(stage, path, translate, size_xyz, color, physics_material_path=None):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if physics_material_path:
        binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        binding.Bind(
            UsdShade.Material(stage.GetPrimAtPath(physics_material_path)),
            bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            materialPurpose="physics",
        )


def add_physics_material(stage, path, sf, df, rest):
    mat = UsdShade.Material.Define(stage, path)
    p = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    p.CreateStaticFrictionAttr(sf)
    p.CreateDynamicFrictionAttr(df)
    p.CreateRestitutionAttr(rest)
    return path


def build_stage(output_path: Path, spec: StairsWorldSpec) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path).replace("\\", "/"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/StairsWorld")
    stage.SetDefaultPrim(root.GetPrim())

    UsdGeom.Scope.Define(stage, "/StairsWorld/PhysicsMaterials")
    material_path = add_physics_material(
        stage,
        "/StairsWorld/PhysicsMaterials/high_grip",
        spec.static_friction,
        spec.dynamic_friction,
        spec.restitution,
    )

    top_height = spec.step_count * spec.step_height_m

    add_collidable_box(
        stage,
        "/StairsWorld/top_platform",
        translate=(-spec.top_x_m / 2.0, 0.0, top_height - spec.plate_thickness_m / 2.0),
        size_xyz=(spec.top_x_m, spec.width_y_m, spec.plate_thickness_m),
        color=(0.70, 0.70, 0.72),
        physics_material_path=material_path,
    )

    # Build steps descending from x=0 in +X direction.
    # Each step extends from x = i*step_depth to (i+1)*step_depth at z = (step_count-1-i)*step_height
    for i in range(spec.step_count):
        z_top = (spec.step_count - 1 - i) * spec.step_height_m
        # The step is a solid block from ground to its top to give it physical mass.
        block_top_z = z_top
        block_height = block_top_z  # block extends from z=0 up to z_top
        if block_height < 1e-4:
            continue
        add_collidable_box(
            stage,
            f"/StairsWorld/step_{i}",
            translate=(
                i * spec.step_depth_m + spec.step_depth_m / 2.0,
                0.0,
                block_top_z / 2.0,
            ),
            size_xyz=(spec.step_depth_m, spec.width_y_m, block_height),
            color=(0.65 + 0.02 * i, 0.55, 0.45),
            physics_material_path=material_path,
        )

    # Bottom ground (extends to the right past the last step)
    bottom_x_start = spec.step_count * spec.step_depth_m
    add_collidable_box(
        stage,
        "/StairsWorld/bottom",
        translate=(
            bottom_x_start + spec.bottom_x_m / 2.0,
            0.0,
            -spec.plate_thickness_m / 2.0,
        ),
        size_xyz=(spec.bottom_x_m, spec.width_y_m, spec.plate_thickness_m),
        color=(0.70, 0.70, 0.72),
        physics_material_path=material_path,
    )

    stage.GetRootLayer().Save()
    return {
        "asset_path": str(output_path),
        "default_prim": "/StairsWorld",
        "top_height_m": top_height,
        "spec": asdict(spec),
    }


def main() -> None:
    spec = StairsWorldSpec()
    result = build_stage(args.output.resolve(), spec)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"CREATE_STAIRS_WORLD_OK asset={result['asset_path']} top_height={result['top_height_m']:.3f}", flush=True)


try:
    main()
finally:
    simulation_app.close()
