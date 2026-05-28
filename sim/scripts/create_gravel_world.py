"""Create a gravel world: flat ground sprinkled with many small static stones.

Stone radius is bounded below the wheel radius so the cart can roll over them
instead of tripping.

Run with:
    D:\\isaac\\isaacsim\\python.bat sim\\scripts\\create_gravel_world.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from isaacsim import SimulationApp


@dataclass(frozen=True)
class GravelWorldSpec:
    ground_xy_m: tuple[float, float] = (6.0, 3.0)
    ground_thickness_m: float = 0.02
    stone_count: int = 300
    stone_min_radius_m: float = 0.005
    stone_max_radius_m: float = 0.018  # well under the wheel radius 0.034
    scatter_x_m: tuple[float, float] = (-1.0, 4.0)
    scatter_y_m: tuple[float, float] = (-0.6, 0.6)
    seed: int = 42
    static_friction: float = 1.0
    dynamic_friction: float = 0.8
    restitution: float = 0.0


def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create a gravel USD world.")
    parser.add_argument(
        "--output",
        type=Path,
        default=sim_dir / "assets" / "gravel_world" / "gravel_world.usda",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=sim_dir / "output" / "create_gravel_world_result.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stone-count", type=int, default=300)
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


def add_collidable_sphere(stage, path, translate, radius, color, physics_material_path=None):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    if physics_material_path:
        binding = UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim())
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


def build_stage(output_path: Path, spec: GravelWorldSpec) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path).replace("\\", "/"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/GravelWorld")
    stage.SetDefaultPrim(root.GetPrim())

    UsdGeom.Scope.Define(stage, "/GravelWorld/PhysicsMaterials")
    material_path = add_physics_material(
        stage,
        "/GravelWorld/PhysicsMaterials/gravel",
        spec.static_friction,
        spec.dynamic_friction,
        spec.restitution,
    )

    # Ground plate
    add_collidable_box(
        stage,
        "/GravelWorld/ground",
        translate=(spec.ground_xy_m[0] / 2.0 - 1.0, 0.0, -spec.ground_thickness_m / 2.0),
        size_xyz=(spec.ground_xy_m[0], spec.ground_xy_m[1], spec.ground_thickness_m),
        color=(0.55, 0.50, 0.45),
        physics_material_path=material_path,
    )

    rng = random.Random(spec.seed)
    UsdGeom.Scope.Define(stage, "/GravelWorld/Stones")
    for i in range(spec.stone_count):
        radius = rng.uniform(spec.stone_min_radius_m, spec.stone_max_radius_m)
        x = rng.uniform(*spec.scatter_x_m)
        y = rng.uniform(*spec.scatter_y_m)
        # Press stone into the ground a bit so half of it is exposed.
        z = radius * 0.6
        # Random gray tone
        g = 0.4 + 0.2 * rng.random()
        add_collidable_sphere(
            stage,
            f"/GravelWorld/Stones/stone_{i:04d}",
            translate=(x, y, z),
            radius=radius,
            color=(g, g * 0.95, g * 0.9),
            physics_material_path=material_path,
        )

    stage.GetRootLayer().Save()
    return {
        "asset_path": str(output_path),
        "default_prim": "/GravelWorld",
        "stone_count": spec.stone_count,
        "stone_max_radius_m": spec.stone_max_radius_m,
        "wheel_radius_m_reference": 0.034,
        "spec": asdict(spec),
    }


def main() -> None:
    spec = GravelWorldSpec(seed=args.seed, stone_count=args.stone_count)
    result = build_stage(args.output.resolve(), spec)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"CREATE_GRAVEL_WORLD_OK asset={result['asset_path']} stones={result['stone_count']}",
        flush=True,
    )


try:
    main()
finally:
    simulation_app.close()
