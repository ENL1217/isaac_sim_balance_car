"""Create a slope-world USD asset: flat approach + tilted ramp + flat top platform.

Run with:
    D:\\isaac\\isaacsim\\python.bat sim\\scripts\\create_slope_world.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from isaacsim import SimulationApp


@dataclass(frozen=True)
class SlopeWorldSpec:
    slope_deg: float = 30.0
    ramp_horizontal_m: float = 0.5
    approach_x_m: float = 2.0
    top_x_m: float = 2.0
    width_y_m: float = 1.5
    plate_thickness_m: float = 0.02
    static_friction: float = 1.2
    dynamic_friction: float = 1.0
    restitution: float = 0.0


def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create a USD asset for the slope-test world.")
    parser.add_argument(
        "--output",
        type=Path,
        default=sim_dir / "assets" / "slope_world" / "slope_world.usda",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=sim_dir / "output" / "create_slope_world_result.json",
    )
    parser.add_argument("--slope-deg", type=float, default=30.0)
    parser.add_argument("--ramp-horizontal-m", type=float, default=0.5)
    parser.add_argument("--approach-x-m", type=float, default=2.0)
    parser.add_argument("--top-x-m", type=float, default=2.0)
    parser.add_argument("--width-y-m", type=float, default=1.5)
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


def add_collidable_box(
    stage: Usd.Stage,
    path: str,
    translate: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    color: tuple[float, float, float],
    physics_material_path: str | None = None,
    rotate_xyz_deg: tuple[float, float, float] | None = None,
) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rotate_xyz_deg is not None:
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz_deg))
    xform.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if physics_material_path is not None:
        binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        binding.Bind(
            UsdShade.Material(stage.GetPrimAtPath(physics_material_path)),
            bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            materialPurpose="physics",
        )


def add_physics_material(
    stage: Usd.Stage,
    path: str,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
) -> str:
    material = UsdShade.Material.Define(stage, path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(static_friction)
    physics_material.CreateDynamicFrictionAttr(dynamic_friction)
    physics_material.CreateRestitutionAttr(restitution)
    return path


def build_stage(output_path: Path, spec: SlopeWorldSpec) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path).replace("\\", "/"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/SlopeWorld")
    stage.SetDefaultPrim(root.GetPrim())

    theta = math.radians(spec.slope_deg)
    ramp_surface_len = spec.ramp_horizontal_m / math.cos(theta)
    ramp_rise = spec.ramp_horizontal_m * math.tan(theta)

    UsdGeom.Scope.Define(stage, "/SlopeWorld/PhysicsMaterials")
    material_path = add_physics_material(
        stage,
        "/SlopeWorld/PhysicsMaterials/high_grip",
        spec.static_friction,
        spec.dynamic_friction,
        spec.restitution,
    )

    add_collidable_box(
        stage,
        "/SlopeWorld/approach",
        translate=(-spec.approach_x_m / 2.0, 0.0, -spec.plate_thickness_m / 2.0),
        size_xyz=(spec.approach_x_m, spec.width_y_m, spec.plate_thickness_m),
        color=(0.70, 0.70, 0.72),
        physics_material_path=material_path,
    )

    # Ramp slab is rotated -slope_deg around Y. Its top surface needs to go from
    # (0, 0, 0) at the approach edge to (ramp_horizontal_m, 0, ramp_rise) at the top
    # edge. The slab's geometric center is offset BELOW its top surface midpoint by
    # half the plate thickness in the LOCAL -Z direction. After RotateY(-slope), the
    # local -Z direction maps to world (+sin θ, 0, -cos θ). So the slab center is
    # the top-surface midpoint plus that offset.
    ramp_top_midpoint = (spec.ramp_horizontal_m / 2.0, 0.0, ramp_rise / 2.0)
    half_thickness = spec.plate_thickness_m / 2.0
    ramp_center = (
        ramp_top_midpoint[0] + half_thickness * math.sin(theta),
        0.0,
        ramp_top_midpoint[2] - half_thickness * math.cos(theta),
    )
    add_collidable_box(
        stage,
        "/SlopeWorld/ramp",
        translate=ramp_center,
        size_xyz=(ramp_surface_len, spec.width_y_m, spec.plate_thickness_m),
        rotate_xyz_deg=(0.0, -spec.slope_deg, 0.0),
        color=(0.65, 0.55, 0.45),
        physics_material_path=material_path,
    )

    add_collidable_box(
        stage,
        "/SlopeWorld/top",
        translate=(
            spec.ramp_horizontal_m + spec.top_x_m / 2.0,
            0.0,
            ramp_rise - spec.plate_thickness_m / 2.0,
        ),
        size_xyz=(spec.top_x_m, spec.width_y_m, spec.plate_thickness_m),
        color=(0.70, 0.70, 0.72),
        physics_material_path=material_path,
    )

    stage.GetRootLayer().Save()
    return {
        "asset_path": str(output_path),
        "default_prim": "/SlopeWorld",
        "slope_deg": spec.slope_deg,
        "ramp_horizontal_m": spec.ramp_horizontal_m,
        "ramp_surface_len_m": ramp_surface_len,
        "ramp_rise_m": ramp_rise,
        "approach_x_m": spec.approach_x_m,
        "top_x_m": spec.top_x_m,
        "width_y_m": spec.width_y_m,
        "plate_thickness_m": spec.plate_thickness_m,
        "spec": asdict(spec),
    }


def main() -> None:
    spec = SlopeWorldSpec(
        slope_deg=args.slope_deg,
        ramp_horizontal_m=args.ramp_horizontal_m,
        approach_x_m=args.approach_x_m,
        top_x_m=args.top_x_m,
        width_y_m=args.width_y_m,
    )
    result = build_stage(args.output.resolve(), spec)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"CREATE_SLOPE_WORLD_OK asset={result['asset_path']} slope={spec.slope_deg} rise={result['ramp_rise_m']:.3f}",
        flush=True,
    )


try:
    main()
finally:
    simulation_app.close()
