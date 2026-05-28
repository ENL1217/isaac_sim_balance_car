"""Create a flat line-following track: ground + a black stripe along +X at y=0.

The stripe is a thin, NON-collidable visual cube. The simulated IR sensor is
implemented in `play_pid_effort.py` using the car's body pose vs the known
stripe geometry (no raycasts yet — this is an idealised sensor model that can
be replaced with PhysX raycasts later without changing the controller).

Run with:
    D:\\isaac\\isaacsim\\python.bat sim\\scripts\\create_line_world.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from isaacsim import SimulationApp


@dataclass(frozen=True)
class LineWorldSpec:
    ground_xy_m: tuple[float, float] = (10.0, 3.0)
    ground_thickness_m: float = 0.02
    line_length_m: float = 8.0
    line_width_m: float = 0.04
    line_thickness_m: float = 0.001
    line_centre_offset_x_m: float = 3.0  # line spans x in [offset - L/2, offset + L/2]
    line_y_m: float = 0.0
    static_friction: float = 1.0
    dynamic_friction: float = 0.8
    restitution: float = 0.0


def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create a line-following USD world.")
    parser.add_argument(
        "--output",
        type=Path,
        default=sim_dir / "assets" / "line_world" / "line_world.usda",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=sim_dir / "output" / "create_line_world_result.json",
    )
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


def add_box(stage, path, translate, size_xyz, color, collidable=True, physics_material_path=None):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collidable:
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


def build_stage(output_path: Path, spec: LineWorldSpec) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path).replace("\\", "/"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/LineWorld")
    stage.SetDefaultPrim(root.GetPrim())

    UsdGeom.Scope.Define(stage, "/LineWorld/PhysicsMaterials")
    material_path = add_physics_material(
        stage,
        "/LineWorld/PhysicsMaterials/track",
        spec.static_friction,
        spec.dynamic_friction,
        spec.restitution,
    )

    add_box(
        stage,
        "/LineWorld/ground",
        translate=(spec.line_centre_offset_x_m, 0.0, -spec.ground_thickness_m / 2.0),
        size_xyz=(spec.ground_xy_m[0], spec.ground_xy_m[1], spec.ground_thickness_m),
        color=(0.92, 0.92, 0.90),
        collidable=True,
        physics_material_path=material_path,
    )

    add_box(
        stage,
        "/LineWorld/line",
        translate=(spec.line_centre_offset_x_m, spec.line_y_m, spec.line_thickness_m / 2.0),
        size_xyz=(spec.line_length_m, spec.line_width_m, spec.line_thickness_m),
        color=(0.05, 0.05, 0.05),
        collidable=False,
    )

    stage.GetRootLayer().Save()
    return {
        "asset_path": str(output_path),
        "default_prim": "/LineWorld",
        "line_y_m": spec.line_y_m,
        "line_centre_offset_x_m": spec.line_centre_offset_x_m,
        "line_length_m": spec.line_length_m,
        "line_x_range": [
            spec.line_centre_offset_x_m - spec.line_length_m / 2.0,
            spec.line_centre_offset_x_m + spec.line_length_m / 2.0,
        ],
        "spec": asdict(spec),
    }


def main() -> None:
    spec = LineWorldSpec()
    result = build_stage(args.output.resolve(), spec)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"CREATE_LINE_WORLD_OK asset={result['asset_path']}", flush=True)


try:
    main()
finally:
    simulation_app.close()
