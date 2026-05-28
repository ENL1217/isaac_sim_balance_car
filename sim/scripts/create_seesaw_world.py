"""Create a seesaw world: ground + fulcrum + plank on a revolute joint.

Run with:
    D:\\isaac\\isaacsim\\python.bat sim\\scripts\\create_seesaw_world.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from isaacsim import SimulationApp


@dataclass(frozen=True)
class SeesawWorldSpec:
    plank_length_m: float = 2.0
    plank_width_m: float = 0.40
    plank_thickness_m: float = 0.04
    plank_mass_kg: float = 2.0
    pivot_height_m: float = 0.10
    fulcrum_base_xy_m: tuple[float, float] = (0.30, 0.50)
    ground_xy_m: tuple[float, float] = (5.0, 3.0)
    ground_thickness_m: float = 0.02
    static_friction: float = 1.2
    dynamic_friction: float = 1.0
    restitution: float = 0.0


def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create a USD asset for the seesaw test world.")
    parser.add_argument(
        "--output",
        type=Path,
        default=sim_dir / "assets" / "seesaw_world" / "seesaw_world.usda",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=sim_dir / "output" / "create_seesaw_world_result.json",
    )
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": True})

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


def add_collidable_box(
    stage: Usd.Stage,
    path: str,
    translate: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    color: tuple[float, float, float],
    physics_material_path: str | None = None,
) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
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
    return cube.GetPrim()


def add_rigid_box(
    stage: Usd.Stage,
    path: str,
    translate: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    color: tuple[float, float, float],
    mass_kg: float,
    physics_material_path: str | None = None,
) -> Usd.Prim:
    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.Xformable(xform.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*translate))
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(mass_kg)

    cube = UsdGeom.Cube.Define(stage, f"{path}/visual")
    cube.CreateSizeAttr(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if physics_material_path is not None:
        binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        binding.Bind(
            UsdShade.Material(stage.GetPrimAtPath(physics_material_path)),
            bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            materialPurpose="physics",
        )
    return xform.GetPrim()


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


def build_stage(output_path: Path, spec: SeesawWorldSpec) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path).replace("\\", "/"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/SeesawWorld")
    stage.SetDefaultPrim(root.GetPrim())

    UsdGeom.Scope.Define(stage, "/SeesawWorld/PhysicsMaterials")
    material_path = add_physics_material(
        stage,
        "/SeesawWorld/PhysicsMaterials/high_grip",
        spec.static_friction,
        spec.dynamic_friction,
        spec.restitution,
    )

    # Ground plate (so the car has something to drive on either side of the seesaw)
    add_collidable_box(
        stage,
        "/SeesawWorld/ground",
        translate=(0.0, 0.0, -spec.ground_thickness_m / 2.0),
        size_xyz=(spec.ground_xy_m[0], spec.ground_xy_m[1], spec.ground_thickness_m),
        color=(0.70, 0.70, 0.72),
        physics_material_path=material_path,
    )

    # Fulcrum: a small static support box. It's collidable so it doesn't intersect
    # the plank visually, but the plank's rotation is constrained by the revolute
    # joint, not by contact with the fulcrum.
    add_collidable_box(
        stage,
        "/SeesawWorld/fulcrum",
        translate=(0.0, 0.0, spec.pivot_height_m / 2.0),
        size_xyz=(spec.fulcrum_base_xy_m[0], spec.fulcrum_base_xy_m[1], spec.pivot_height_m),
        color=(0.55, 0.55, 0.60),
        physics_material_path=material_path,
    )

    # Plank as a rigid body centered above the fulcrum.
    plank_z = spec.pivot_height_m + spec.plank_thickness_m / 2.0
    add_rigid_box(
        stage,
        "/SeesawWorld/plank",
        translate=(0.0, 0.0, plank_z),
        size_xyz=(spec.plank_length_m, spec.plank_width_m, spec.plank_thickness_m),
        color=(0.78, 0.62, 0.45),
        mass_kg=spec.plank_mass_kg,
        physics_material_path=material_path,
    )

    # Revolute joint: plank rotates around world Y axis at the pivot point.
    # body0 = world (None), body1 = plank.
    joint_path = "/SeesawWorld/plank_pivot"
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/SeesawWorld/plank")])
    joint.CreateAxisAttr("Y")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, spec.pivot_height_m))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, -spec.plank_thickness_m / 2.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLowerLimitAttr(-25.0)
    joint.CreateUpperLimitAttr(25.0)

    stage.GetRootLayer().Save()
    return {
        "asset_path": str(output_path),
        "default_prim": "/SeesawWorld",
        "plank_length_m": spec.plank_length_m,
        "plank_width_m": spec.plank_width_m,
        "plank_thickness_m": spec.plank_thickness_m,
        "pivot_height_m": spec.pivot_height_m,
        "joint_path": joint_path,
        "spec": asdict(spec),
    }


def main() -> None:
    spec = SeesawWorldSpec()
    result = build_stage(args.output.resolve(), spec)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"CREATE_SEESAW_WORLD_OK asset={result['asset_path']}", flush=True)


try:
    main()
finally:
    simulation_app.close()
