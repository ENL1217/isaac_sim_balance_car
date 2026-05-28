"""Standalone minimal balance-board scene — diagnostic for revolute joint physics.

Per OpenAI design guidance (see chat log 2026-05-28):
- thin rigid plank as dynamic rigid body
- eccentric pivot via revolute joint anchored to a kinematic body
- a 5 kg test load box dropped on one side of the plank
- initial plank angle = 0° (HORIZONTAL); let gravity tip naturally
- joint limits ±25°
- everything simple primitives, no fancy geometry

Usage:
    python.bat sim/scripts/create_balance_board_usd.py [--output PATH]

Then test via:
    python.bat sim/scripts/test_balance_board.py
(or load balance_board.usda in Isaac Sim and press Play)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

# Isaac Sim bootstrap must happen before any pxr import.
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402


@dataclass
class BalanceBoardSpec:
    # Plank
    plank_length_m: float = 2.0
    plank_width_m: float = 1.0
    plank_thickness_m: float = 0.04   # thicker than seesaw plank (0.003) so it
                                       # looks like a real board; OpenAI spec.
    plank_mass_kg: float = 2.0

    # Pivot geometry
    pivot_height_m: float = 0.20      # height of joint above the ground plane
    pivot_offset_x_m: float = 0.10    # joint anchor offset from plank center
                                       # along plank-local +X (toward exit side).
                                       # This creates the mass eccentricity.

    # Load box
    load_size_m: tuple = (0.20, 0.20, 0.20)
    load_mass_kg: float = 5.0
    # Load X position relative to plank center. Negative = toward entry side
    # (which is the longer/heavier side). Positive = toward exit side.
    # We place it ON THE EXIT SIDE so it should TIP the plank exit-down.
    load_x_offset_m: float = 0.6

    # Ground
    ground_size_m: tuple = (5.0, 5.0, 0.02)

    # Physics material
    static_friction: float = 1.0
    dynamic_friction: float = 0.8
    restitution: float = 0.0

    # Joint
    joint_lower_limit_deg: float = -25.0
    joint_upper_limit_deg: float = 25.0


def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create a minimal balance-board test scene.")
    parser.add_argument(
        "--output",
        type=Path,
        default=sim_dir / "assets" / "balance_board" / "balance_board.usda",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=sim_dir / "output" / "create_balance_board_result.json",
    )
    return parser.parse_args()


def add_static_box(stage, path, translate, size_xyz, color, physics_material_path):
    """Add a static collidable cube (ground / floor / static collider)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath(physics_material_path)),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def build_stage(output_path: Path, spec: BalanceBoardSpec) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("metersPerUnit", 1.0)

    root_prim = UsdGeom.Xform.Define(stage, "/BalanceBoard")
    stage.SetDefaultPrim(root_prim.GetPrim())

    # Physics scene with default gravity.
    scene = UsdPhysics.Scene.Define(stage, "/BalanceBoard/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)

    # Physics material (shared by ground and plank).
    material_path = "/BalanceBoard/physicsMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(spec.static_friction)
    physics_material.CreateDynamicFrictionAttr(spec.dynamic_friction)
    physics_material.CreateRestitutionAttr(spec.restitution)

    # Ground plane (static box covering plenty of area).
    add_static_box(
        stage,
        "/BalanceBoard/ground",
        translate=(0.0, 0.0, -spec.ground_size_m[2] / 2.0),
        size_xyz=spec.ground_size_m,
        color=(0.55, 0.55, 0.55),
        physics_material_path=material_path,
    )

    # Kinematic anchor at the pivot location. Verified to be necessary in
    # Isaac Sim 5.0 — joints with body0 left empty (world-anchored via missing
    # rel) appear to behave inconsistently; a kinematic body anchor is the
    # safest pattern.
    anchor_path = "/BalanceBoard/pivot_anchor"
    anchor_xform = UsdGeom.Xform.Define(stage, anchor_path)
    UsdGeom.Xformable(anchor_xform.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, spec.pivot_height_m)
    )
    anchor_rb = UsdPhysics.RigidBodyAPI.Apply(anchor_xform.GetPrim())
    anchor_rb.CreateRigidBodyEnabledAttr(True)
    anchor_rb.CreateKinematicEnabledAttr(True)

    # Plank — dynamic rigid body. Initial pose is HORIZONTAL (rotate_y=0).
    # The joint anchor on the plank is offset by +pivot_offset_x_m along
    # plank-local +X. That puts the plank's CoM (geometric center) on the -X
    # side of the joint, creating the mass eccentricity.
    plank_path = "/BalanceBoard/plank"
    plank_xform = UsdGeom.Xform.Define(stage, plank_path)
    # plank_center.x is at world x = 0 - pivot_offset_x_m (anchor world x is 0).
    # plank_center.z is at world z = pivot_height_m + plank_thickness/2 so that
    # the joint anchor (plank-local +pivot_offset, 0, -thickness/2) sits at
    # world z = pivot_height (the anchor's z).
    plank_center_x = 0.0 - spec.pivot_offset_x_m
    plank_center_z = spec.pivot_height_m + spec.plank_thickness_m / 2.0
    UsdGeom.Xformable(plank_xform.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(plank_center_x, 0.0, plank_center_z)
    )
    # Initial rotation 0° (horizontal). Gravity + eccentric pivot will naturally
    # tip the plank toward whichever side is heavier.
    UsdGeom.Xformable(plank_xform.GetPrim()).AddRotateXYZOp().Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )
    UsdPhysics.RigidBodyAPI.Apply(plank_xform.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(plank_xform.GetPrim()).CreateMassAttr(spec.plank_mass_kg)
    plank_physx = PhysxSchema.PhysxRigidBodyAPI.Apply(plank_xform.GetPrim())
    plank_physx.CreateLinearDampingAttr(0.0)
    plank_physx.CreateAngularDampingAttr(0.0)
    plank_physx.CreateMaxDepenetrationVelocityAttr(1.0)
    plank_physx.CreateEnableGyroscopicForcesAttr(False)

    # Plank visual + collision — a simple box.
    plank_visual = UsdGeom.Cube.Define(stage, f"{plank_path}/visual")
    plank_visual.CreateSizeAttr(1.0)
    UsdGeom.Xformable(plank_visual.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(spec.plank_length_m, spec.plank_width_m, spec.plank_thickness_m)
    )
    UsdGeom.Gprim(plank_visual.GetPrim()).CreateDisplayColorAttr(
        [Gf.Vec3f(1.0, 0.85, 0.20)]
    )
    UsdPhysics.CollisionAPI.Apply(plank_visual.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(plank_visual.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath(material_path)),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )

    # Revolute joint: plank pivots around the kinematic anchor's Y axis.
    joint_path = "/BalanceBoard/pivot_joint"
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateAxisAttr("Y")
    joint.CreateBody0Rel().SetTargets([anchor_path])
    joint.CreateBody1Rel().SetTargets([plank_path])
    # body0 (anchor) frame is at world origin (anchor world position).
    # Joint anchor in anchor-local: (0,0,0).
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    # body1 (plank) frame: joint anchor is offset +pivot_offset along plank +X,
    # at plank underside (plank-local z = -thickness/2).
    joint.CreateLocalPos1Attr(
        Gf.Vec3f(spec.pivot_offset_x_m, 0.0, -spec.plank_thickness_m / 2.0)
    )
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLowerLimitAttr(spec.joint_lower_limit_deg)
    joint.CreateUpperLimitAttr(spec.joint_upper_limit_deg)
    physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
    physx_joint.CreateJointFrictionAttr(0.0)
    physx_joint.CreateArmatureAttr(0.0)
    physx_joint.CreateMaxJointVelocityAttr(1000.0)

    # Load box — a 5 kg dynamic rigid body cube dropped on the plank's exit
    # side. This creates an unbalanced torque that should tip the plank.
    load_path = "/BalanceBoard/load"
    load_xform = UsdGeom.Xform.Define(stage, load_path)
    # Position: above the plank's exit side (plank center is at x=-pivot_offset,
    # so exit edge at x = -pivot_offset + plank_length/2; we drop the load
    # somewhere between plank center and exit edge).
    load_world_x = plank_center_x + spec.load_x_offset_m
    load_world_z = (
        plank_center_z + spec.plank_thickness_m / 2.0
        + spec.load_size_m[2] / 2.0 + 0.05  # drop from 5 cm above plank top
    )
    UsdGeom.Xformable(load_xform.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(load_world_x, 0.0, load_world_z)
    )
    UsdPhysics.RigidBodyAPI.Apply(load_xform.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(load_xform.GetPrim()).CreateMassAttr(spec.load_mass_kg)

    load_visual = UsdGeom.Cube.Define(stage, f"{load_path}/visual")
    load_visual.CreateSizeAttr(1.0)
    UsdGeom.Xformable(load_visual.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(*spec.load_size_m)
    )
    UsdGeom.Gprim(load_visual.GetPrim()).CreateDisplayColorAttr(
        [Gf.Vec3f(0.20, 0.50, 0.95)]
    )
    UsdPhysics.CollisionAPI.Apply(load_visual.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(load_visual.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath(material_path)),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )

    # Lighting + camera so it's viewable in Isaac Sim GUI.
    distant = UsdLux.DistantLight.Define(stage, "/BalanceBoard/sunLight")
    distant.CreateIntensityAttr(3000.0)
    distant.CreateAngleAttr(2.0)
    UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set((-45.0, 0.0, 30.0))
    dome = UsdLux.DomeLight.Define(stage, "/BalanceBoard/domeLight")
    dome.CreateIntensityAttr(500.0)

    stage.GetRootLayer().Save()
    return {
        "asset_path": str(output_path),
        "default_prim": "/BalanceBoard",
        "plank_center_x_m": plank_center_x,
        "plank_center_z_m": plank_center_z,
        "plank_initial_rot_deg": 0.0,
        "load_world_x_m": load_world_x,
        "load_world_z_m": load_world_z,
        "joint_path": joint_path,
        "anchor_path": anchor_path,
        "spec": asdict(spec),
    }


def main() -> None:
    args = parse_args()
    spec = BalanceBoardSpec()
    result = build_stage(args.output.resolve(), spec)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"CREATE_BALANCE_BOARD_OK asset={result['asset_path']}", flush=True)


try:
    main()
finally:
    simulation_app.close()
