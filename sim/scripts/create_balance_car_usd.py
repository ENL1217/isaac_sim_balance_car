"""Create a simplified Yahboom-style two-wheel balance car USD asset.

Run with:
    D:\isaac\isaacsim\python.bat sim\scripts\create_balance_car_usd.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from isaacsim import SimulationApp


@dataclass(frozen=True)
class BalanceCarSpec:
    """All values derived from real Yahboom STM32 平衡小車 datasheets and
    product spec photos (May 2026). See `docs/SIM_VS_REAL.md` (TBD) for the
    source of every number.

    Real Yahboom (target sim2real ≥ 90%):
    - Total mass 942 g, dimensions 194 × 84 × 139.59 mm, three-tier
      acrylic+metal structure with battery at the bottom.
    - Wheels: 67 mm OD (33.5 mm radius), 30 mm wide rubber tire on
      blue plastic hub, wheel track ~170 mm.
    - Motors: 2× GB37-520 with 1:30 reduction. Output-shaft no-load
      speed 330 rpm = 34.6 rad/s; stall current 6.5 A at 12 V; back-
      calculated output stall torque ≈ 2.17 N·m; back-EMF coefficient
      ≈ 0.063 N·m·s/rad (drives joint-level damping below).
    - IMU MPU6050; encoder 11-line Hall AB-phase × gear 30 × 2 edges =
      780 pulses per wheel revolution (matches STM32 firmware mode).
    - Battery at the bottom layer → CoM very low (~55 mm absolute from
      ground), unlike the old USD's 139 mm CoM which made the cart
      much harder to balance than real hardware.
    """
    wheel_radius_m: float = 0.0335                 # real 67 mm OD
    # Compromise between real 30 mm and the old 50 mm hack. PhysX cylinder
    # collision is a convex-hull approximation; a 30 mm tire gives narrow
    # contact patches that amplify numerical lateral perturbations into
    # monotonic roll drift, tipping the cart in ~15-30 s with NO input
    # (GUI mode, C4/C5 testing). 40 mm widens the contact enough to
    # stabilise numerics without straying too far from the real ~30 mm
    # spec. Real cart's lateral friction is computed across this contact
    # patch — wider patch ≈ smoother restoring force at tiny tilt angles.
    wheel_width_m: float = 0.040
    wheel_track_m: float = 0.170                   # body 84 mm + motor stack
    # Real Yahboom body is 84 mm front-to-back (X) × 140 mm side-to-side (Y).
    # Previous spec had X/Y swapped (visual rotated 90° from real). Note the
    # 194 mm spec dimension is wheel-to-wheel WITH wheels included.
    plate_xyz_m: tuple[float, float, float] = (0.084, 0.140, 0.003)
    arduino_xyz_m: tuple[float, float, float] = (0.060, 0.082, 0.004)
    battery_xyz_m: tuple[float, float, float] = (0.040, 0.080, 0.024)
    motor_radius_m: float = 0.019                  # GB37 body radius
    motor_length_m: float = 0.050                  # GB37 motor + 22 mm gearbox
    # Total chassis (everything except wheels): 942 g − 2 × 60 g wheels ≈ 822 g
    chassis_mass_kg: float = 0.822
    # CoM relative to base_link origin (which sits at z=axle = 33.5 mm). The
    # absolute height we target is ~55 mm (battery dominates → low CoM).
    # 55 − 33.5 = ~22 mm → 0.021 m in base_link-local frame.
    chassis_center_of_mass_z_m: float = 0.021
    # Each wheel: rubber tire + plastic hub + axle adapter ≈ 60 g.
    wheel_mass_kg: float = 0.060
    # GB37 stall torque computed from datasheet (V=12, I_stall=6.5,
    # K_e=0.0111 V·s/rad motor side, gear ratio 30): 0.0723 × 30 = 2.17 N·m
    # per wheel at the output shaft. Use this as the joint drive force cap.
    motor_drive_max_force: float = 2.17
    # Back-EMF coefficient = stall_torque / no_load_speed = 2.17 / 34.6 ≈
    # 0.063 N·m·s/rad. PhysX joint damping reproduces the linear motor
    # torque-speed curve naturally: at ω = 34.6 rad/s the joint applies
    # exactly −2.17 N·m back-EMF, cancelling stall torque → ω cannot
    # exceed the real motor's no-load speed.
    motor_drive_damping: float = 0.063
    # Output-shaft no-load speed; informational, used for max_velocity below.
    motor_no_load_omega_rad_s: float = 34.6
    # Coulomb (dry / bearing) friction at the wheel joint. Real GB37 has
    # bearing + gearbox stiction roughly 2–5% of stall torque. Without
    # this, sim wheels are perfectly frictionless and any micro-command
    # from the PID accumulates into limit-cycle hunting → cart eventually
    # tips itself over from drift. Real Yahboom users report standing
    # indefinitely (until battery dies after ~20 h), which is consistent
    # with stiction + PWM dead-band absorbing sub-threshold commands.
    motor_joint_friction_nm: float = 0.05
    # Tire friction (rubber on flat surface). Tuned for sim2real behaviour
    # under continuous turn commands. Earlier values 0.9/0.8 were typical
    # rubber-on-rough-floor coefficients but in PhysX produced "sticky"
    # wheels that refused to slip even at large lateral loads -- so any
    # continuous A/D press created accumulating roll torque on the single-
    # axle chassis until it tipped. Real rubber DOES slip when lateral
    # force exceeds threshold (mu * N), distributing the load and saving
    # the cart. Lowering to 0.7/0.5 matches rubber-on-smooth-tile and
    # restores the slip behaviour during turns.
    tire_static_friction: float = 0.7
    tire_dynamic_friction: float = 0.5
    tire_restitution: float = 0.05
    tire_density_kg_per_m3: float = 1200.0


def parse_args() -> argparse.Namespace:
    default_asset = Path(__file__).resolve().parents[1] / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda"
    default_report = Path(__file__).resolve().parents[1] / "output" / "create_balance_car_usd_result.json"

    parser = argparse.ArgumentParser(description="Create a primitive USD asset for the two-wheel balance car.")
    parser.add_argument("--output", type=Path, default=default_asset, help="USD/USD ASCII asset path to write.")
    parser.add_argument("--report", type=Path, default=default_report, help="JSON report path to write.")
    return parser.parse_args()


def set_display_color(prim: Usd.Prim, color: tuple[float, float, float]) -> None:
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])


def set_xform(
    prim: Usd.Prim,
    translate: tuple[float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
    rotate_xyz_deg: tuple[float, float, float] | None = None,
) -> None:
    xform = UsdGeom.Xformable(prim)
    if translate is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rotate_xyz_deg is not None:
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz_deg))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))


def add_rigid_link(
    stage: Usd.Stage,
    path: str,
    translate: tuple[float, float, float],
    mass_kg: float,
    enable_gyroscopic_forces: bool = True,
) -> Usd.Prim:
    link = UsdGeom.Xform.Define(stage, path)
    set_xform(link.GetPrim(), translate=translate)
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr(mass_kg)
    # PhysX rigid body tuning. Defaults from Flamingo Edu v1 (the closest
    # 2-wheel balance-robot reference): keep linear/angular damping at 0,
    # cap depenetration velocity at 1 m/s so contact corrections don't
    # punt the body. Enable gyroscopic forces (WheeledLab MUSHR sets it;
    # Flamingo does not, so it's mildly experimental but harmless).
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(link.GetPrim())
    physx_rb.CreateEnableGyroscopicForcesAttr(enable_gyroscopic_forces)
    physx_rb.CreateMaxDepenetrationVelocityAttr(1.0)
    physx_rb.CreateLinearDampingAttr(0.0)
    physx_rb.CreateAngularDampingAttr(0.0)
    physx_rb.CreateRetainAccelerationsAttr(False)
    return link.GetPrim()


def add_box(
    stage: Usd.Stage,
    path: str,
    translate: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = True,
    rotate_xyz_deg: tuple[float, float, float] | None = None,
) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    set_xform(cube.GetPrim(), translate=translate, rotate_xyz_deg=rotate_xyz_deg, scale=size_xyz)
    set_display_color(cube.GetPrim(), color)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


def add_cylinder(
    stage: Usd.Stage,
    path: str,
    radius_m: float,
    height_m: float,
    axis: str,
    color: tuple[float, float, float],
    collision: bool = True,
    translate: tuple[float, float, float] | None = None,
    rotate_xyz_deg: tuple[float, float, float] | None = None,
) -> Usd.Prim:
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(radius_m)
    cylinder.CreateHeightAttr(height_m)
    cylinder.CreateAxisAttr(axis)
    set_xform(cylinder.GetPrim(), translate=translate, rotate_xyz_deg=rotate_xyz_deg)
    set_display_color(cylinder.GetPrim(), color)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    return cylinder.GetPrim()


def add_rounded_plate(
    stage: Usd.Stage,
    path: str,
    translate: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
) -> None:
    x_size, y_size, z_size = size_xyz
    white = (0.95, 0.95, 0.90)
    dark_edge = (0.70, 0.70, 0.66)
    add_box(stage, f"{path}_slab", translate, size_xyz, white)

    corner_radius = 0.014
    for index, sx in enumerate((-1.0, 1.0)):
        for sy in (-1.0, 1.0):
            add_cylinder(
                stage,
                f"{path}_rounded_corner_{index}_{'p' if sy > 0 else 'n'}",
                corner_radius,
                z_size * 1.04,
                "Z",
                white,
                collision=False,
                translate=(
                    translate[0] + sx * (x_size / 2.0 - corner_radius),
                    translate[1] + sy * (y_size / 2.0 - corner_radius),
                    translate[2],
                ),
            )

    hole_positions = [
        (-x_size * 0.40, -y_size * 0.34),
        (-x_size * 0.40, y_size * 0.34),
        (x_size * 0.40, -y_size * 0.34),
        (x_size * 0.40, y_size * 0.34),
        (0.0, -y_size * 0.28),
        (0.0, y_size * 0.28),
    ]
    for i, (x, y) in enumerate(hole_positions):
        add_cylinder(
            stage,
            f"{path}_mount_hole_{i}",
            0.003,
            z_size * 1.10,
            "Z",
            dark_edge,
            collision=False,
            translate=(translate[0] + x, translate[1] + y, translate[2] + z_size * 0.05),
        )


def add_standoff_set(
    stage: Usd.Stage,
    base_link_path: str,
    spec: BalanceCarSpec,
    name: str,
    z_center_m: float,
    height_m: float,
    x_factor: float = 0.42,
    y_factor: float = 0.39,
) -> None:
    x_size, y_size, _ = spec.plate_xyz_m
    brass = (0.75, 0.54, 0.12)
    for i, sx in enumerate((-1.0, 1.0)):
        for sy in (-1.0, 1.0):
            add_cylinder(
                stage,
                f"{base_link_path}/brass_standoff_{name}_{i}_{'p' if sy > 0 else 'n'}",
                0.0032,
                height_m,
                "Z",
                brass,
                collision=False,
                translate=(sx * (x_size * x_factor), sy * (y_size * y_factor), z_center_m),
            )


def add_controller_stack(stage: Usd.Stage, base_link_path: str, spec: BalanceCarSpec) -> None:
    # PCB drive board sits at the MIDDLE tier of the real cart (~80 mm
    # absolute → 80 − 33.5 = 46 mm relative to base_link origin).
    board_z = 0.046
    add_box(
        stage,
        f"{base_link_path}/controller_blue_board",
        (0.006, 0.0, board_z),
        spec.arduino_xyz_m,
        (0.02, 0.23, 0.55),
        collision=False,
    )
    add_box(stage, f"{base_link_path}/usb_b_socket", (0.052, -0.020, board_z + 0.008), (0.024, 0.018, 0.014), (0.72, 0.75, 0.78), collision=False)
    add_box(stage, f"{base_link_path}/barrel_jack", (0.047, 0.032, board_z + 0.010), (0.026, 0.020, 0.020), (0.03, 0.03, 0.035), collision=False)
    add_box(stage, f"{base_link_path}/main_chip", (-0.006, -0.005, board_z + 0.007), (0.026, 0.022, 0.006), (0.02, 0.02, 0.02), collision=False)
    add_box(stage, f"{base_link_path}/driver_chip", (-0.036, 0.018, board_z + 0.007), (0.022, 0.018, 0.006), (0.02, 0.02, 0.02), collision=False)
    add_box(stage, f"{base_link_path}/black_header_left", (0.000, -0.036, board_z + 0.007), (0.084, 0.006, 0.010), (0.015, 0.015, 0.015), collision=False)
    add_box(stage, f"{base_link_path}/black_header_right", (0.000, 0.036, board_z + 0.007), (0.084, 0.006, 0.010), (0.015, 0.015, 0.015), collision=False)
    add_cylinder(stage, f"{base_link_path}/red_reset_button", 0.004, 0.003, "Z", (0.75, 0.05, 0.03), collision=False, translate=(-0.034, -0.022, board_z + 0.011))


def add_battery_pack(stage: Usd.Stage, base_link_path: str, spec: BalanceCarSpec) -> None:
    # Real Yahboom keeps the 2200 mAh battery in a closed compartment at the
    # BOTTOM tier — this dominates the cart's CoM (which is why the chassis
    # CoM in the spec is ~22 mm above base_link origin = ~55 mm absolute).
    battery_z = 0.005
    add_box(
        stage,
        f"{base_link_path}/upper_black_battery_holder",
        (-0.008, 0.0, battery_z),
        spec.battery_xyz_m,
        (0.02, 0.02, 0.025),
        collision=False,
    )
    add_box(
        stage,
        f"{base_link_path}/blue_battery_cells_visible",
        (-0.016, 0.0, battery_z + 0.016),
        (0.082, 0.030, 0.008),
        (0.00, 0.14, 0.80),
        collision=False,
    )
    for y in (-0.012, 0.012):
        add_cylinder(
            stage,
            f"{base_link_path}/battery_front_screw_{'p' if y > 0 else 'n'}",
            0.003,
            0.002,
            "X",
            (0.78, 0.78, 0.76),
            collision=False,
            translate=(0.042, y, battery_z + 0.003),
        )


def add_motor_and_axle_visuals(stage: Usd.Stage, base_link_path: str, spec: BalanceCarSpec) -> None:
    half_track = spec.wheel_track_m / 2.0
    motor_color = (0.66, 0.64, 0.60)
    bracket_color = (0.18, 0.18, 0.18)
    motor_y = half_track - (spec.wheel_width_m / 2.0 + spec.motor_length_m / 2.0)
    for side_name, sign in (("left", 1.0), ("right", -1.0)):
        y = sign * motor_y
        outer_cap_y = sign * (motor_y + spec.motor_length_m / 2.0 - 0.002)
        add_cylinder(
            stage,
            f"{base_link_path}/{side_name}_silver_motor_can",
            spec.motor_radius_m,
            spec.motor_length_m,
            "Y",
            motor_color,
            collision=False,
            translate=(-0.008, y, 0.0),
        )
        add_cylinder(
            stage,
            f"{base_link_path}/{side_name}_black_motor_end_cap",
            spec.motor_radius_m * 0.96,
            0.004,
            "Y",
            (0.03, 0.03, 0.035),
            collision=False,
            translate=(-0.008, outer_cap_y, 0.0),
        )
        add_box(stage, f"{base_link_path}/{side_name}_motor_bracket", (0.016, y, 0.020), (0.030, 0.006, 0.030), bracket_color, collision=False)
        add_box(stage, f"{base_link_path}/{side_name}_black_motor_wires", (-0.030, sign * 0.025, 0.028), (0.040, 0.004, 0.004), (0.01, 0.01, 0.012), collision=False, rotate_xyz_deg=(0.0, 0.0, sign * 18.0))
    add_cylinder(stage, f"{base_link_path}/axle_visual", 0.003, spec.wheel_track_m + 0.030, "Y", (0.58, 0.58, 0.58), collision=False, translate=(0.0, 0.0, 0.0))


def add_tire_material(stage: Usd.Stage, path: str, spec: BalanceCarSpec) -> str:
    """Create a PhysicsMaterial for the tire-on-ground contact.

    Friction values chosen to match real Yahboom rubber on smooth tile/wood
    (roughly μ_s 0.7–0.9). Bound to each wheel's tire cylinder so the tire
    contact uses these properties even when the cart spawns on world surfaces
    that have a different (or no) material.
    """
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(spec.tire_static_friction)
    api.CreateDynamicFrictionAttr(spec.tire_dynamic_friction)
    api.CreateRestitutionAttr(spec.tire_restitution)
    api.CreateDensityAttr(spec.tire_density_kg_per_m3)
    return path


def bind_physics_material(prim: Usd.Prim, material_path: str) -> None:
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(
        UsdShade.Material(prim.GetStage().GetPrimAtPath(material_path)),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def add_wheel_visual(
    stage: Usd.Stage,
    wheel_link_path: str,
    spec: BalanceCarSpec,
    side_name: str,
    tire_material_path: str | None = None,
) -> None:
    # Cylinder collision (back to standard PhysX cylinder collider). The
    # multi-sphere experiment from C10 was based on a wrong assumption: we
    # thought the contact-geometry was the root cause of the turn-tip
    # instability. Looking at WheeledLab's MUSHR_CFG showed that real
    # production-grade Isaac Lab wheeled robots just use plain cylinder /
    # mesh collision; the missing ingredient was PhysX's
    # `enableGyroscopicForces` flag (set on the wheel rigid bodies in
    # add_rigid_link below). A spinning wheel has gyroscopic angular
    # momentum that resists axle-tilt; without that flag PhysX skips the
    # computation and the wheel acts like a non-spinning hoop — losing
    # the real cart's built-in roll stabilisation.
    tire_prim = add_cylinder(
        stage,
        f"{wheel_link_path}/black_rubber_tire",
        spec.wheel_radius_m,
        spec.wheel_width_m,
        "Y",
        (0.015, 0.015, 0.015),
    )
    if tire_material_path is not None:
        bind_physics_material(tire_prim, tire_material_path)
    # Remaining visual decorations (hub, axle cap, spokes, tread blocks).
    add_cylinder(
        stage,
        f"{wheel_link_path}/blue_wheel_hub",
        spec.wheel_radius_m * 0.68,
        spec.wheel_width_m * 1.04,
        "Y",
        (0.01, 0.06, 0.58),
        collision=False,
    )
    add_cylinder(
        stage,
        f"{wheel_link_path}/silver_axle_cap",
        spec.wheel_radius_m * 0.16,
        spec.wheel_width_m * 1.10,
        "Y",
        (0.75, 0.75, 0.74),
        collision=False,
    )
    for angle in range(0, 180, 30):
        add_box(
            stage,
            f"{wheel_link_path}/{side_name}_blue_spoke_{angle}",
            (0.0, 0.0, 0.0),
            (spec.wheel_radius_m * 1.18, 0.003, 0.003),
            (0.02, 0.08, 0.70),
            collision=False,
            rotate_xyz_deg=(0.0, float(angle), 0.0),
        )
    # NOTE: The original USD also drew 15 black tire-tread blocks around the
    # circumference. They were removed when the collision changed to multi-
    # sphere — visually the red collision spheres now play the "studded tread"
    # role and adding black tread on top made the wheel look like a paddle
    # wheel with two layers of studs. Keep this clean.


def add_revolute_wheel_joint(
    stage: Usd.Stage,
    path: str,
    base_link_path: str,
    wheel_link_path: str,
    base_local_pos: tuple[float, float, float],
    drive_max_force: float,
    drive_damping: float,
    joint_friction_nm: float = 0.0,
) -> Usd.Prim:
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(base_link_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(wheel_link_path)])
    joint.CreateAxisAttr("Y")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*base_local_pos))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLowerLimitAttr(-1.0e9)
    joint.CreateUpperLimitAttr(1.0e9)

    # Apply PhysxJointAPI to set BOTH Coulomb friction (bearing stiction)
    # and Armature (reflected rotor inertia). Earlier attempts used a
    # self-invented "physics:armature" custom attribute that PhysX silently
    # ignored — found by grepping Isaac Sim source for the real API
    # (PhysxSchema.PhysxJointAPI.GetArmatureAttr() at
    # isaacsim.core.experimental.prims/.../articulation.py:1352).
    physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
    if joint_friction_nm > 0.0:
        physx_joint.CreateJointFrictionAttr(joint_friction_nm)
    # ARMATURE = reflected rotor + gearbox inertia. Flamingo Edu v1 uses
    # 0.01 on wheel joints (a 2-wheel balance robot, our closest reference).
    # Without armature PhysX treats the wheel as a massless-rotor system;
    # differential commands accelerate the wheels instantly, producing huge
    # lateral impulses that flip the single-axle chassis on first A/D press.
    # With armature the wheel acceleration is rate-limited by reflected
    # motor+gearbox inertia (real GB37 with 1:30 gear has ~1e-3 kg·m²
    # reflected to output shaft).
    # 0.01 = match Flamingo Edu v1 exactly. Tried 0.02 but it caused
    # the cart to self-tip at t≈27 s standing — beyond a certain
    # threshold PhysX's solver appears to over-correct the joint and
    # introduce its own drift. 0.01 gave 49 s stand stability in GUI.
    physx_joint.CreateArmatureAttr(0.01)

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(drive_damping)
    drive.CreateMaxForceAttr(drive_max_force)
    return joint.GetPrim()


def build_stage(output_path: Path, spec: BalanceCarSpec) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path).replace("\\", "/"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/BalanceCar")
    stage.SetDefaultPrim(root.GetPrim())

    root_path = "/BalanceCar"
    base_link_path = f"{root_path}/base_link"
    left_wheel_path = f"{root_path}/left_wheel_link"
    right_wheel_path = f"{root_path}/right_wheel_link"
    joints_path = f"{root_path}/joints"
    looks_path = f"{root_path}/Looks"

    UsdGeom.Scope.Define(stage, joints_path)
    UsdGeom.Scope.Define(stage, looks_path)
    tire_material_path = add_tire_material(
        stage, f"{looks_path}/rubber_tire_material", spec
    )

    axle_z = spec.wheel_radius_m
    half_track = spec.wheel_track_m / 2.0
    base = add_rigid_link(stage, base_link_path, (0.0, 0.0, axle_z), spec.chassis_mass_kg)
    base_mass_api = UsdPhysics.MassAPI.Apply(base)
    base_mass_api.CreateCenterOfMassAttr().Set(
        Gf.Vec3f(0.0, 0.0, spec.chassis_center_of_mass_z_m)
    )
    # EXPLICIT diagonal inertia tensor. Without this, PhysX auto-computes
    # inertia from the AABB of child collision shapes — which means
    # changing the visual plate dimensions silently changes the cart's
    # pitch/roll/yaw response. With explicit inertia the dynamics are
    # decoupled from visual geometry.
    #
    # Approximating chassis as a rectangular box of dimensions
    # X_body × Y_body × Z_body = 0.084 × 0.140 × 0.140 m (real Yahboom
    # spec: 84 mm front-back × 140 mm side-side × 140 mm tall):
    #   Ixx (roll)  = m * (Y² + Z²) / 12
    #   Iyy (pitch) = m * (X² + Z²) / 12
    #   Izz (yaw)   = m * (X² + Y²) / 12
    body_x = spec.plate_xyz_m[0]    # 0.084
    body_y = spec.plate_xyz_m[1]    # 0.140
    body_z = 0.140                  # cart height
    ixx = spec.chassis_mass_kg * (body_y ** 2 + body_z ** 2) / 12.0
    iyy = spec.chassis_mass_kg * (body_x ** 2 + body_z ** 2) / 12.0
    izz = spec.chassis_mass_kg * (body_x ** 2 + body_y ** 2) / 12.0
    base_mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(ixx, iyy, izz))
    base_mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    UsdPhysics.ArticulationRootAPI.Apply(base)
    physx_articulation = PhysxSchema.PhysxArticulationAPI.Apply(base)
    physx_articulation.CreateArticulationEnabledAttr(True)
    # Solver iterations. WheeledLab MUSHR_CFG uses (pos=4, vel=0). The
    # extra position-solver passes help PhysX resolve wheel-ground
    # contacts and the joint constraints together without numerical jitter
    # that would manifest as roll drift or wheel slip in our single-axle
    # chassis. Defaults are typically (pos=8, vel=1) — using 4/0 trades a
    # bit of CPU for the more direct contact/joint solver path that
    # WheeledLab found works on their wheeled platforms.
    # Solver iter counts copy Flamingo Edu v1's (pos=4, vel=1). Earlier
    # WheeledLab MUSHR used (4, 0) but Flamingo, being a 2-wheel balance
    # robot like us, uses vel=1 — and Flamingo is empirically known to
    # work for this exact platform type.
    physx_articulation.CreateSolverPositionIterationCountAttr(4)
    physx_articulation.CreateSolverVelocityIterationCountAttr(1)
    physx_articulation.CreateSleepThresholdAttr(0.005)
    physx_articulation.CreateStabilizationThresholdAttr(0.001)

    # Plate stack compressed to match the real cart's 140 mm total height
    # (exploded view in the product spec: bottom metal chassis + battery →
    # middle PCB drive board → top acrylic with OLED + ultrasonic). All
    # offsets are relative to base_link origin which sits at z = wheel_radius.
    lower_plate_z = 0.005   # bottom chassis / battery shelf, abs ~38 mm
    middle_plate_z = 0.046  # PCB drive board, abs ~80 mm
    upper_plate_z = 0.106   # top acrylic deck, abs ~140 mm
    add_rounded_plate(stage, f"{base_link_path}/lower_white_plate", (0.0, 0.0, lower_plate_z), spec.plate_xyz_m)
    add_rounded_plate(stage, f"{base_link_path}/middle_white_plate", (0.0, 0.0, middle_plate_z), spec.plate_xyz_m)
    add_rounded_plate(stage, f"{base_link_path}/upper_white_plate", (0.0, 0.0, upper_plate_z), spec.plate_xyz_m)
    add_standoff_set(stage, base_link_path, spec, "lower_outer", (lower_plate_z + middle_plate_z) / 2.0, middle_plate_z - lower_plate_z)
    add_standoff_set(stage, base_link_path, spec, "upper_outer", (middle_plate_z + upper_plate_z) / 2.0, upper_plate_z - middle_plate_z)
    add_standoff_set(stage, base_link_path, spec, "lower_inner", (lower_plate_z + middle_plate_z) / 2.0, middle_plate_z - lower_plate_z, x_factor=0.25, y_factor=0.24)
    add_controller_stack(stage, base_link_path, spec)
    add_battery_pack(stage, base_link_path, spec)
    add_motor_and_axle_visuals(stage, base_link_path, spec)

    add_rigid_link(stage, left_wheel_path, (0.0, half_track, axle_z), spec.wheel_mass_kg)
    add_wheel_visual(stage, left_wheel_path, spec, "left", tire_material_path)

    add_rigid_link(stage, right_wheel_path, (0.0, -half_track, axle_z), spec.wheel_mass_kg)
    add_wheel_visual(stage, right_wheel_path, spec, "right", tire_material_path)

    left_joint_path = f"{joints_path}/left_wheel_joint"
    right_joint_path = f"{joints_path}/right_wheel_joint"
    add_revolute_wheel_joint(
        stage,
        left_joint_path,
        base_link_path,
        left_wheel_path,
        (0.0, half_track, 0.0),
        spec.motor_drive_max_force,
        spec.motor_drive_damping,
        spec.motor_joint_friction_nm,
    )
    add_revolute_wheel_joint(
        stage,
        right_joint_path,
        base_link_path,
        right_wheel_path,
        (0.0, -half_track, 0.0),
        spec.motor_drive_max_force,
        spec.motor_drive_damping,
        spec.motor_joint_friction_nm,
    )

    stage.GetRootLayer().Save()
    return {
        "asset_path": str(output_path),
        "default_prim": "/BalanceCar",
        "articulation_root": base_link_path,
        "base_link": base_link_path,
        "wheel_joints": [left_joint_path, right_joint_path],
        "wheel_links": [left_wheel_path, right_wheel_path],
        "wheel_layout": "segway_style_left_right_same_axle",
        "visual_style": "yahboom_balance_robot_three_plate",
        "spec": asdict(spec),
    }


args = parse_args()
simulation_app = SimulationApp({"headless": True})

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def main() -> None:
    spec = BalanceCarSpec()
    result = build_stage(args.output.resolve(), spec)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"CREATE_BALANCE_CAR_USD_OK asset={result['asset_path']}", flush=True)


try:
    main()
except Exception as e:  # noqa: BLE001 — print before sim app shuts down
    import traceback
    print("CREATE_BALANCE_CAR_USD_FAIL", flush=True)
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
