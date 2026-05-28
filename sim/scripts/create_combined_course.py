"""Create a combined obstacle course that exposes the balance car to a
progression of obstacles in a single scene. Layout, in +X order:

    start (HIGH platform) -> stairs DOWN -> flat -> gentle ramp UP -> apex
        platform -> gentle ramp DOWN -> flat -> seesaw (entry-low, plank flips
        when cart passes pivot) -> finish

A black IR line runs along y=0 through every segment so a single line-follower
policy can be tested end-to-end.

Designed for the balance-car PID/RL teleop demo: start elevated, descend,
hop a gentle bump, then play with the seesaw.

Run with:
    D:\\isaac\\isaacsim\\python.bat sim\\scripts\\create_combined_course.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from isaacsim import SimulationApp


@dataclass(frozen=True)
class CourseSpec:
    # ------------------------------------------------------------------
    # New layout (+X order):
    #   1. start_zone  (ELEVATED at z = stairs_total_height)
    #   2. stairs_down (steps from elevated platform DOWN to z=0)
    #   3. flat_after_stairs
    #   4. ramp_up   (gentle 3 deg bump up to ramp_apex_height_m)
    #   5. apex_platform (short flat top)
    #   6. ramp_down (symmetric gentle bump back down to z=0)
    #   7. flat_before_seesaw
    #   8. seesaw    (pre-tilted entry-low; cart climbs across pivot, plank flips)
    #   9. finish_zone
    # ------------------------------------------------------------------

    # X-axis segment lengths
    start_len_m: float = 2.0
    flat_after_stairs_len_m: float = 1.5
    apex_platform_len_m: float = 0.5
    flat_before_seesaw_len_m: float = 1.0
    # Widened from 4.2 m to 7.0 m so the asymmetric plank (cart-entry side
    # 2.0 m + far side 1.0 m) AND its 1.25 m entry bevel both fit cleanly
    # inside without overlapping flat_before_seesaw.
    seesaw_lane_len_m: float = 7.0
    finish_len_m: float = 2.0

    # Stairs (descend from start platform down to ground)
    stairs_step_count: int = 4
    stairs_step_height_m: float = 0.025          # 4 × 0.025 = 0.10 m total drop
    stairs_step_depth_m: float = 0.20            # 0.20 m deep per step

    # Gentle bump (ramp_up + apex + ramp_down)
    # slope_deg matches --drive-pitch-offset-deg 3.0 used at runtime so PID
    # at lean equilibrium needs ~zero residual torque to stay on slope.
    slope_deg: float = 3.0
    ramp_apex_height_m: float = 0.03             # 0.03 / tan(3°) = 0.572 m ramp horizontal

    # Thin plates (3 mm) so a tilted slab's lower-front corner is essentially
    # at the same z as the slab top — the cylinder wheel rolls over without
    # tripping on the slab's underside edge. Earlier 20 mm plates created a
    # vertical "lip" the cylinder hit when transitioning flat→ramp, lifting
    # the cart off the ground (the "爬坡跳起來" bug). We compensate the loss
    # of visual presence by giving the ramp/seesaw slabs HIGH-CONTRAST colors
    # below.
    plate_thickness_m: float = 0.003

    # Lane width
    lane_width_y_m: float = 1.5

    # Ramp bevel (smooth lead-in so wheels do not catch on the corner).
    # For a gentle 3 deg main ramp the bevel must also be GENTLE — a 15 deg
    # bevel is much steeper than the main ramp and creates a "lip" that
    # bounces the wheel. 1.5 deg over 0.12 m gives an almost-flat lead-in
    # so the wheel rolls onto the ramp without the chassis pitching.
    ramp_bevel_horizontal_m: float = 0.12
    ramp_bevel_angle_deg: float = 1.5

    # Seesaw
    # Geometry chosen so plank tilt EQUALS the ramp_up slope (3 deg). The
    # plank is made VERY THIN (3 mm) so the cart wheel sees almost no
    # step at the plank's low edge (~9% wheel-r) and the entry bevel can
    # also be short enough to fit in the seesaw_lane approach with the
    # SAME slope as the plank — eliminating the corner kink that
    # previously perturbed the cart's roll axis.
    # We REVERTED from 25mm back to 3mm because the thicker "looks like a
    # real plank" version reintroduced the cylinder-wheel-trip-on-edge bug.
    # Visibility is recovered via bright color (see plank color below).
    plank_length_m: float = 3.0
    # Widened from 0.80 to 1.20 m. The cart drifts laterally during
    # traversal (single-axle roll instability) and the previous narrow
    # plank let the cart slide off the side when the plank flipped at
    # the pivot. 1.20 m gives ~0.5 m of slack on either side of the
    # cart's ~22 cm track width.
    plank_width_m: float = 1.20
    plank_thickness_m: float = 0.003
    plank_mass_kg: float = 1.0
    # Pivot height — plank tilts until its cart-entry edge touches ground.
    # Max tilt = asin(pivot_height / longer_side_length).
    # 0.04 m gave a barely-visible 1.4°→1.7° flip (3° total visual change).
    # 0.08 m gives 2.8°→3.4° (6° total visual change) — clearly visible
    # see-saw motion that matches what a real wobble board does.
    pivot_height_m: float = 0.080
    # The plank's revolute joint is offset from the plank center along +X
    # (toward the far/exit side of the plank) by this amount. The result
    # is that the cart-entry side of the plank is LONGER and therefore
    # HEAVIER, so gravity around the joint axis tips the plank toward
    # cart-entry until the long edge hits the ground. This avoids needing
    # a CoM-offset hack: an asymmetric plank pivoted off-center naturally
    # behaves like a seesaw resting on its longer end.
    #
    # Tuning: this offset controls how far past the pivot the cart must
    # travel before the plank flips. The flip threshold is
    #   x_flip = (plank_mass * offset * L) / (cart_mass * 1.0)
    #          ≈ offset × 1.06  (for our 1kg plank / 0.94kg cart with L≈1m).
    # At offset=0.50, x_flip≈0.53m. Exit-side plank length is only 1.0m, so
    # the cart slowed down before reaching x_flip and the plank never tipped.
    # At offset=0.15, x_flip≈0.16m — well within reach, real seesaw behavior.
    joint_offset_x_m: float = 0.15
    # Fulcrum cylinder length (along Y) must be wider than plank_width so the
    # plank doesn't fall off the side of the rolling pin during tilt.
    # Kept wider than plank_width_m = 1.20.
    fulcrum_base_xy_m: tuple[float, float] = (0.30, 1.40)

    # IR line
    line_width_m: float = 0.04
    line_thickness_m: float = 0.001

    # Physics material
    # Friction set at a level that lets the cart break static friction at
    # drive_pitch_offset ~4 deg, without locking the cart in place. Higher
    # friction blocks motion; lower friction allows roll instability.
    static_friction: float = 1.6
    dynamic_friction: float = 1.4
    restitution: float = 0.0


def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create the combined obstacle-course world.")
    parser.add_argument(
        "--output",
        type=Path,
        default=sim_dir / "assets" / "combined_course" / "combined_course.usda",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=sim_dir / "output" / "create_combined_course_result.json",
    )
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": True})

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402


def add_static_box(
    stage,
    path,
    translate,
    size_xyz,
    color,
    rotate_xyz_deg=None,
    physics_material_path=None,
    collidable=True,
):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rotate_xyz_deg is not None:
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz_deg))
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


def add_rigid_box(
    stage,
    path,
    translate,
    size_xyz,
    color,
    mass_kg,
    physics_material_path=None,
):
    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.Xformable(xform.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*translate))
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(mass_kg)

    cube = UsdGeom.Cube.Define(stage, f"{path}/visual")
    cube.CreateSizeAttr(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(Gf.Vec3f(*size_xyz))
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
    UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*translate))
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


def add_line_segment(stage, path, x_center, x_length, y, z_top, width, color=(0.05, 0.05, 0.05), rotate_y_deg=0.0):
    # Offset the line a hair along the SLAB-NORMAL direction so it floats just
    # above the underlying surface (avoiding z-fighting) instead of in world +Z.
    # For horizontal slabs the slab normal is +Z so this collapses to the same
    # +0.0005 we used before.
    offset = 0.0006
    theta = math.radians(rotate_y_deg)
    # Slab local +Z, after RotateY(angle), maps to world (sin angle, 0, cos angle).
    dx = math.sin(theta) * offset
    dz = math.cos(theta) * offset
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(x_center + dx, y, z_top + dz))
    if rotate_y_deg != 0.0:
        xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, rotate_y_deg, 0.0))
    xform.AddScaleOp().Set(Gf.Vec3f(x_length, width, 0.001))
    UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])


def build_stage(output_path: Path, spec: CourseSpec) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path).replace("\\", "/"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/CombinedCourse")
    stage.SetDefaultPrim(root.GetPrim())

    # Embed lighting directly in the USD so the scene is never pitch-black at
    # load time regardless of how the world is referenced into a parent stage.
    UsdGeom.Scope.Define(stage, "/CombinedCourse/Lights")
    distant = UsdLux.DistantLight.Define(stage, "/CombinedCourse/Lights/SunLight")
    distant.CreateIntensityAttr(1500.0)
    distant.CreateAngleAttr(2.0)
    UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))

    dome = UsdLux.DomeLight.Define(stage, "/CombinedCourse/Lights/SkyDome")
    dome.CreateIntensityAttr(300.0)

    UsdGeom.Scope.Define(stage, "/CombinedCourse/PhysicsMaterials")
    grip = add_physics_material(
        stage,
        "/CombinedCourse/PhysicsMaterials/grip",
        spec.static_friction,
        spec.dynamic_friction,
        spec.restitution,
    )

    theta = math.radians(spec.slope_deg)
    stairs_total_height = spec.stairs_step_count * spec.stairs_step_height_m
    start_platform_height = stairs_total_height  # start zone elevated by total stair drop
    # Gentle bump geometry (ramp_up + apex + ramp_down)
    ramp_apex_height = spec.ramp_apex_height_m
    bevel_theta = math.radians(spec.ramp_bevel_angle_deg)
    bevel_rise = spec.ramp_bevel_horizontal_m * math.tan(bevel_theta)
    if ramp_apex_height <= bevel_rise:
        # If apex is so low the bevel alone covers it, suppress the main ramp.
        bevel_rise = ramp_apex_height
        bevel_horizontal = ramp_apex_height / math.tan(bevel_theta) if bevel_theta > 0 else 0.0
    else:
        bevel_horizontal = spec.ramp_bevel_horizontal_m
    main_ramp_horizontal = max(0.0, (ramp_apex_height - bevel_rise) / math.tan(theta))
    main_ramp_surface_len = max(0.0, (ramp_apex_height - bevel_rise) / math.sin(theta))
    half_thickness = spec.plate_thickness_m / 2.0

    course_segments: list[dict] = []

    # 1. Start zone -- ELEVATED. The cart spawns here at z = start_platform_height.
    #    Modelled as a single tall slab from z=0 to z=start_platform_height so the
    #    elevation is visually anchored (not a floating slab).
    cursor_x = -spec.start_len_m  # start zone runs in negative X
    add_static_box(
        stage,
        "/CombinedCourse/start_zone",
        translate=(cursor_x + spec.start_len_m / 2.0, 0.0, start_platform_height / 2.0),
        size_xyz=(spec.start_len_m, spec.lane_width_y_m, start_platform_height),
        color=(0.70, 0.72, 0.78),
        physics_material_path=grip,
    )
    course_segments.append(
        {"name": "start_zone", "x_range": [cursor_x, cursor_x + spec.start_len_m], "z_top": start_platform_height}
    )
    cursor_x += spec.start_len_m  # cursor now at 0

    # 2. Stairs DESCENDING from start_platform_height down to z=0. Step i occupies
    #    x in [cursor_x + i*depth, cursor_x + (i+1)*depth] with its top at
    #    z = start_platform_height - (i+1) * step_height. Each block goes all
    #    the way down to z=0 so the cliff face is solid.
    #    The LAST step (z_top=0) was previously skipped, leaving a gap; now we
    #    always emit a block — for z_top≈0 we emit a flat slab so the ground is
    #    continuous into flat_after_stairs.
    stair_top_info: list[dict] = []
    for i in range(spec.stairs_step_count):
        z_top = start_platform_height - (i + 1) * spec.stairs_step_height_m
        x_lo = cursor_x + i * spec.stairs_step_depth_m
        x_hi = x_lo + spec.stairs_step_depth_m
        if z_top > 1e-5:
            # Real step: block from z=0 to z=z_top.
            block_center_z = z_top / 2.0
            block_height = z_top
        else:
            # Final landing step: thin slab from z=-plate_thickness to z=0.
            block_center_z = -spec.plate_thickness_m / 2.0
            block_height = spec.plate_thickness_m
            z_top = 0.0
        add_static_box(
            stage,
            f"/CombinedCourse/stair_{i}",
            translate=(
                cursor_x + (i + 0.5) * spec.stairs_step_depth_m,
                0.0,
                block_center_z,
            ),
            size_xyz=(spec.stairs_step_depth_m, spec.lane_width_y_m, block_height),
            color=(0.60 + 0.03 * i, 0.55, 0.45),
            physics_material_path=grip,
        )
        stair_top_info.append({"x_range": [x_lo, x_hi], "z_top": z_top})
    stairs_x_end = cursor_x + spec.stairs_step_count * spec.stairs_step_depth_m
    course_segments.append(
        {
            "name": "stairs_down",
            "x_range": [cursor_x, stairs_x_end],
            "step_count": spec.stairs_step_count,
            "step_height_m": spec.stairs_step_height_m,
            "step_depth_m": spec.stairs_step_depth_m,
            "stair_tops": stair_top_info,
        }
    )
    cursor_x = stairs_x_end

    # 3. Flat recovery between stairs and the gentle bump.
    add_static_box(
        stage,
        "/CombinedCourse/flat_after_stairs",
        translate=(
            cursor_x + spec.flat_after_stairs_len_m / 2.0,
            0.0,
            -spec.plate_thickness_m / 2.0,
        ),
        size_xyz=(spec.flat_after_stairs_len_m, spec.lane_width_y_m, spec.plate_thickness_m),
        color=(0.72, 0.72, 0.75),
        physics_material_path=grip,
    )
    course_segments.append(
        {
            "name": "flat_after_stairs",
            "x_range": [cursor_x, cursor_x + spec.flat_after_stairs_len_m],
            "z_top": 0.0,
        }
    )
    cursor_x += spec.flat_after_stairs_len_m

    # 4. Gentle ramp UP. Two slabs: bevel lead-in (1.5 deg over bevel_horizontal)
    #    then main ramp (slope_deg over main_ramp_horizontal) up to ramp_apex_height.
    ramp_up_x_start = cursor_x
    # 4a. bevel
    bevel_surface_len = bevel_horizontal / math.cos(bevel_theta) if bevel_horizontal > 0 else 0.0
    if bevel_surface_len > 1e-6:
        bevel_top_midpoint = (
            ramp_up_x_start + bevel_horizontal / 2.0,
            0.0,
            bevel_rise / 2.0,
        )
        bevel_center = (
            bevel_top_midpoint[0] + half_thickness * math.sin(bevel_theta),
            0.0,
            bevel_top_midpoint[2] - half_thickness * math.cos(bevel_theta),
        )
        add_static_box(
            stage,
            "/CombinedCourse/ramp_up_bevel",
            translate=bevel_center,
            size_xyz=(bevel_surface_len, spec.lane_width_y_m, spec.plate_thickness_m),
            rotate_xyz_deg=(0.0, -spec.ramp_bevel_angle_deg, 0.0),
            color=(0.98, 0.55, 0.10),  # bright orange — climb-side ramp
            physics_material_path=grip,
        )
    # 4b. main ramp
    bevel_end_x = ramp_up_x_start + bevel_horizontal
    if main_ramp_surface_len > 1e-6:
        main_top_midpoint = (
            bevel_end_x + main_ramp_horizontal / 2.0,
            0.0,
            (bevel_rise + ramp_apex_height) / 2.0,
        )
        main_center = (
            main_top_midpoint[0] + half_thickness * math.sin(theta),
            0.0,
            main_top_midpoint[2] - half_thickness * math.cos(theta),
        )
        add_static_box(
            stage,
            "/CombinedCourse/ramp_up_main",
            translate=main_center,
            size_xyz=(main_ramp_surface_len, spec.lane_width_y_m, spec.plate_thickness_m),
            rotate_xyz_deg=(0.0, -spec.slope_deg, 0.0),
            color=(0.98, 0.40, 0.05),  # deeper orange — climb-side main ramp
            physics_material_path=grip,
        )
    ramp_up_x_end = bevel_end_x + main_ramp_horizontal
    course_segments.append(
        {
            "name": "ramp_up",
            "x_range": [ramp_up_x_start, ramp_up_x_end],
            "z_top": ramp_apex_height,
            "bevel_horizontal_m": bevel_horizontal,
            "bevel_angle_deg": spec.ramp_bevel_angle_deg,
        }
    )
    cursor_x = ramp_up_x_end

    # 5. Apex platform — short flat top at ramp_apex_height.
    add_static_box(
        stage,
        "/CombinedCourse/apex_platform",
        translate=(
            cursor_x + spec.apex_platform_len_m / 2.0,
            0.0,
            ramp_apex_height - half_thickness,
        ),
        size_xyz=(spec.apex_platform_len_m, spec.lane_width_y_m, spec.plate_thickness_m),
        color=(0.72, 0.72, 0.72),
        physics_material_path=grip,
    )
    course_segments.append(
        {
            "name": "apex_platform",
            "x_range": [cursor_x, cursor_x + spec.apex_platform_len_m],
            "z_top": ramp_apex_height,
        }
    )
    cursor_x += spec.apex_platform_len_m

    # 6. Gentle ramp DOWN (mirror of ramp up).
    #    Main ramp from z=ramp_apex_height down to z=bevel_rise, then bevel
    #    from z=bevel_rise down to z=0.
    ramp_down_x_start = cursor_x
    # 6a. main ramp descending. Tilt is +slope_deg (local +X end goes DOWN in z).
    if main_ramp_surface_len > 1e-6:
        main_top_midpoint = (
            ramp_down_x_start + main_ramp_horizontal / 2.0,
            0.0,
            (ramp_apex_height + bevel_rise) / 2.0,
        )
        main_center = (
            main_top_midpoint[0] - half_thickness * math.sin(theta),
            0.0,
            main_top_midpoint[2] - half_thickness * math.cos(theta),
        )
        add_static_box(
            stage,
            "/CombinedCourse/ramp_down_main",
            translate=main_center,
            size_xyz=(main_ramp_surface_len, spec.lane_width_y_m, spec.plate_thickness_m),
            rotate_xyz_deg=(0.0, spec.slope_deg, 0.0),
            color=(0.20, 0.60, 0.98),  # bright blue — descent-side main ramp
            physics_material_path=grip,
        )
    main_down_end_x = ramp_down_x_start + main_ramp_horizontal
    # 6b. bevel descending
    if bevel_surface_len > 1e-6:
        bevel_top_midpoint = (
            main_down_end_x + bevel_horizontal / 2.0,
            0.0,
            bevel_rise / 2.0,
        )
        bevel_center = (
            bevel_top_midpoint[0] - half_thickness * math.sin(bevel_theta),
            0.0,
            bevel_top_midpoint[2] - half_thickness * math.cos(bevel_theta),
        )
        add_static_box(
            stage,
            "/CombinedCourse/ramp_down_bevel",
            translate=bevel_center,
            size_xyz=(bevel_surface_len, spec.lane_width_y_m, spec.plate_thickness_m),
            rotate_xyz_deg=(0.0, spec.ramp_bevel_angle_deg, 0.0),
            color=(0.35, 0.70, 0.98),  # bright blue — descent-side bevel
            physics_material_path=grip,
        )
    ramp_down_x_end = main_down_end_x + bevel_horizontal
    course_segments.append(
        {
            "name": "ramp_down",
            "x_range": [ramp_down_x_start, ramp_down_x_end],
            "z_top": 0.0,
            "z_top_start": ramp_apex_height,
            "bevel_horizontal_m": bevel_horizontal,
            "bevel_angle_deg": spec.ramp_bevel_angle_deg,
            "main_horizontal_m": main_ramp_horizontal,
        }
    )
    cursor_x = ramp_down_x_end

    # 7. Flat approach to the seesaw.
    add_static_box(
        stage,
        "/CombinedCourse/flat_before_seesaw",
        translate=(
            cursor_x + spec.flat_before_seesaw_len_m / 2.0,
            0.0,
            -spec.plate_thickness_m / 2.0,
        ),
        size_xyz=(spec.flat_before_seesaw_len_m, spec.lane_width_y_m, spec.plate_thickness_m),
        color=(0.72, 0.72, 0.75),
        physics_material_path=grip,
    )
    course_segments.append(
        {
            "name": "flat_before_seesaw",
            "x_range": [cursor_x, cursor_x + spec.flat_before_seesaw_len_m],
            "z_top": 0.0,
        }
    )
    cursor_x += spec.flat_before_seesaw_len_m

    # 8. Seesaw lane: a flat ground patch with a half-cylinder fulcrum and a
    #    revolute-joint plank above it.
    #
    # The seesaw_ground floor is positioned BELOW z=0 (sunken) so that the
    # plank's entry-edge bottom corner in its natural resting state at -tilt_deg
    # does NOT touch the floor. PROVED-VIA-CSV: with the floor at z=0, the
    # plank entry corner ends up at z≈-0.0002 (penetrating 0.2 mm), which
    # PhysX resolves with a contact constraint at static friction μ=1.6. The
    # high-friction contact then LOCKED the plank — even 7 N·m of cart torque
    # on the exit side couldn't break the static friction, so the plank never
    # rotated. Sinking the floor by `seesaw_floor_drop_m` removes this
    # spurious contact and lets the joint handle the plank's rest pose alone.
    seesaw_x_start = cursor_x
    seesaw_floor_drop_m = 0.01  # 1 cm below z=0 — well clear of plank rest pose
    add_static_box(
        stage,
        "/CombinedCourse/seesaw_ground",
        translate=(
            seesaw_x_start + spec.seesaw_lane_len_m / 2.0,
            0.0,
            -seesaw_floor_drop_m - spec.plate_thickness_m / 2.0,
        ),
        size_xyz=(spec.seesaw_lane_len_m, spec.lane_width_y_m, spec.plate_thickness_m),
        color=(0.72, 0.72, 0.75),
        physics_material_path=grip,
    )
    seesaw_center_x = seesaw_x_start + spec.seesaw_lane_len_m / 2.0
    # Fulcrum modelled as a HORIZONTAL CYLINDER (axis along Y, the cart's
    # cross-direction). The plank's underside rests on the cylinder's TOP
    # TANGENT — a single line contact that slides smoothly along the
    # cylinder's top arc as the plank tilts. A box fulcrum (flat top) would
    # cause the plank's underside contact point to snap between the box's
    # top edges as it tilts, which PhysX resolves as a huge impulse and
    # launches anything on the plank into the air.
    #
    # Geometry: cylinder radius = pivot_height/2. Cylinder center at
    # z = pivot_height/2 so the cylinder TOP sits at z = pivot_height
    # (= the revolute joint anchor) and the cylinder BOTTOM sits on the
    # ground at z = 0.
    fulcrum_radius = spec.pivot_height_m / 2.0
    fulcrum_center_z = spec.pivot_height_m / 2.0
    fulcrum_length_y = spec.fulcrum_base_xy_m[1]
    fulcrum_path = "/CombinedCourse/seesaw_fulcrum"
    fulcrum_cyl = UsdGeom.Cylinder.Define(stage, fulcrum_path)
    fulcrum_cyl.CreateAxisAttr("Y")
    fulcrum_cyl.CreateHeightAttr(fulcrum_length_y)
    fulcrum_cyl.CreateRadiusAttr(fulcrum_radius)
    UsdGeom.Xformable(fulcrum_cyl.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(seesaw_center_x, 0.0, fulcrum_center_z)
    )
    UsdGeom.Gprim(fulcrum_cyl.GetPrim()).CreateDisplayColorAttr(
        [Gf.Vec3f(0.55, 0.55, 0.60)]
    )
    # NO CollisionAPI on the fulcrum — earlier design (#53) had no joint and
    # relied on the cylinder's top tangent to physically support the plank.
    # The CURRENT design (#55 onwards) uses a RevoluteJoint that fully
    # constrains the plank's translation and roll/yaw, leaving only the Y
    # pivot free. With both the joint AND a collidable fulcrum, PhysX
    # generated a permanent contact constraint between the plank underside
    # and the cylinder top (they meet exactly at the joint position). The
    # high seesaw-grip friction (μ=1.6) on that contact then prevented the
    # plank from flipping when the cart crossed the pivot — the joint was
    # free to rotate but the cylinder contact wouldn't let it. Removing the
    # CollisionAPI here keeps the cylinder as a VISUAL pivot indicator while
    # letting the joint do all the physics. The cart never touches the
    # cylinder anyway (it rolls on the plank top, ~3mm above cylinder top).
    # The plank pivots on a revolute joint whose body1-local anchor is OFFSET
    # from the plank's geometric center by +joint_offset_x_m (toward the FAR
    # exit side). Because the joint is off-center, the plank's CoM at its
    # geometric center sits on the cart-entry side of the joint axis. Gravity
    # then produces a permanent torque around the joint that tips the plank
    # toward cart-entry until the long edge meets the ground.
    #
    # Resting tilt geometry: the cart-entry side of the plank, measured from
    # the joint, has length (plank_length/2 + joint_offset_x_m). Setting this
    # length's vertical drop equal to pivot_height_m gives:
    #     sin(tilt) = pivot_height_m / (plank_length/2 + joint_offset_x_m)
    half_length = spec.plank_length_m / 2.0
    longer_side_len = half_length + spec.joint_offset_x_m
    sin_tilt = min(1.0, spec.pivot_height_m / longer_side_len)
    tilt_rad = math.asin(sin_tilt)
    tilt_deg = math.degrees(tilt_rad)
    # Start plank HORIZONTAL (0°) instead of pre-tilted. Verified by both
    # the standalone balance_board test and a combined_course diagnostic
    # (plank started at 0° tipped naturally to -2.71° under gravity in 0.3 s).
    # When plank started pre-tilted at -2.78° with the entry edge resting on
    # the seesaw_ground floor, PhysX appeared to lock the entry-floor contact
    # in a way that prevented the joint from rotating despite ~7 N·m of cart
    # torque on the exit side. Starting at 0° avoids the initial floor contact
    # so the joint can freely rotate to whatever the load+gravity dictates.
    rotate_y_value = 0.0
    # The joint anchor in plank-local frame is (+joint_offset_x_m, 0,
    # -thickness/2). After rotating the plank by RotateY(rotate_y_value),
    # this anchor must land at the world fulcrum position
    # (seesaw_center_x, 0, pivot_height_m). Solve for plank center:
    #   plank_center = world_anchor − R_y(rotate_y) @ anchor_local
    # Under row-vector R_y(theta):
    #   (offset, 0, -t/2) * R_y(theta) =
    #     (offset*cos theta + (-t/2)*sin theta,
    #      0,
    #      offset*(-sin theta) + (-t/2)*cos theta)
    # With theta = -tilt:
    #   x' =  offset*cos(tilt) + (-t/2)*(-sin(tilt))
    #      =  offset*cos(tilt) + (t/2)*sin(tilt)
    #   z' =  offset*sin(tilt) + (-t/2)*cos(tilt)
    #      =  offset*sin(tilt) − (t/2)*cos(tilt)
    half_thickness = spec.plank_thickness_m / 2.0
    # Compute anchor offset using the ACTUAL initial rotation (rotate_y_value),
    # not the resting tilt (tilt_rad). For DIAGNOSTIC mode (rotate_y_value=0),
    # plank is horizontal at startup and gravity tips it down naturally.
    initial_rot_rad = math.radians(rotate_y_value)
    # Sign: rotate_y_value=-tilt sends local +X up. For row-vector convention,
    # local (offset, 0, -half_thickness) maps to:
    #   x_world_off = offset*cos(-rot) + (-half)*sin(-rot)
    #               = offset*cos(rot) - half*sin(rot)? No wait
    # Use the same derivation as the code comment but with theta=rotate_y_value:
    cos_t = math.cos(initial_rot_rad)
    sin_t = math.sin(initial_rot_rad)
    # Local (offset_x, 0, -half) under USD RotateY(theta):
    #   x' = offset_x*cos(theta) + (-half)*sin(theta) = offset_x*cos - half*sin
    #   z' = offset_x*(-sin(theta)) + (-half)*cos(theta) = -offset_x*sin - half*cos
    # Wait — re-derive cleanly. USD RotateY sends +X to (cos, 0, -sin).
    # So for point (x, 0, z), after RotateY(theta):
    #   x' = x*cos(theta) + z*sin(theta)
    #   z' = -x*sin(theta) + z*cos(theta)
    # For (offset, 0, -half):
    #   x' = offset*cos - half*sin
    #   z' = -offset*sin - half*cos
    anchor_offset_x = spec.joint_offset_x_m * cos_t - half_thickness * sin_t
    anchor_offset_z = -spec.joint_offset_x_m * sin_t - half_thickness * cos_t
    # plank_center + anchor_offset = joint_world_position
    # → plank_center = joint_world_position - anchor_offset
    plank_center = (
        seesaw_center_x - anchor_offset_x,
        0.0,
        spec.pivot_height_m - anchor_offset_z,
    )
    # Build the plank rigid body. The joint below constrains all DOF except
    # Y-axis rotation, so the plank cannot translate or roll — only pivot
    # like a real seesaw hinge.
    plank_path = "/CombinedCourse/seesaw_plank"
    plank_xform = UsdGeom.Xform.Define(stage, plank_path)
    UsdGeom.Xformable(plank_xform.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*plank_center))
    UsdGeom.Xformable(plank_xform.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(0.0, rotate_y_value, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(plank_xform.GetPrim()).CreateRigidBodyEnabledAttr(True)
    plank_mass_api = UsdPhysics.MassAPI.Apply(plank_xform.GetPrim())
    plank_mass_api.CreateMassAttr(spec.plank_mass_kg)
    # Explicit zero damping so the joint rotates freely. PhysX default
    # angularDamping is 0.05 which would slowly bleed off the flip motion.
    plank_physx = PhysxSchema.PhysxRigidBodyAPI.Apply(plank_xform.GetPrim())
    plank_physx.CreateLinearDampingAttr(0.0)
    plank_physx.CreateAngularDampingAttr(0.0)
    plank_physx.CreateMaxDepenetrationVelocityAttr(1.0)
    plank_physx.CreateEnableGyroscopicForcesAttr(False)
    # No CoM offset — the joint being off-center already produces the
    # asymmetric gravity torque we want. CoM at the plank's geometric center.
    plank_visual = UsdGeom.Cube.Define(stage, f"{plank_path}/visual")
    plank_visual.CreateSizeAttr(1.0)
    UsdGeom.Xformable(plank_visual.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(spec.plank_length_m, spec.plank_width_m, spec.plank_thickness_m)
    )
    UsdGeom.Gprim(plank_visual.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(1.00, 0.92, 0.15)])  # bright yellow — seesaw plank
    UsdPhysics.CollisionAPI.Apply(plank_visual.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(plank_visual.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath(grip)),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )

    # IR line on the plank itself: a thin black stripe along the plank's local
    # +X axis, sitting just above the plank top surface. Because it's a child
    # of the plank Xform, it inherits the plank's tilt and seesaw rotation —
    # the line stays glued to the plank as it pivots.
    plank_line = UsdGeom.Cube.Define(stage, f"{plank_path}/ir_line")
    plank_line.CreateSizeAttr(1.0)
    UsdGeom.Xformable(plank_line.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, spec.plank_thickness_m / 2.0 + 0.0005)
    )
    UsdGeom.Xformable(plank_line.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(spec.plank_length_m * 0.98, spec.line_width_m, 0.001)
    )
    UsdGeom.Gprim(plank_line.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.05, 0.05)])

    # Seesaw entry bevel — tilted slab matching plank tilt that bridges
    # seesaw_ground (z=0) up to the plank's low-corner top surface. Slope
    # equals plank tilt so wheel transitions onto plank with NO step.
    # NOTE: tested PID controllers still cannot reliably traverse the
    # seesaw (integrator windup + single-axle roll instability stall the
    # cart at the bevel). Use the RL policy for the seesaw section.
    low_corner_world_x = seesaw_center_x - longer_side_len * math.cos(tilt_rad)
    plank_low_edge_top_z = spec.plank_thickness_m * math.cos(tilt_rad)
    if plank_low_edge_top_z > 1e-4 and tilt_rad > 1e-4:
        # CRITICAL: leave a small vertical gap between the bevel and the plank's
        # underside. Verified by CSV: without this gap (bevel right edge meets
        # plank low corner exactly), PhysX creates a permanent contact
        # constraint between the static bevel and the dynamic plank's
        # underside. The high-friction material (μ=1.6) then LOCKS the plank's
        # rotation — the joint allows rotation in principle but the bevel-plank
        # contact's static friction prevents the corner from sliding the small
        # amount needed for plank rotation around the offset joint. Result:
        # plank stays at -2.78° forever, no matter how much cart torque applied.
        # Dropping the bevel by 5 mm removes this contact and frees the joint.
        bevel_z_gap_m = 0.005  # vertical clearance between bevel top and plank
        entry_bevel_horizontal = plank_low_edge_top_z / math.tan(tilt_rad)
        entry_bevel_surface_len = plank_low_edge_top_z / math.sin(tilt_rad)
        entry_bevel_x_end = low_corner_world_x   # meets plank low corner
        entry_bevel_x_start = entry_bevel_x_end - entry_bevel_horizontal
        bevel_slab_half_thickness = spec.plate_thickness_m / 2.0
        top_mid_x = (entry_bevel_x_start + entry_bevel_x_end) / 2.0
        top_mid_z = plank_low_edge_top_z / 2.0 - bevel_z_gap_m  # sunken
        bevel_center_x = top_mid_x + bevel_slab_half_thickness * math.sin(tilt_rad)
        bevel_center_z = top_mid_z - bevel_slab_half_thickness * math.cos(tilt_rad)
        add_static_box(
            stage,
            "/CombinedCourse/seesaw_entry_bevel",
            translate=(bevel_center_x, 0.0, bevel_center_z),
            size_xyz=(entry_bevel_surface_len, spec.lane_width_y_m, spec.plate_thickness_m),
            rotate_xyz_deg=(0.0, -tilt_deg, 0.0),
            color=(1.00, 0.85, 0.10),  # bright yellow — seesaw entry bevel
            physics_material_path=grip,
        )

    # Y-axis revolute joint between seesaw_ground (static, body0) and the
    # plank (rigid, body1). The joint's body1-local anchor is OFFSET from
    # the plank's geometric center by +joint_offset_x_m along the plank's
    # long axis. That makes the cart-entry side LONGER and HEAVIER, so
    # gravity acting at the plank's geometric CoM produces a permanent
    # torque around the joint axis that tilts the plank toward cart-entry
    # until the long edge meets the ground.
    #
    # Why a joint instead of "free body + fulcrum contact":
    # - A free plank has 6 DOF. Only 1 (Y rotation) is wanted; the other
    #   5 (X/Z rotation + 3 translations) lead to PhysX contact-resolution
    #   instabilities that catapult the cart sideways when the plank flips.
    # - body0 is seesaw_ground (NOT the fulcrum cylinder) because static
    #   colliders work reliably as joint anchors; cylinder primitives in
    #   PhysX can be approximated as convex hulls which jitter contacts.
    # KINEMATIC anchor body for the seesaw joint. Verified by experiment:
    #   - World-anchored joint (no body0 OR body0=None): joint loads but
    #     ACTS LIKE A FIXED JOINT. Plank stays at initial -2.78° forever.
    #     CSV logging of plank_pitch_deg confirmed zero rotation across
    #     entire 30 s run despite 10.7 N·m of cart torque on exit side.
    #   - Static collider as body0 (e.g. seesaw_ground): PhysX misinterprets
    #     LocalPos0 (per earlier project notes).
    # Solution: explicit KINEMATIC rigid body at the joint world position,
    # used as body0. Kinematic = PhysX-tracked but immovable. This gives
    # PhysX a proper rigid anchor for the joint constraint while ensuring
    # the anchor itself never moves.
    anchor_path = "/CombinedCourse/seesaw_pivot_anchor"
    anchor_xform = UsdGeom.Xform.Define(stage, anchor_path)
    UsdGeom.Xformable(anchor_xform.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(seesaw_center_x, 0.0, spec.pivot_height_m)
    )
    anchor_rb = UsdPhysics.RigidBodyAPI.Apply(anchor_xform.GetPrim())
    anchor_rb.CreateRigidBodyEnabledAttr(True)
    anchor_rb.CreateKinematicEnabledAttr(True)

    joint_path = "/CombinedCourse/seesaw_joint"
    seesaw_joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    seesaw_joint.CreateAxisAttr("Y")
    seesaw_joint.CreateBody0Rel().SetTargets([anchor_path])
    seesaw_joint.CreateBody1Rel().SetTargets([plank_path])
    # body0 is the kinematic anchor at world (seesaw_center_x, 0, pivot_height).
    # Joint anchor is at the anchor's ORIGIN (LocalPos0 = (0,0,0) in anchor frame).
    seesaw_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    seesaw_joint.CreateLocalPos1Attr(
        Gf.Vec3f(spec.joint_offset_x_m, 0.0, -spec.plank_thickness_m / 2.0)
    )
    seesaw_joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    seesaw_joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    seesaw_joint.CreateLowerLimitAttr(-30.0)
    seesaw_joint.CreateUpperLimitAttr(30.0)
    # Explicit PhysxJointAPI: zero everything (no damping, no friction, no
    # armature) so the joint is truly free in its allowed DOF. Earlier tests
    # showed the joint allowed rotation when plank started at 0° (gravity tipped
    # it to -2.71°), but when plank started pre-tilted at -2.78° with a load on
    # the exit side, plank stayed at -2.78°. Likely PhysX defaults add some
    # damping/friction that resists rotation away from the rest pose.
    physx_joint = PhysxSchema.PhysxJointAPI.Apply(seesaw_joint.GetPrim())
    physx_joint.CreateJointFrictionAttr(0.0)
    physx_joint.CreateArmatureAttr(0.0)
    physx_joint.CreateMaxJointVelocityAttr(1000.0)
    course_segments.append(
        {
            "name": "seesaw",
            "x_range": [seesaw_x_start, seesaw_x_start + spec.seesaw_lane_len_m],
            "z_top": 0.0,
            "plank_center_x": seesaw_center_x,
            "pivot_height_m": spec.pivot_height_m,
            "initial_tilt_deg": tilt_deg,
        }
    )
    cursor_x = seesaw_x_start + spec.seesaw_lane_len_m

    # 9. Finish zone
    finish_x_start = cursor_x
    add_static_box(
        stage,
        "/CombinedCourse/finish_zone",
        translate=(
            finish_x_start + spec.finish_len_m / 2.0,
            0.0,
            -spec.plate_thickness_m / 2.0,
        ),
        size_xyz=(spec.finish_len_m, spec.lane_width_y_m, spec.plate_thickness_m),
        color=(0.75, 0.85, 0.65),
        physics_material_path=grip,
    )
    course_segments.append(
        {
            "name": "finish_zone",
            "x_range": [finish_x_start, finish_x_start + spec.finish_len_m],
            "z_top": 0.0,
        }
    )
    cursor_x = finish_x_start + spec.finish_len_m

    # IR line — draw segments along y=0 at the top surface z of each segment.
    # For ramp_up and ramp_down the line follows bevel + main slabs. Stairs
    # get one line per step top. Seesaw is split around the fulcrum (the
    # plank's own IR line is parented to the plank so it tilts with it).
    UsdGeom.Scope.Define(stage, "/CombinedCourse/Line")
    for i, seg in enumerate(course_segments):
        if seg["name"] == "seesaw":
            # Draw line on the seesaw's ground, on both sides of the fulcrum,
            # so it appears continuous through the section. The plank's own
            # tilt prevents drawing a continuous line that stays on its
            # surface, but the under-plank ground line keeps the visual cue.
            x_lo, x_hi = seg["x_range"]
            fulcrum_half = spec.fulcrum_base_xy_m[0] / 2.0
            mid = seg["plank_center_x"]
            left_x_lo = x_lo
            left_x_hi = mid - fulcrum_half
            right_x_lo = mid + fulcrum_half
            right_x_hi = x_hi
            if left_x_hi > left_x_lo:
                add_line_segment(
                    stage,
                    f"/CombinedCourse/Line/seg_{i}_seesaw_pre",
                    x_center=(left_x_lo + left_x_hi) / 2.0,
                    x_length=(left_x_hi - left_x_lo),
                    y=0.0,
                    z_top=0.0,
                    width=spec.line_width_m,
                )
            if right_x_hi > right_x_lo:
                add_line_segment(
                    stage,
                    f"/CombinedCourse/Line/seg_{i}_seesaw_post",
                    x_center=(right_x_lo + right_x_hi) / 2.0,
                    x_length=(right_x_hi - right_x_lo),
                    y=0.0,
                    z_top=0.0,
                    width=spec.line_width_m,
                )
            continue
        if seg["name"] == "stairs_down":
            for j, stair in enumerate(seg.get("stair_tops", [])):
                sx_lo, sx_hi = stair["x_range"]
                add_line_segment(
                    stage,
                    f"/CombinedCourse/Line/seg_{i}_stair_{j}",
                    x_center=(sx_lo + sx_hi) / 2.0,
                    x_length=(sx_hi - sx_lo),
                    y=0.0,
                    z_top=stair["z_top"],
                    width=spec.line_width_m,
                )
            continue
        x_lo, x_hi = seg["x_range"]
        if seg["name"] == "ramp_up":
            # Bevel + main, both tilted to follow the surface. Apex height comes
            # from the segment's z_top (the platform on top of the bump).
            bevel_h = seg["bevel_horizontal_m"]
            bevel_a = seg["bevel_angle_deg"]
            apex_z = seg["z_top"]
            bevel_rise_local = bevel_h * math.tan(math.radians(bevel_a))
            bevel_surface_len_local = bevel_h / math.cos(math.radians(bevel_a)) if bevel_h > 0 else 0.0
            if bevel_surface_len_local > 1e-6:
                add_line_segment(
                    stage,
                    f"/CombinedCourse/Line/seg_{i}_ramp_up_bevel",
                    x_center=x_lo + bevel_h / 2.0,
                    x_length=bevel_surface_len_local,
                    y=0.0,
                    z_top=bevel_rise_local / 2.0,
                    width=spec.line_width_m,
                    rotate_y_deg=-bevel_a,
                )
            main_horizontal = (x_hi - x_lo) - bevel_h
            if main_horizontal > 1e-6:
                main_rise = apex_z - bevel_rise_local
                main_surface_len = main_rise / math.sin(theta) if abs(theta) > 1e-6 else main_horizontal
                main_x_center = x_lo + bevel_h + main_horizontal / 2.0
                main_z_center = (bevel_rise_local + apex_z) / 2.0
                add_line_segment(
                    stage,
                    f"/CombinedCourse/Line/seg_{i}_ramp_up_main",
                    x_center=main_x_center,
                    x_length=main_surface_len,
                    y=0.0,
                    z_top=main_z_center,
                    width=spec.line_width_m,
                    rotate_y_deg=-spec.slope_deg,
                )
            continue
        if seg["name"] == "ramp_down":
            # Mirror of ramp_up: main first (descending), then bevel.
            bevel_h = seg["bevel_horizontal_m"]
            bevel_a = seg["bevel_angle_deg"]
            apex_z = seg["z_top_start"]
            main_horizontal = seg["main_horizontal_m"]
            bevel_rise_local = bevel_h * math.tan(math.radians(bevel_a))
            bevel_surface_len_local = bevel_h / math.cos(math.radians(bevel_a)) if bevel_h > 0 else 0.0
            if main_horizontal > 1e-6:
                main_rise = apex_z - bevel_rise_local
                main_surface_len = main_rise / math.sin(theta) if abs(theta) > 1e-6 else main_horizontal
                main_x_center = x_lo + main_horizontal / 2.0
                main_z_center = (apex_z + bevel_rise_local) / 2.0
                add_line_segment(
                    stage,
                    f"/CombinedCourse/Line/seg_{i}_ramp_down_main",
                    x_center=main_x_center,
                    x_length=main_surface_len,
                    y=0.0,
                    z_top=main_z_center,
                    width=spec.line_width_m,
                    rotate_y_deg=spec.slope_deg,
                )
            if bevel_surface_len_local > 1e-6:
                add_line_segment(
                    stage,
                    f"/CombinedCourse/Line/seg_{i}_ramp_down_bevel",
                    x_center=x_lo + main_horizontal + bevel_h / 2.0,
                    x_length=bevel_surface_len_local,
                    y=0.0,
                    z_top=bevel_rise_local / 2.0,
                    width=spec.line_width_m,
                    rotate_y_deg=bevel_a,
                )
            continue
        z_top = seg.get("z_top", 0.0)
        add_line_segment(
            stage,
            f"/CombinedCourse/Line/seg_{i}_{seg['name']}",
            x_center=(x_lo + x_hi) / 2.0,
            x_length=(x_hi - x_lo),
            y=0.0,
            z_top=z_top,
            width=spec.line_width_m,
        )

    stage.GetRootLayer().Save()
    return {
        "asset_path": str(output_path),
        "default_prim": "/CombinedCourse",
        "start_platform_height_m": start_platform_height,
        "ramp_apex_height_m": ramp_apex_height,
        "ramp_main_horizontal_m": main_ramp_horizontal,
        "ramp_bevel_horizontal_m": bevel_horizontal,
        "course_x_min_m": -spec.start_len_m,
        "course_x_max_m": cursor_x,
        "segments": course_segments,
        "spec": asdict(spec),
    }


def main() -> None:
    spec = CourseSpec()
    result = build_stage(args.output.resolve(), spec)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"CREATE_COMBINED_COURSE_OK asset={result['asset_path']} "
        f"x_range=[{result['course_x_min_m']:.2f}, {result['course_x_max_m']:.2f}] "
        f"start_platform_h={result['start_platform_height_m']:.3f} "
        f"apex_h={result['ramp_apex_height_m']:.3f}",
        flush=True,
    )


try:
    main()
finally:
    simulation_app.close()
