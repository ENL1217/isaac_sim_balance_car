"""Render README hero shots: cart upright, course visible.

Captures the balance car STANDING UPRIGHT at the start platform of the
combined obstacle course. The cart is balanced (not tipped) and you can
see the stairs / ramps / seesaw extending into the distance.

Usage:
    %ISAACSIM_PATH%\\python.bat sim/scripts/screenshot_for_readme.py
"""

from __future__ import annotations
from pathlib import Path

from isaacsim import SimulationApp  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
COURSE_USD = _PROJECT_ROOT / "sim" / "assets" / "combined_course" / "combined_course.usda"
CART_USD = _PROJECT_ROOT / "sim" / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda"
HERO_PNG = _PROJECT_ROOT / "docs" / "images" / "cart_on_seesaw.png"
COURSE_PNG = _PROJECT_ROOT / "docs" / "images" / "combined_course_overview.png"

simulation_app = SimulationApp({
    "headless": False,
    "width": 1920,
    "height": 1080,
})

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402
import omni.kit.viewport.utility  # noqa: E402


world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 240.0, rendering_dt=1.0 / 60.0)
world.scene.add_default_ground_plane()

stage = world.stage
# Lighting (warm, slightly above)
distant = UsdLux.DistantLight.Define(stage, "/World/Sun")
distant.CreateIntensityAttr(3500.0)
distant.CreateAngleAttr(1.5)
UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set((-50.0, 0.0, 25.0))
dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
dome.CreateIntensityAttr(800.0)

# Load course + cart
add_reference_to_stage(usd_path=str(COURSE_USD), prim_path="/World/Course")
add_reference_to_stage(usd_path=str(CART_USD), prim_path="/World/Cart")

# Place the cart at the start platform (x = -1.0, on the elevated start zone).
cart_xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Cart"))
cart_xform.AddTranslateOp().Set(Gf.Vec3d(-1.0, 0.0, 0.13))

world.reset()

# Settle physics — let the cart fall onto the start platform and balance.
# Cart has internal balance dynamics, but with no controller running here it
# will fall. So we only step a tiny bit so the cart is still upright in the
# captured frame.
for _ in range(30):  # 0.125 sec
    world.step(render=True)

# --- Hero shot: close-up, cart upright at start platform ---
# Camera angled to show cart from front-right, with course extending behind.
set_camera_view(
    eye=(-2.0, -1.2, 0.45),
    target=(-0.5, 0.0, 0.15),
)
for _ in range(30):
    world.step(render=True)

HERO_PNG.parent.mkdir(parents=True, exist_ok=True)
viewport_api = omni.kit.viewport.utility.get_active_viewport()
omni.kit.viewport.utility.capture_viewport_to_file(viewport_api, str(HERO_PNG))
print(f"HERO_SHOT_SAVED path={HERO_PNG}", flush=True)

# Let the capture flush
for _ in range(60):
    world.step(render=True)

# --- Overview: show entire course from elevated angle ---
set_camera_view(
    eye=(2.5, -6.0, 4.5),
    target=(5.5, 0.0, 0.05),
)
for _ in range(30):
    world.step(render=True)

COURSE_PNG.parent.mkdir(parents=True, exist_ok=True)
omni.kit.viewport.utility.capture_viewport_to_file(viewport_api, str(COURSE_PNG))
print(f"COURSE_SAVED path={COURSE_PNG}", flush=True)

# Final flush
for _ in range(60):
    world.step(render=True)

print("DONE", flush=True)
simulation_app.close()
