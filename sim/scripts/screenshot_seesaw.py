"""Render the combined course's seesaw section to a PNG so we can verify
the geometry actually looks like a seesaw. Saves the screenshot to
sim/output/seesaw_view.png.

Run with:
    D:/isaac/isaacsim/python.bat sim/scripts/screenshot_seesaw.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from isaacsim import SimulationApp

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_PATH = _PROJECT_ROOT / "assets" / "combined_course" / "combined_course.usda"
OUTPUT_PNG = _PROJECT_ROOT / "output" / "seesaw_view.png"

simulation_app = SimulationApp({
    "headless": False,  # required for rendering
    "width": 1280,
    "height": 720,
})

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import UsdLux, UsdGeom  # noqa: E402
import omni.kit.viewport.utility  # noqa: E402

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

stage = world.stage
# Lighting
distant = UsdLux.DistantLight.Define(stage, "/World/Sun")
distant.CreateIntensityAttr(3500.0)
distant.CreateAngleAttr(1.5)
UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set((-45.0, 0.0, 30.0))
dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
dome.CreateIntensityAttr(900.0)

add_reference_to_stage(usd_path=str(ASSET_PATH), prim_path="/World/Course")

world.reset()
# Settle physics long enough that gravity tilts the plank via CoM offset.
# At ~60 Hz physics, 240 steps = 4 sec. Plenty of time for ~3° tilt.
for _ in range(240):
    world.step(render=True)

# Look at the seesaw from a slight angle so we see the tilt clearly.
# Seesaw is at world x ≈ 5–8.7 with the new geometry.
# Camera positioned roughly side-on, slightly above.
SEESAW_CENTER_X = 8.56  # with seesaw_lane_len=7.0 + offset joint
# Pure side view (camera along -Y axis) so we see the plank's tilt
# profile clearly. Camera at y=-2.5, looking back at +Y direction.
set_camera_view(
    eye=(SEESAW_CENTER_X, -2.5, 0.30),
    target=(SEESAW_CENTER_X, 0.0, 0.08),
)

# Render a few extra frames so the camera position takes effect
for _ in range(10):
    world.step(render=True)

# Inspect plank's actual world transform AFTER physics settle so we know
# whether PhysX snapped it horizontal or kept the initial tilt.
from pxr import Usd  # noqa: E402
plank_prim = stage.GetPrimAtPath("/World/Course/seesaw_plank")
xformable = UsdGeom.Xformable(plank_prim)
m = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
print(f"PLANK_WORLD_MATRIX:", flush=True)
for row in range(4):
    print(f"  [{m[row][0]:+.4f} {m[row][1]:+.4f} {m[row][2]:+.4f} {m[row][3]:+.4f}]", flush=True)
# Extract rotation. USD uses row-vector convention, so for RotateY(θ) the
# matrix row 0 is [cos θ, 0, -sin θ, 0]. That means matrix[0][2] = -sin(θ),
# i.e. the rotation angle is the NEGATIVE asin of m[0][2].
# In our scene the plank's design rotation is -tilt_deg around Y so the
# cart-entry side (-X local) sits on the ground. A correctly settled plank
# reports actual_rotation_deg ≈ -3.06°.
import math as _math
sin_neg_tilt = m[0][2]  # = -sin(θ) for row-vector RotateY(θ)
actual_rotation_deg = -_math.degrees(_math.asin(max(-1.0, min(1.0, sin_neg_tilt))))
print(
    f"PLANK_ACTUAL_ROTATION_Y_DEG={actual_rotation_deg:.3f}  "
    f"(negative = cart-entry side LOW; expected approx -1.15)",
    flush=True,
)

# Capture the active viewport via Kit's screen capture.
viewport_api = omni.kit.viewport.utility.get_active_viewport()
OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
omni.kit.viewport.utility.capture_viewport_to_file(viewport_api, str(OUTPUT_PNG))

# Render a few more steps so the capture has time to flush to disk
for _ in range(30):
    world.step(render=True)

print(f"SCREENSHOT_SAVED path={OUTPUT_PNG} exists={OUTPUT_PNG.exists()}", flush=True)
simulation_app.close()
