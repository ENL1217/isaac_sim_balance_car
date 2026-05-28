"""Render an annotated uncontrolled-fall test video and model photo.

Run with:
    D:\isaac\isaacsim\python.bat sim\scripts\render_uncontrolled_fall_media.py --headless
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render balance-car uncontrolled-fall media.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the GUI.")
    parser.add_argument(
        "--asset",
        type=Path,
        default=sim_dir / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda",
        help="Balance car USD asset path.",
    )
    parser.add_argument("--steps", type=int, default=360, help="Number of simulation steps to run.")
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate.")
    parser.add_argument(
        "--initial-pitch-deg",
        type=float,
        default=5.0,
        help="Initial forward/back pitch perturbation around the wheel axle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=sim_dir / "output" / "media",
        help="Directory for rendered media.",
    )
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdGeom


WIDTH = 1280
HEIGHT = 720


def load_font(size: int):
    for font_name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_L = load_font(34)
FONT_M = load_font(24)
FONT_S = load_font(18)


def world_transform(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"Prim does not exist: {prim_path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def up_tilt_deg(matrix) -> float:
    local_z_world = matrix.ExtractRotationMatrix().GetRow(2)
    z = max(-1.0, min(1.0, float(local_z_world[2])))
    return math.degrees(math.acos(abs(z)))


def signed_pitch_deg(matrix) -> float:
    rotation = matrix.ExtractRotationMatrix()
    local_z_world = rotation.GetRow(2)
    tilt = up_tilt_deg(matrix)
    direction = -1.0 if float(local_z_world[0]) < 0 else 1.0
    return direction * tilt


def draw_rotated_rect(draw: ImageDraw.ImageDraw, center, size, angle_deg, fill, outline=None, width=3):
    cx, cy = center
    w, h = size
    angle = math.radians(angle_deg)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    points = []
    for x, y in corners:
        rx = x * math.cos(angle) - y * math.sin(angle)
        ry = x * math.sin(angle) + y * math.cos(angle)
        points.append((cx + rx, cy + ry))
    draw.polygon(points, fill=fill)
    if outline:
        draw.line(points + [points[0]], fill=outline, width=width)


def draw_wheel(draw: ImageDraw.ImageDraw, center, radius, fill="#2f3237"):
    cx, cy = center
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(box, fill=fill, outline="#111827", width=5)
    draw.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), fill="#d1d5db", outline="#6b7280", width=2)


def draw_car_side(draw: ImageDraw.ImageDraw, center, pitch_deg, scale=1.0):
    cx, cy = center
    wheel_r = 50 * scale
    wheel_y = cy + 110 * scale
    wheel_dx = 96 * scale
    draw_wheel(draw, (cx - wheel_dx, wheel_y), wheel_r)
    draw_wheel(draw, (cx + wheel_dx, wheel_y), wheel_r)

    chassis_angle = -pitch_deg
    draw_rotated_rect(
        draw,
        (cx, cy),
        (270 * scale, 112 * scale),
        chassis_angle,
        fill="#e5e7eb",
        outline="#475569",
        width=max(2, int(4 * scale)),
    )
    draw_rotated_rect(
        draw,
        (cx, cy - 74 * scale),
        (210 * scale, 42 * scale),
        chassis_angle,
        fill="#93c5fd",
        outline="#1d4ed8",
        width=max(2, int(3 * scale)),
    )
    draw_rotated_rect(
        draw,
        (cx - 25 * scale, cy + 7 * scale),
        (108 * scale, 55 * scale),
        chassis_angle,
        fill="#86efac",
        outline="#166534",
        width=max(2, int(3 * scale)),
    )
    draw_rotated_rect(
        draw,
        (cx + 82 * scale, cy + 9 * scale),
        (72 * scale, 48 * scale),
        chassis_angle,
        fill="#fbbf24",
        outline="#92400e",
        width=max(2, int(3 * scale)),
    )
    draw.line((cx - wheel_dx, wheel_y, cx + wheel_dx, wheel_y), fill="#111827", width=max(2, int(4 * scale)))


def make_background() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(image)
    for y in range(0, HEIGHT):
        shade = int(248 - y * 16 / HEIGHT)
        draw.line((0, y, WIDTH, y), fill=(shade, shade + 3, min(255, shade + 7)))
    draw.rectangle((0, 560, WIDTH, HEIGHT), fill="#e2e8f0")
    draw.line((0, 560, WIDTH, 560), fill="#64748b", width=4)
    return image


def render_frame(sample, frame_index: int, total_frames: int) -> np.ndarray:
    image = make_background()
    draw = ImageDraw.Draw(image)

    pitch = sample["pitch_deg"]
    tilt = abs(pitch)
    x = 250 + 680 * frame_index / max(1, total_frames - 1)
    y = 368 + min(120, max(0, tilt - 10) * 1.5)

    draw_car_side(draw, (x, y), pitch, scale=1.0)
    draw.text((44, 38), "Two-Wheel Balance Car - Isaac Sim 5.0 Test", fill="#0f172a", font=FONT_L)
    draw.text((48, 86), "Uncontrolled fall: no PID / no balance controller", fill="#334155", font=FONT_M)
    draw.text((48, 130), f"time: {sample['time_s']:.2f} s", fill="#0f172a", font=FONT_M)
    draw.text((48, 164), f"tilt: {tilt:.1f} deg", fill="#0f172a", font=FONT_M)
    draw.text((48, 198), f"z: {sample['z_m']:.3f} m", fill="#0f172a", font=FONT_M)

    status = "FALLING / FALLEN" if tilt > 25 else "UPRIGHT BUT UNCONTROLLED"
    color = "#b91c1c" if tilt > 25 else "#166534"
    draw.rounded_rectangle((48, 244, 360, 294), radius=8, fill=color)
    draw.text((68, 257), status, fill="#ffffff", font=FONT_M)

    draw.text(
        (48, 640),
        "This is an annotated test render from the primitive USD model, not a real-car camera photo.",
        fill="#475569",
        font=FONT_S,
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render_photo(output_path: Path) -> None:
    image = make_background()
    draw = ImageDraw.Draw(image)

    def wheel(cx, cy):
        draw.ellipse((cx - 54, cy - 92, cx + 54, cy + 92), fill="#15171c", outline="#050608", width=5)
        draw.ellipse((cx - 35, cy - 63, cx + 35, cy + 63), fill="#0b3f9c", outline="#08235f", width=4)
        for angle in range(0, 180, 24):
            length = 54
            rad = math.radians(angle)
            dx = math.cos(rad) * length * 0.54
            dy = math.sin(rad) * length * 0.88
            draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill="#1d64d8", width=4)
        draw.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), fill="#d1d5db", outline="#475569", width=2)

    def plate(points, offset=18):
        side = [(x, y + offset) for x, y in points]
        draw.polygon([points[0], points[1], side[1], side[0]], fill="#d9d8d0")
        draw.polygon([points[1], points[2], side[2], side[1]], fill="#c7c6bd")
        draw.polygon([points[2], points[3], side[3], side[2]], fill="#d2d1c9")
        draw.polygon(points, fill="#f4f1e7", outline="#64748b")
        draw.line(points + [points[0]], fill="#475569", width=3)
        for x, y in points:
            draw.ellipse((x - 6, y - 4, x + 6, y + 8), fill="#e5e7eb", outline="#94a3b8", width=2)

    lower = [(365, 326), (915, 326), (1010, 424), (460, 424)]
    upper = [(338, 192), (888, 192), (982, 292), (432, 292)]
    board = [(500, 208), (760, 208), (814, 260), (554, 260)]

    left_wheel_center = (360, 452)
    right_wheel_center = (920, 452)
    draw.line(
        (left_wheel_center[0], left_wheel_center[1], right_wheel_center[0], right_wheel_center[1]),
        fill="#475569",
        width=8,
    )
    wheel(*left_wheel_center)
    wheel(*right_wheel_center)
    plate(lower)
    for top, bottom in zip(upper, lower):
        tx, ty = top
        bx, by = bottom
        draw.line((tx, ty + 12, bx, by - 2), fill="#8a6b16", width=8)
        draw.line((tx + 2, ty + 12, bx + 2, by - 2), fill="#d5b13d", width=3)
    plate(upper)
    draw.polygon(board, fill="#0f4f9e", outline="#082f62")
    draw.line(board + [board[0]], fill="#0b2f63", width=3)

    component_specs = [
        ((602, 222), (666, 222), (684, 239), (620, 239), "#1f2937"),
        ((704, 224), (752, 224), (770, 244), (722, 244), "#111827"),
        ((775, 238), (830, 238), (858, 265), (802, 265), "#cbd5e1"),
        ((520, 220), (568, 220), (584, 236), (536, 236), "#111827"),
        ((548, 246), (578, 246), (588, 256), (558, 256), "#dc2626"),
    ]
    for poly in component_specs:
        *pts, color = poly
        draw.polygon(list(pts), fill=color, outline="#0f172a")

    for x, y in [(500, 282), (600, 284), (720, 284), (842, 282)]:
        draw.ellipse((x - 5, y - 4, x + 5, y + 6), fill="#0f172a")

    draw.text((44, 38), "Yahboom-Style Balance-Car Model", fill="#0f172a", font=FONT_L)
    draw.text((48, 86), "Corrected wheel layout: both wheels share one transverse axle under the chassis", fill="#334155", font=FONT_M)
    draw.text((48, 640), "Model reference image from regenerated USD primitives; not an official CAD export.", fill="#475569", font=FONT_S)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    if not args.asset.exists():
        raise FileNotFoundError(f"Balance car asset does not exist: {args.asset}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "uncontrolled_fall_test.mp4"
    photo_path = args.output_dir / "balance_car_model_photo.png"
    report_path = args.output_dir / "uncontrolled_fall_media_result.json"

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=str(args.asset.resolve()), prim_path="/World/BalanceCar")
    if args.initial_pitch_deg:
        root_prim = world.stage.GetPrimAtPath("/World/BalanceCar")
        UsdGeom.Xformable(root_prim).AddRotateYOp().Set(args.initial_pitch_deg)
    world.scene.add(Articulation(prim_paths_expr="/World/BalanceCar/base_link", name="balance_car"))
    world.reset()

    samples = []
    base_path = "/World/BalanceCar/base_link"
    for step in range(args.steps + 1):
        matrix = world_transform(world.stage, base_path)
        pos = matrix.ExtractTranslation()
        samples.append(
            {
                "step": step,
                "time_s": step / 120.0,
                "pitch_deg": signed_pitch_deg(matrix),
                "tilt_deg": up_tilt_deg(matrix),
                "z_m": float(pos[2]),
            }
        )
        if step < args.steps:
            world.step(render=not args.headless)

    render_photo(photo_path)

    video = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (WIDTH, HEIGHT),
    )
    if not video.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")

    frame_count = int(args.fps * (args.steps / 120.0))
    frame_count = max(frame_count, 1)
    for frame_index in range(frame_count):
        sample_index = round(frame_index * (len(samples) - 1) / max(1, frame_count - 1))
        video.write(render_frame(samples[sample_index], frame_index, frame_count))
    video.release()

    max_tilt = max(sample["tilt_deg"] for sample in samples)
    end_tilt = samples[-1]["tilt_deg"]
    report = {
        "ok": True,
        "asset_path": str(args.asset.resolve()),
        "video_path": str(video_path.resolve()),
        "photo_path": str(photo_path.resolve()),
        "steps": args.steps,
        "fps": args.fps,
        "initial_pitch_deg": args.initial_pitch_deg,
        "end_tilt_deg": end_tilt,
        "max_tilt_deg": max_tilt,
        "fell_without_controller": end_tilt > 25.0 or max_tilt > 35.0,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"MEDIA_RENDER_OK video={video_path} photo={photo_path} "
        f"end_tilt={end_tilt:.2f} max_tilt={max_tilt:.2f}",
        flush=True,
    )


try:
    main()
finally:
    simulation_app.close()
