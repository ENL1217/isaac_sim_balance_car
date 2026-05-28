"""Run the two-wheel balance car with pure physics effort PID control.

This script does not use pose assist. It reads the simulated body pose and wheel
joint states, then applies left/right wheel efforts.

Run:
    D:\isaac\isaacsim\python.bat sim\scripts\play_pid_effort.py --headless
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    sim_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Pure physics PID effort controller for the two-wheel balance car.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument(
        "--asset",
        type=Path,
        default=sim_dir / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda",
        help="Balance car USD asset path.",
    )
    parser.add_argument("--steps", type=int, default=1200, help="Number of physics steps.")
    parser.add_argument("--physics-hz", type=float, default=240.0, help="Physics/control update rate.")
    parser.add_argument("--initial-pitch-deg", type=float, default=1.0, help="Initial pitch disturbance.")
    parser.add_argument("--command-profile", choices=("stand", "drive_demo", "climb", "descend", "teleop", "stress_teleop"), default="stand")
    parser.add_argument(
        "--teleop-velocity-m-s",
        type=float,
        default=0.20,
        help="Target forward velocity (m/s) sent to LQR/LQI when W is held in teleop mode.",
    )
    parser.add_argument(
        "--teleop-yaw-rate-rad-s",
        type=float,
        default=0.6,
        help="Target yaw rate (rad/s) sent to LQR/LQI when A/D are held in teleop mode.",
    )
    parser.add_argument(
        "--follow-camera",
        action="store_true",
        help="GUI only: each tick, reposition the viewport camera to track the cart from behind.",
    )
    parser.add_argument(
        "--slope-feedforward",
        action="store_true",
        help="Estimate slope angle from filtered pitch and add a gravity-compensation torque to the controller output. Required to climb 30 deg ramps with PID/LQR.",
    )
    parser.add_argument(
        "--adaptive-slope-boost",
        action="store_true",
        help="Automatically increase drive-pitch-offset-deg and teleop-velocity-m-s while the cart is on a detected slope. Smoothly interpolates from baseline values on flat ground to --boost-* values at --boost-slope-full-deg.",
    )
    parser.add_argument(
        "--boost-pitch-offset-deg",
        type=float,
        default=5.0,
        help="Target drive-pitch-offset-deg when the detected slope reaches --boost-slope-full-deg.",
    )
    parser.add_argument(
        "--boost-velocity-m-s",
        type=float,
        default=0.5,
        help="Target teleop-velocity-m-s when the detected slope reaches --boost-slope-full-deg.",
    )
    parser.add_argument(
        "--boost-slope-threshold-deg",
        type=float,
        default=1.0,
        help="Slope magnitude (deg) below which no boost is applied (cart treated as on flat ground).",
    )
    parser.add_argument(
        "--boost-slope-full-deg",
        type=float,
        default=3.0,
        help="Slope magnitude (deg) at and above which the full boost is applied.",
    )
    parser.add_argument(
        "--slope-est-alpha",
        type=float,
        default=0.005,
        help="Low-pass filter coefficient for slope angle estimation (radians per tick).",
    )
    parser.add_argument(
        "--slope-ff-total-mass-kg",
        type=float,
        default=0.94,
        help="Total mass used for slope feed-forward computation (chassis + wheels).",
    )
    parser.add_argument("--controller", choices=("yahboom", "legacy", "lqr", "lqi", "threering"), default="yahboom")
    # ----- threering reference controller (W13 教材 v2 三環並聯) -----
    parser.add_argument("--threering-kp-balance", type=float, default=250.0,
                        help="Balance PD: Kp on pitch (deg). Spec default 250.")
    parser.add_argument("--threering-kd-balance", type=float, default=5.0,
                        help="Balance PD: Kd on gyro_pitch (deg/s). Spec default 5.")
    parser.add_argument("--threering-middle-angle-deg", type=float, default=0.0,
                        help="Mechanical zero offset for pitch.")
    parser.add_argument("--threering-kp-velocity", type=float, default=0.7,
                        help="Velocity PI: Kp on filtered speed error (counts/period).")
    parser.add_argument("--threering-ki-velocity", type=float, default=0.01,
                        help="Velocity PI: Ki on position error (integrated counts).")
    parser.add_argument("--threering-lpf-alpha", type=float, default=0.86,
                        help="Velocity loop LPF: filt = alpha*filt + (1-alpha)*raw. Spec 0.86.")
    parser.add_argument("--threering-position-clamp", type=float, default=10000.0,
                        help="Anti-windup clamp on velocity-loop position integral.")
    parser.add_argument("--threering-target-speed-pulses", type=float, default=16.0,
                        help="Encoder counts/period when forward/back is held. Spec 0.5 m/s = 16.")
    parser.add_argument("--threering-kp-turn", type=float, default=0.5,
                        help="Turn PD: Kp on target_yaw_rate (deg/s).")
    parser.add_argument("--threering-kd-turn", type=float, default=0.3,
                        help="Turn PD: Kd on gyro_yaw (deg/s). Active only when moving.")
    parser.add_argument("--threering-target-yaw-rate-deg-s", type=float, default=50.0,
                        help="Target yaw rate magnitude when A/D held. Spec default 50 deg/s.")
    parser.add_argument("--threering-pwm-limit", type=float, default=3000.0,
                        help="Spec-side PWM saturation. Gains are tuned for STM32-style raw PWM units.")
    parser.add_argument("--threering-antiwindup-decay", type=float, default=0.9,
                        help="When stalled (wheels not really moving and no command), shrink position_pulses by this factor each speed-loop tick. Set to 1.0 to disable (matches the spec literally; works on real Yahboom thanks to stopflag, but not on our sim).")
    parser.add_argument("--threering-stall-threshold", type=float, default=10.0,
                        help="|speed_filter| below this counts as 'stalled' for the anti-windup decay above.")
    parser.add_argument("--lqi-q-integral", type=float, default=20.0, help="LQI Q weight on the velocity-error integral state.")
    parser.add_argument("--lqi-integral-clamp", type=float, default=2.0, help="Anti-windup clamp on the LQI integral state (m).")
    parser.add_argument("--lqr-q-pitch", type=float, default=5000.0, help="LQR Q weight on pitch (rad).")
    parser.add_argument("--lqr-q-pitch-rate", type=float, default=100.0, help="LQR Q weight on pitch rate (rad/s).")
    parser.add_argument("--lqr-q-pos", type=float, default=1.0, help="LQR Q weight on wheel position error (m).")
    parser.add_argument("--lqr-q-vel", type=float, default=0.5, help="LQR Q weight on wheel velocity error (m/s).")
    parser.add_argument("--lqr-r-torque", type=float, default=0.1, help="LQR R weight on wheel torque (N*m).")
    parser.add_argument("--lqr-yaw-kp", type=float, default=0.02, help="Yaw-rate proportional gain for LQR mode (N*m per rad/s).")
    parser.add_argument("--lqr-yaw-rate-cmd-scale", type=float, default=2.5, help="Scale from drive_demo target_yaw_rate to commanded yaw rate (rad/s).")
    parser.add_argument("--lqr-target-velocity-scale", type=float, default=1.0, help="Scale from command target velocity to LQR target wheel velocity (m/s).")
    parser.add_argument("--lqr-chassis-mass", type=float, default=0.78, help="Chassis mass for LQR linearization (kg).")
    parser.add_argument("--lqr-wheel-mass-total", type=float, default=0.16, help="Total wheel mass (both wheels) for LQR linearization (kg).")
    parser.add_argument("--lqr-com-height", type=float, default=0.105, help="Chassis COM height above axle (m).")
    parser.add_argument("--lqr-chassis-pitch-inertia", type=float, default=0.0115, help="Chassis pitch moment of inertia about COM (kg*m^2).")
    parser.add_argument("--lqr-wheel-inertia-total", type=float, default=0.0001, help="Total wheel rotational inertia about axle (kg*m^2).")
    parser.add_argument("--effort-limit", type=float, default=1.2, help="Wheel torque/effort saturation (peak motor torque, N·m).")
    parser.add_argument(
        "--wheel-damping-coef",
        type=float,
        default=0.04,
        help="Velocity-dependent damping torque applied to each wheel: "
             "tau_damp = -coef * wheel_omega (rad/s). Models the back-EMF of "
             "a real DC motor that limits max wheel speed. With coef=0.04 and "
             "effort=1.2, max wheel ω ≈ 30 rad/s ≈ 1 m/s cart speed. Without "
             "this, sim motors can accelerate wheels infinitely fast, which "
             "amplifies turn-differentials into roll instability that doesn't "
             "happen on the real hardware.",
    )
    parser.add_argument(
        "--no-encoder",
        action="store_true",
        help="Simulate a car variant with no wheel encoders (only MPU6050). Controllers see pitch/gyro only; wheel position/velocity feedback is zeroed.",
    )
    parser.add_argument(
        "--line-follow",
        action="store_true",
        help="Enable line-following yaw correction (post-process layered on top of the controller).",
    )
    parser.add_argument(
        "--line-follow-y",
        type=float,
        default=0.0,
        help="World-frame y coordinate of the line (matches line_world.usda default of 0).",
    )
    parser.add_argument(
        "--line-follow-kp",
        type=float,
        default=2.5,
        help="P gain from lateral error (m) to differential wheel torque (N*m).",
    )
    parser.add_argument(
        "--line-follow-kd",
        type=float,
        default=0.15,
        help="D gain from yaw_rate (rad/s) to differential wheel torque (N*m).",
    )
    parser.add_argument("--effort-sign", type=float, default=1.0, help="Global effort sign.")
    parser.add_argument("--balance-kp", type=float, default=7.0, help="Upright loop Kp.")
    parser.add_argument("--balance-kd", type=float, default=0.45, help="Upright loop Kd.")
    parser.add_argument("--speed-kp", type=float, default=0.30, help="Wheel speed to target pitch gain.")
    parser.add_argument("--position-kp", type=float, default=0.12, help="Wheel position to target pitch gain.")
    parser.add_argument("--target-pitch-limit-deg", type=float, default=12.0, help="Outer-loop pitch target limit.")
    parser.add_argument("--yaw-kp", type=float, default=0.02, help="Yaw-rate effort gain.")
    parser.add_argument("--loop-period-ms", type=float, default=5.0, help="Yahboom-like interrupt loop period.")
    parser.add_argument("--speed-loop-divisor", type=int, default=8, help="Run speed PI every N interrupt loops.")
    parser.add_argument("--turn-loop-divisor", type=int, default=8, help="Run turn PD every N interrupt loops.")
    parser.add_argument("--encoder-cpr", type=float, default=780.0, help="Simulated Hall encoder counts per wheel revolution.")
    parser.add_argument("--angle-kp", type=float, default=38.0, help="Angle-loop P. Yahboom reference uses 191.3 for a ~0.05 N·m real motor; our 1.2 N·m sim motor needs ~5× less.")
    parser.add_argument("--angle-kd", type=float, default=0.58, help="Angle-loop D on gyro (deg/s). Yahboom 0.21 × 16.4 LSB/deg = 3.44, scaled down by motor strength.")
    parser.add_argument("--angle-balance-offset-deg", type=float, default=0.0, help="Mechanical balance offset.")
    parser.add_argument(
        "--drive-pitch-offset-deg",
        type=float,
        default=1.2,
        help="When forward/back command is active, add this magnitude (with sign) to the angle-loop setpoint so the cart actively leans into the drive direction. Set to 0 to recover the legacy vendor mapping.",
    )
    parser.add_argument(
        "--drive-offset-ramp-tau-s",
        type=float,
        default=0.20,
        help="Low-pass time-constant (seconds) for the drive offset PRESS transition "
             "(0 → target). Faster ramp = more responsive but bigger initial shove.",
    )
    parser.add_argument(
        "--drive-offset-release-tau-s",
        type=float,
        default=0.3,
        help="Low-pass time-constant (seconds) for the drive offset RELEASE transition "
             "(target → 0). Was 1.5 s in C0-C4 to give the cart momentum time to dissipate "
             "before the PID stops leaning. With C5's joint friction + PWM dead-band, the "
             "cart no longer hunts, and the long release just causes brief-tap-then-tip "
             "oscillation (cart over-corrects backward when lean stays after speed dies). "
             "0.3 s matches the press ramp's responsiveness without the over-corrected tip.",
    )
    parser.add_argument(
        "--drive-offset-turn-brake-tau-s",
        type=float,
        default=0.15,
        help="Fast time-constant for collapsing drive_offset to 0 when the user presses "
             "ONLY a turn key (A/D) without a forward/back key. The single-axle chassis "
             "cannot safely turn while moving forward (centripetal roll-tip), so we "
             "force a quick brake before the turn-differential applies.",
    )
    parser.add_argument("--speed-p", type=float, default=0.8, help="Speed-loop velocity-error P (sim-tuned; Yahboom STM32 ref uses 7.0 for ~0.05 N·m motors).")
    parser.add_argument("--speed-i", type=float, default=0.02, help="Speed-loop position-error I (sim-tuned; Yahboom STM32 ref uses 0.1 for smaller motors).")
    parser.add_argument(
        "--bluetooth-speed-magnitude",
        type=float,
        default=100.0,
        help="Yahboom BST_fBluetoothSpeed magnitude. Added to position integrator each "
             "speed-loop tick when forward/back command active. STM32 uses 800 for "
             "normal drive; ours scaled to 100 for the stronger sim motor (otherwise "
             "the cart accelerates past control authority and tips forward).",
    )
    parser.add_argument(
        "--bluetooth-direction-magnitude",
        type=float,
        default=30.0,
        help="Yahboom BST_fBluetoothDirectionNew magnitude. Direct PWM differential "
             "applied to wheels when turn command active. STM32 uses 300 for normal "
             "turn; ours scaled to 60 for sim. NO PID, NO ramp, NO accumulator.",
    )
    parser.add_argument(
        "--speed-pulses-clamp",
        type=float,
        default=3000.0,
        help="Symmetric clamp on the yahboom speed-loop position_pulses integrator. "
             "Lower values reduce integrator wind-up (helps the cart push through "
             "high-friction terrain without the integrator cancelling the angle loop).",
    )
    parser.add_argument(
        "--speed-antiwindup-decay",
        type=float,
        default=1.0,
        help="When >0 and <1, multiply position_pulses by this factor each speed-loop tick "
             "while the cart is stalled (|speed_filter|<threshold but forward/back command held). "
             "1.0 = no anti-windup, 0.9 = aggressive decay. Helps PID push through ramps.",
    )
    parser.add_argument(
        "--speed-stall-threshold",
        type=float,
        default=0.5,
        help="Pulses-per-tick magnitude below which the wheel is considered stalled "
             "for the purpose of anti-windup decay. 0.5 ≈ ~4 mm wheel motion per "
             "speed-loop tick — catches creeping but not target-speed travel.",
    )
    parser.add_argument("--turn-p", type=float, default=1.0, help="Yahboom-style turn-loop proportional gain in simulated pulse units.")
    parser.add_argument(
        "--turn-speed-softscale",
        type=float,
        default=0.30,
        help="(Deprecated; superseded by --turn-speed-gate-m-s.) Kept for backward "
             "compatibility with legacy invocations.",
    )
    parser.add_argument(
        "--turn-speed-gate-m-s",
        type=float,
        default=0.08,
        help="Speed threshold (m/s) above which the turn differential is BLOCKED. "
             "The single-axle chassis cannot safely turn while moving forward — "
             "centripetal force tips the cart sideways. Below this threshold the "
             "full ±bluetooth_direction_magnitude is applied (clean in-place pivot); "
             "above it the turn is suppressed until the user releases W or the cart "
             "slows down. 0.08 m/s is just above sensor jitter, so pivoting always works.",
    )
    parser.add_argument("--turn-d", type=float, default=0.02, help="Yahboom-style turn-loop gyro gain in simulated pulse units.")
    parser.add_argument("--command-pulse-step", type=float, default=1.0, help="Yahboom-style forward/back position increment per speed loop in simulated pulse units.")
    parser.add_argument("--pwm-limit", type=float, default=255.0, help="PWM saturation. 255 keeps existing tuning; Yahboom hardware uses 3000.")
    parser.add_argument(
        "--pwm-dead-band", type=float, default=8.0,
        help="Final PWM command magnitudes below this are squashed to 0. "
             "Models the TB6612FNG H-bridge turn-on threshold and bearing "
             "stiction. ~3 percent of pwm-limit. Set to 0 to disable (recovers "
             "old hunting behaviour).",
    )
    # ---- Sensor noise (high-fidelity sim2real) ----
    parser.add_argument(
        "--imu-pitch-noise-deg", type=float, default=0.0,
        help="Std-dev (deg) of Gaussian noise added to pitch reading per tick. "
             "Models MPU6050 accel-derived pitch noise + tilt-angle quantization. "
             "Typical real value 0.05–0.1 deg. Default 0 (clean sensor).",
    )
    parser.add_argument(
        "--imu-gyro-noise-deg-s", type=float, default=0.0,
        help="Std-dev (deg/s) of Gaussian noise added to pitch_rate and yaw_rate. "
             "Models MPU6050 gyro chip noise. Typical real value 0.3–0.6 deg/s.",
    )
    parser.add_argument(
        "--imu-gyro-bias-deg-s", type=float, default=0.0,
        help="Constant gyro bias offset (deg/s) for this run. Sampled once at "
             "startup from N(0, bias_sigma) where bias_sigma is this value. "
             "Models MPU6050 turn-on bias. Typical real after calibration ~0.05 deg/s.",
    )
    parser.add_argument(
        "--turn-ramp-tau-s", type=float, default=0.2,
        help="First-order low-pass time-constant (s) for the turn PWM differential "
             "in the yahboom controller. Without ramp the differential jumps 0 -> "
             "±magnitude in one tick, hitting the single-axle chassis with a lateral "
             "impulse it can't recover from. 0.2 s lets the cart's pitch loop catch "
             "up to the lateral disturbance. Set to 0 for instant (legacy) behaviour. "
             "Real STM32 reference (B570/control.c:Turn()) has NO ramp — it relies "
             "on motor rotor inertia (armature) and Kd*gyro damping (see "
             "--turn-kd-pwm-per-dps) to keep yaw from over-shooting.",
    )
    parser.add_argument(
        "--turn-kd-pwm-per-dps", type=float, default=0.0,
        help="Real STM32 Turn() Kd damping: subtracts `Kd*yaw_rate_dps` from the "
             "turn PWM differential, but ONLY while forward/back is held (matches "
             "the gain-scheduled `Kd = Turn_Kd if Flag_front||Flag_back else 0` in "
             "B570/control.c line 168). Real default Turn_Kd=60 (/100=0.6) applied "
             "to MPU6050 raw counts (16.4 LSB/dps at ±2000dps full scale) in "
             "real PWM range 6900. Equivalent in our PWM range 255: "
             "0.6 * 16.4 * 255/6900 ≈ 0.364 PWM/dps. Default 0.0 (off / no damping) "
             "for backward compatibility. Set to 0.364 to match real STM32 behaviour.",
    )
    parser.add_argument(
        "--turn-disable-speed-taper", action="store_true",
        help="Disable the sim-only `turn_scale = max(0.3, 1 - 0.5*v_ratio)` taper "
             "that reduces turn magnitude with forward speed. Real STM32 keeps full "
             "turn proportional authority at all speeds (the only speed-dependent "
             "term is the Kd*gyro damping that activates when moving). When you set "
             "--turn-kd-pwm-per-dps to a non-zero value, you typically also want to "
             "disable this taper so the two stability mechanisms don't double-count.",
    )
    parser.add_argument(
        "--encoder-quantize", action="store_true",
        help="Round joint-position deltas to integer encoder counts before the "
             "controller sees them. Models Hall encoder discrete quantization. "
             "Off by default; turn on for high-fidelity controller robustness tests.",
    )
    parser.add_argument("--fall-angle-deg", type=float, default=30.0, help="Cut motor and report fall beyond this pitch.")
    parser.add_argument(
        "--world",
        choices=("flat", "slope", "seesaw", "stairs", "gravel", "line", "combined"),
        default="flat",
        help="Scene to load. flat: default ground plane. slope/seesaw/stairs/gravel/line/combined: load the corresponding world USD. combined chains every obstacle into a single course suitable for RL.",
    )
    parser.add_argument(
        "--slope-world-asset",
        type=Path,
        default=sim_dir / "assets" / "slope_world" / "slope_world.usda",
        help="Slope-world USD asset path (used when --world slope).",
    )
    parser.add_argument(
        "--seesaw-world-asset",
        type=Path,
        default=sim_dir / "assets" / "seesaw_world" / "seesaw_world.usda",
        help="Seesaw-world USD asset path (used when --world seesaw).",
    )
    parser.add_argument(
        "--stairs-world-asset",
        type=Path,
        default=sim_dir / "assets" / "stairs_world" / "stairs_world.usda",
        help="Stairs-world USD asset path (used when --world stairs).",
    )
    parser.add_argument(
        "--gravel-world-asset",
        type=Path,
        default=sim_dir / "assets" / "gravel_world" / "gravel_world.usda",
        help="Gravel-world USD asset path (used when --world gravel).",
    )
    parser.add_argument(
        "--line-world-asset",
        type=Path,
        default=sim_dir / "assets" / "line_world" / "line_world.usda",
        help="Line-following-track USD asset path (used when --world line).",
    )
    parser.add_argument(
        "--combined-course-asset",
        type=Path,
        default=sim_dir / "assets" / "combined_course" / "combined_course.usda",
        help="Combined obstacle course USD asset path (used when --world combined).",
    )
    parser.add_argument(
        "--car-start-x-m",
        type=float,
        default=0.0,
        help="Initial x position of the car. Use a negative value (e.g. -1.0) when --world slope so the car starts on the flat approach and can drive into the ramp.",
    )
    parser.add_argument(
        "--car-start-z-m",
        type=float,
        default=0.0,
        help="Initial z offset of the car (added on top of the chassis-axle baseline). Use this with --world slope to place the car on the ramp surface at the chosen x.",
    )
    parser.add_argument(
        "--car-start-y-m",
        type=float,
        default=0.0,
        help="Initial y offset (lateral). Use for line-following start-off-line tests.",
    )
    parser.add_argument(
        "--car-start-yaw-deg",
        type=float,
        default=0.0,
        help="Initial yaw rotation around Z (deg). Use for line-following heading tests.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=sim_dir / "output" / "pid_effort_result.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=sim_dir / "output" / "pid_effort_timeseries.csv",
        help="Telemetry CSV path.",
    )
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdGeom


WHEEL_JOINT_NAMES = ["left_wheel_joint", "right_wheel_joint"]
BASE_PATH = "/World/BalanceCar/base_link"
WHEEL_RADIUS_M = 0.0335   # matches v2 USD asset (real Yahboom 67 mm OD)
WHEEL_TRACK_M = 0.170     # matches v2 USD asset (real Yahboom ~170 mm)

# Mutable keyboard state for --command-profile teleop. Updated either by the
# carb input callback OR by per-step polling in the main loop, whichever
# manages to work in the current Isaac Sim configuration.
TELEOP_KEYS = {"forward": False, "back": False, "left": False, "right": False}
_TELEOP_SUBSCRIPTION = None  # carb input subscription handle, kept alive
_TELEOP_INPUT_IFACE = None  # carb.input.IInput handle for polling
_TELEOP_KEYBOARD = None  # keyboard device handle for polling
_TELEOP_KI = None  # carb.input.KeyboardInput enum module reference
_TELEOP_KEYS_DOWN_PREV: set[str] = set()  # for press/release print debug


def setup_teleop_keyboard() -> None:
    """Subscribe to keyboard events AND prepare polling fallback. Both paths
    update TELEOP_KEYS; the main loop calls poll_teleop_keyboard() each tick."""
    global _TELEOP_SUBSCRIPTION, _TELEOP_INPUT_IFACE, _TELEOP_KEYBOARD, _TELEOP_KI
    try:
        import carb.input
        import omni.appwindow

        _TELEOP_INPUT_IFACE = carb.input.acquire_input_interface()
        appwindow = omni.appwindow.get_default_app_window()
        _TELEOP_KEYBOARD = appwindow.get_keyboard()
        _TELEOP_KI = carb.input.KeyboardInput

        ET = carb.input.KeyboardEventType
        key_to_flag = {
            _TELEOP_KI.W: "forward",
            _TELEOP_KI.UP: "forward",
            _TELEOP_KI.S: "back",
            _TELEOP_KI.DOWN: "back",
            _TELEOP_KI.A: "left",
            _TELEOP_KI.LEFT: "left",
            _TELEOP_KI.D: "right",
            _TELEOP_KI.RIGHT: "right",
        }

        def on_kb_event(event, *_args, **_kwargs):
            flag = key_to_flag.get(event.input)
            if flag is None:
                return True
            if event.type == ET.KEY_PRESS:
                TELEOP_KEYS[flag] = True
            elif event.type == ET.KEY_RELEASE:
                TELEOP_KEYS[flag] = False
            return True

        _TELEOP_SUBSCRIPTION = _TELEOP_INPUT_IFACE.subscribe_to_keyboard_events(
            _TELEOP_KEYBOARD, on_kb_event
        )
        print("TELEOP_KEYBOARD_READY keys=W/A/S/D or arrow keys (callback+polling)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"TELEOP_KEYBOARD_WARN {exc}", flush=True)


def poll_teleop_keyboard() -> None:
    """Polling fallback. Reads carb keyboard button flags each tick and
    updates TELEOP_KEYS. Works even when callback-based subscription is
    eclipsed by Kit's hotkey system."""
    global _TELEOP_KEYS_DOWN_PREV
    if _TELEOP_INPUT_IFACE is None or _TELEOP_KEYBOARD is None or _TELEOP_KI is None:
        return
    try:
        import carb.input

        # Combined key set per movement flag
        keys_for_flag = {
            "forward": (_TELEOP_KI.W, _TELEOP_KI.UP),
            "back": (_TELEOP_KI.S, _TELEOP_KI.DOWN),
            "left": (_TELEOP_KI.A, _TELEOP_KI.LEFT),
            "right": (_TELEOP_KI.D, _TELEOP_KI.RIGHT),
        }
        # carb.input.BUTTON_FLAG_DOWN bit: 1 << 0 in some versions; safer to
        # rely on the named constant.
        DOWN = carb.input.BUTTON_FLAG_DOWN
        currently_down: set[str] = set()
        for flag, keys in keys_for_flag.items():
            pressed = False
            for key in keys:
                state = _TELEOP_INPUT_IFACE.get_keyboard_button_flags(_TELEOP_KEYBOARD, key)
                if state & DOWN:
                    pressed = True
                    break
            TELEOP_KEYS[flag] = pressed
            if pressed:
                currently_down.add(flag)
        # Log transitions for visibility.
        if currently_down != _TELEOP_KEYS_DOWN_PREV:
            print(f"TELEOP_KEYS_DOWN={sorted(currently_down)}", flush=True)
            _TELEOP_KEYS_DOWN_PREV = currently_down
    except Exception as exc:  # noqa: BLE001
        # Only print once; failing every tick would spam.
        if not hasattr(poll_teleop_keyboard, "_warned"):
            print(f"TELEOP_POLL_WARN {exc}", flush=True)
            poll_teleop_keyboard._warned = True  # type: ignore[attr-defined]


@dataclass
class State:
    # IMU bias values sampled once at startup; constant for the run. Models a
    # real MPU6050's turn-on bias that persists across the session.
    imu_pitch_bias_rad: float = 0.0
    imu_gyro_pitch_bias_rad_s: float = 0.0
    imu_gyro_yaw_bias_rad_s: float = 0.0
    # Quantized encoder-position state (truth - residual = what controller saw).
    enc_left_residual_rad: float = 0.0
    enc_right_residual_rad: float = 0.0
    previous_pitch: float | None = None
    previous_yaw: float | None = None
    previous_left_joint: float | None = None
    previous_right_joint: float | None = None
    target_position_m: float = 0.0
    target_position_initialised: bool = False
    lqi_velocity_error_integral: float = 0.0
    slope_est_rad: float = 0.0
    slope_used_rad: float = 0.0
    loop_accumulator_s: float = 0.0
    speed_counter: int = 0
    turn_counter: int = 0
    left_pulse_accum: float = 0.0
    right_pulse_accum: float = 0.0
    speed_filter_old: float = 0.0
    position_pulses: float = 0.0
    angle_output_pwm: float = 0.0
    speed_output_pwm: float = 0.0
    turn_output_pwm: float = 0.0
    turn_accumulator: float = 0.0
    pwm_left: float = 0.0
    pwm_right: float = 0.0
    # Smoothed drive offset — ramps toward the commanded value rather than
    # snapping, so a brief W tap doesn't slam the PID into a huge instant
    # lean command and topple the cart.
    drive_offset_smoothed_deg: float = 0.0
    # Same idea for the turn differential: ramp 0 -> ±magnitude rather
    # than stepping, so a brief A/D tap doesn't slam the single-axle
    # chassis with an unrecoverable lateral impulse.
    turn_pwm_smoothed: float = 0.0


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def world_transform(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"Prim does not exist: {prim_path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def body_angles(matrix) -> tuple[float, float, float]:
    rotation = matrix.ExtractRotationMatrix()
    local_x_world = rotation.GetRow(0)
    local_z_world = rotation.GetRow(2)
    pitch = math.atan2(-float(local_z_world[0]), float(local_z_world[2]))
    yaw = math.atan2(float(local_x_world[1]), float(local_x_world[0]))
    roll = math.atan2(float(local_z_world[1]), float(local_z_world[2]))
    return pitch, yaw, roll


def command_profile(step: int, dt: float) -> tuple[float, float]:
    if args.command_profile == "stand":
        return 0.0, 0.0
    if args.command_profile == "teleop":
        vx = 0.0
        if TELEOP_KEYS["forward"]:
            vx += args.teleop_velocity_m_s
        if TELEOP_KEYS["back"]:
            vx -= args.teleop_velocity_m_s
        yaw = 0.0
        if TELEOP_KEYS["left"]:
            yaw += args.teleop_yaw_rate_rad_s
        if TELEOP_KEYS["right"]:
            yaw -= args.teleop_yaw_rate_rad_s
        return vx, yaw
    time_s = step * dt
    if args.command_profile == "climb":
        if time_s < 0.5:
            return 0.0, 0.0
        return 0.20, 0.0
    if args.command_profile == "descend":
        if time_s < 0.5:
            return 0.0, 0.0
        return -0.15, 0.0
    if time_s < 1.0:
        return 0.0, 0.0
    if time_s < 3.0:
        return 0.10, 0.0
    if time_s < 4.5:
        return 0.0, 0.0
    if time_s < 6.5:
        return -0.08, 0.0
    if time_s < 8.0:
        return 0.0, 0.45
    if time_s < 9.5:
        return 0.0, -0.45
    return 0.0, 0.0


def _stress_teleop_flags(time_s: float) -> tuple[int, int, int, int, int, int]:
    """Scripted WASD sequence that mimics a human player so we can validate
    the controller headless. Returns (forward, back, turn_left, turn_right,
    spin_left, spin_right). Each phase prints a label via the cart's print
    layer (not here; the caller sees the actions in the CSV)."""
    # Full stress profile testing all real-world driving patterns.
    # With proper motor back-EMF modeling, A/D should be safe like real Yahboom.
    phases = [
        (0.0,  2.0,  0, 0, 0, 0),    # stand
        (2.0,  6.0,  1, 0, 0, 0),    # W hold 4s
        (6.0,  6.3,  1, 0, 1, 0),    # W+A 300ms (forward-left)
        (6.3,  8.0,  1, 0, 0, 0),    # W only
        (8.0,  8.3,  1, 0, 0, 1),    # W+D 300ms (forward-right)
        (8.3,  10.0, 1, 0, 0, 0),    # W only
        (10.0, 11.5, 0, 0, 0, 0),    # release, drift
        (11.5, 12.0, 0, 0, 1, 0),    # A only (in-place left)
        (12.0, 12.5, 0, 0, 0, 0),
        (12.5, 13.0, 0, 0, 0, 1),    # D only (in-place right)
        (13.0, 14.0, 0, 0, 0, 0),
        (14.0, 15.0, 0, 1, 0, 0),    # S brake
        (15.0, 16.0, 0, 0, 0, 0),
        (16.0, 25.0, 1, 0, 0, 0),    # W hold to finish course
    ]
    for start, end, f, b, l, r in phases:
        if start <= time_s < end:
            return f, b, l, r, 0, 0
    return 0, 0, 0, 0, 0, 0


def yahboom_command_flags(step: int, dt: float) -> tuple[int, int, int, int, int, int]:
    if args.command_profile == "stand":
        return 0, 0, 0, 0, 0, 0
    if args.command_profile == "teleop":
        forward = 1 if TELEOP_KEYS["forward"] else 0
        back = 1 if TELEOP_KEYS["back"] else 0
        turn_left = 1 if TELEOP_KEYS["left"] else 0
        turn_right = 1 if TELEOP_KEYS["right"] else 0
        return forward, back, turn_left, turn_right, 0, 0
    if args.command_profile == "stress_teleop":
        return _stress_teleop_flags(step * dt)
    time_s = step * dt
    if args.command_profile == "climb":
        if time_s < 0.5:
            return 0, 0, 0, 0, 0, 0
        return 1, 0, 0, 0, 0, 0
    if args.command_profile == "descend":
        if time_s < 0.5:
            return 0, 0, 0, 0, 0, 0
        return 0, 1, 0, 0, 0, 0
    if time_s < 1.0:
        return 0, 0, 0, 0, 0, 0
    if time_s < 3.0:
        return 1, 0, 0, 0, 0, 0
    if time_s < 4.5:
        return 0, 0, 0, 0, 0, 0
    if time_s < 6.5:
        return 0, 1, 0, 0, 0, 0
    if time_s < 8.0:
        return 0, 0, 1, 0, 0, 0
    if time_s < 9.5:
        return 0, 0, 0, 1, 0, 0
    return 0, 0, 0, 0, 0, 0


def update_yahboom_controller(
    state: State,
    step: int,
    dt: float,
    pitch: float,
    pitch_rate: float,
    yaw_rate: float,
    joint_positions: np.ndarray,
) -> tuple[float, float, dict]:
    if state.previous_left_joint is None:
        state.previous_left_joint = float(joint_positions[0])
        state.previous_right_joint = float(joint_positions[1])

    delta_left_rad = float(joint_positions[0]) - state.previous_left_joint
    delta_right_rad = float(joint_positions[1]) - state.previous_right_joint
    state.previous_left_joint = float(joint_positions[0])
    state.previous_right_joint = float(joint_positions[1])

    counts_per_rad = args.encoder_cpr / (2.0 * math.pi)
    state.left_pulse_accum += delta_left_rad * counts_per_rad
    state.right_pulse_accum += delta_right_rad * counts_per_rad

    forward_flag, back_flag, turn_left, turn_right, spin_left, spin_right = yahboom_command_flags(step, dt)
    # Hybrid architecture — keeps the OLD code's pitch-bias drive (which is
    # what actually works in our sim: high motor torque + low chassis inertia
    # need a forward-LEAN bias, not just speed-integral injection), but
    # adopts the STM32 Yahboom reference's TURN handling (direct PWM
    # differential ±BluetoothDirection, no PID accumulator, no ramp).
    forward_increment = args.command_pulse_step if forward_flag else 0.0
    back_increment = -args.command_pulse_step if back_flag else 0.0
    drive_offset_target_deg = 0.0
    if forward_flag:
        drive_offset_target_deg = args.drive_pitch_offset_deg
    elif back_flag:
        drive_offset_target_deg = -args.drive_pitch_offset_deg
    # Asymmetric ramp: press fast, release slow (cart momentum dissipates
    # before PID stops leaning, avoiding the over-correction-and-topple).
    # EXCEPT: if the user is pressing a turn key (A/D) without a forward/back
    # key, we collapse drive_offset to 0 FAST. Rationale: a single-axle car
    # cannot safely turn while still rolling forward — centripetal force
    # tips it. Forcing a near-instant brake before the turn applies keeps
    # the cart upright.
    turn_active_no_fwd = (turn_left or turn_right or spin_left or spin_right) \
        and not (forward_flag or back_flag)
    pressing = abs(drive_offset_target_deg) > abs(state.drive_offset_smoothed_deg)
    if turn_active_no_fwd:
        tau = args.drive_offset_turn_brake_tau_s
    else:
        tau = args.drive_offset_ramp_tau_s if pressing else args.drive_offset_release_tau_s
    ramp_alpha = dt / (tau + dt) if tau > 0.0 else 1.0
    state.drive_offset_smoothed_deg += ramp_alpha * (drive_offset_target_deg - state.drive_offset_smoothed_deg)
    drive_offset_deg = state.drive_offset_smoothed_deg
    # Turn: direct PWM differential (Yahboom STM32 style, no accumulator).
    # Signs are FLIPPED vs the STM32 reference because our PWM combination
    # has a global negation (pwm_left = -angle - speed - turn) that the
    # STM32 doesn't have. Concretely: pressing A (turn_left) must end up
    # slowing LEFT and speeding RIGHT, i.e. pwm_left more NEGATIVE, which
    # requires `turn_output_pwm` to be POSITIVE here.
    # Target turn PWM (instant step). Without ramping this differential
    # straight from 0 -> ±magnitude in one tick creates a lateral force
    # impulse that the single-axle chassis can't recover from. Ramp it
    # via state.turn_pwm_smoothed below.
    bluetooth_target = 0.0
    if turn_left or spin_left:
        bluetooth_target = args.bluetooth_direction_magnitude
    elif turn_right or spin_right:
        bluetooth_target = -args.bluetooth_direction_magnitude
    # Ramp the differential at args.turn_ramp_tau_s. Same first-order
    # low-pass we use for drive_offset.
    tau_turn = max(args.turn_ramp_tau_s, 1e-6)
    alpha_turn = dt / (tau_turn + dt)
    state.turn_pwm_smoothed += alpha_turn * (bluetooth_target - state.turn_pwm_smoothed)
    bluetooth_direction = state.turn_pwm_smoothed

    loop_period_s = args.loop_period_ms * 0.001
    state.loop_accumulator_s += dt
    ran_loop = False
    while state.loop_accumulator_s >= loop_period_s:
        state.loop_accumulator_s -= loop_period_s
        ran_loop = True

        pitch_deg = math.degrees(pitch)
        gyro_x_deg_s = math.degrees(pitch_rate)
        gyro_z_deg_s = math.degrees(yaw_rate)
        # Angle loop: target pitch = -drive_offset_deg (when W held the cart
        # actively leans into the drive direction). Slope effects handled by
        # external feed-forward at the wheel command stage.
        state.angle_output_pwm = (
            args.angle_kp * (pitch_deg + args.angle_balance_offset_deg + drive_offset_deg)
            + args.angle_kd * gyro_x_deg_s
        )

        state.speed_counter += 1
        if state.speed_counter >= args.speed_loop_divisor:
            if args.no_encoder:
                state.left_pulse_accum = 0.0
                state.right_pulse_accum = 0.0
                state.speed_filter_old = 0.0
                state.position_pulses = 0.0
                state.speed_output_pwm = 0.0
            else:
                pulses = state.left_pulse_accum + state.right_pulse_accum
                state.left_pulse_accum = 0.0
                state.right_pulse_accum = 0.0
                speed_filter = state.speed_filter_old * 0.7 + pulses * 0.3
                state.speed_filter_old = speed_filter
                # Anti-windup: when the wheels are not actually moving (creep
                # or stalled), decay position_pulses so it doesn't sit at the
                # +/-3000 clamp forever. Two scenarios both benefit:
                #   1. Cart held against a slope while W is pressed — without
                #      decay, the integrator wins out over the angle loop and
                #      the cart sits in static equilibrium.
                #   2. Cart at rest after W release — without decay, the
                #      integrator is still commanding heavy brake, and a
                #      subsequent A/D press launches the cart sideways.
                # Note: we no longer require cmd_active. Decay whenever the
                # wheels aren't moving, period.
                stalled = abs(speed_filter) < args.speed_stall_threshold
                if stalled and 0.0 < args.speed_antiwindup_decay < 1.0:
                    state.position_pulses *= args.speed_antiwindup_decay
                else:
                    state.position_pulses = clamp(
                        state.position_pulses + speed_filter + forward_increment + back_increment,
                        args.speed_pulses_clamp,
                    )
                state.speed_output_pwm = (
                    args.speed_i * (0.0 - state.position_pulses)
                    + args.speed_p * (0.0 - speed_filter)
                )
            state.speed_counter = 0

        # Turn = direct PWM differential (Yahboom STM32 style, no PID).
        #
        # Two layered mechanisms keep the single-axle chassis stable while
        # turning. They mirror real STM32 reference firmware (B570 W13
        # control.c:Turn(), line 160-172):
        #
        #   1. Speed-taper (sim-only fallback). Reduces turn magnitude with
        #      forward speed so a fast cart can't be commanded into a sharp
        #      pivot. Disable with --turn-disable-speed-taper to match real
        #      STM32 which keeps full turn proportional authority at all
        #      speeds.
        #         v = 0           -> turn × 1.0   (full in-place spin)
        #         v = gate / 2    -> turn × 0.75
        #         v = gate        -> turn × 0.5
        #         v >> gate       -> turn × 0.3   (clamp floor)
        #
        #   2. Gyro-rate damping (real STM32 port). Subtracts
        #      `Kd * yaw_rate_dps` from the turn output, but only while
        #      forward/back is held (matches `Kd=Turn_Kd if Flag_front||
        #      Flag_back else 0` on line 168). This is the mechanism the
        #      real cart uses to keep yaw from over-shooting on the
        #      instant 0→±27 Turn_Target step. Enable with
        #      --turn-kd-pwm-per-dps 0.364 (= real Turn_Kd=0.6 LSB^-1
        #      scaled to our PWM range 255 from real 6900).
        wheel_v_m_s = (
            state.speed_filter_old
            / max(args.encoder_cpr, 1.0)
            * 2.0 * math.pi * WHEEL_RADIUS_M * 25.0
        )
        if args.turn_disable_speed_taper:
            turn_scale = 1.0
        else:
            gate = max(args.turn_speed_gate_m_s, 0.01)
            v_ratio = min(2.0, abs(wheel_v_m_s) / gate)
            turn_scale = max(0.3, 1.0 - 0.5 * v_ratio)
        # Kd*gyro damping. The sign mirrors real STM32: a left-turn
        # command (bluetooth_direction > 0 in our convention) drives the
        # cart to a positive yaw_rate; subtracting `Kd*yaw_rate` then
        # eats into the command magnitude as the actual yaw rate grows,
        # producing a steady-state turn that depends on motor friction
        # rather than on hand-tuning the magnitude.
        turn_kd_damp_pwm = 0.0
        if args.turn_kd_pwm_per_dps > 0.0 and (forward_flag or back_flag):
            turn_kd_damp_pwm = args.turn_kd_pwm_per_dps * gyro_z_deg_s
        state.turn_output_pwm = bluetooth_direction * turn_scale - turn_kd_damp_pwm
        state.turn_accumulator = bluetooth_direction  # diagnostic only

        state.pwm_left = -state.angle_output_pwm - state.speed_output_pwm - state.turn_output_pwm
        state.pwm_right = -state.angle_output_pwm - state.speed_output_pwm + state.turn_output_pwm
        state.pwm_left = clamp(state.pwm_left, args.pwm_limit)
        state.pwm_right = clamp(state.pwm_right, args.pwm_limit)
        # PWM dead-band: real STM32 firmware ignores small PWM commands
        # because the H-bridge driver (TB6612FNG) has a non-zero turn-on
        # threshold and the motor has bearing stiction. Without this in
        # sim, micro-corrections from the PID accumulate into a hunting
        # limit-cycle that eventually tips the cart even with no input.
        if abs(state.pwm_left) < args.pwm_dead_band:
            state.pwm_left = 0.0
        if abs(state.pwm_right) < args.pwm_dead_band:
            state.pwm_right = 0.0
        if abs(pitch_deg) > args.fall_angle_deg:
            state.pwm_left = 0.0
            state.pwm_right = 0.0

    left_effort = args.effort_sign * (state.pwm_left / args.pwm_limit) * args.effort_limit
    right_effort = args.effort_sign * (state.pwm_right / args.pwm_limit) * args.effort_limit
    debug = {
        "ran_loop": ran_loop,
        "angle_output_pwm": state.angle_output_pwm,
        "speed_output_pwm": state.speed_output_pwm,
        "turn_output_pwm": state.turn_output_pwm,
        "speed_filter_pulses": state.speed_filter_old,
        "position_pulses": state.position_pulses,
        "pwm_left": state.pwm_left,
        "pwm_right": state.pwm_right,
        "drive_offset_deg": drive_offset_deg,
    }
    return left_effort, right_effort, debug


def update_threering_controller(
    state: State,
    step: int,
    dt: float,
    pitch: float,
    pitch_rate: float,
    yaw_rate: float,
    joint_positions: np.ndarray,
) -> tuple[float, float, dict]:
    """W13 教材 v2「三環並聯」reference implementation.

    Balance PD + Velocity PI (with LPF + position-integral) + Turn PD
    (gain-scheduled: Kd active only when moving). Combine via:
        motor_left  =  -(balance + velocity)  -  turn
        motor_right =  -(balance + velocity)  +  turn
    The global negation matches our asset's effort_sign convention
    (positive pitch must drive wheels forward, see drive_pitch_offset_decision).
    """
    # ----- Encoder accumulation (same as yahboom) -----
    if state.previous_left_joint is None:
        state.previous_left_joint = float(joint_positions[0])
        state.previous_right_joint = float(joint_positions[1])
    delta_l_rad = float(joint_positions[0]) - state.previous_left_joint
    delta_r_rad = float(joint_positions[1]) - state.previous_right_joint
    state.previous_left_joint = float(joint_positions[0])
    state.previous_right_joint = float(joint_positions[1])
    counts_per_rad = args.encoder_cpr / (2.0 * math.pi)
    state.left_pulse_accum += delta_l_rad * counts_per_rad
    state.right_pulse_accum += delta_r_rad * counts_per_rad

    # ----- Teleop command flags → targets -----
    fwd, back, turn_l, turn_r, spin_l, spin_r = yahboom_command_flags(step, dt)
    target_speed_pulses = 0.0
    if fwd:
        target_speed_pulses = +args.threering_target_speed_pulses
    elif back:
        target_speed_pulses = -args.threering_target_speed_pulses
    target_yaw_rate_deg_s = 0.0
    if turn_l or spin_l:
        target_yaw_rate_deg_s = -args.threering_target_yaw_rate_deg_s
    elif turn_r or spin_r:
        target_yaw_rate_deg_s = +args.threering_target_yaw_rate_deg_s

    # ----- Run the discrete PID loop at loop_period_ms -----
    loop_period_s = args.loop_period_ms * 0.001
    state.loop_accumulator_s += dt
    ran_loop = False
    while state.loop_accumulator_s >= loop_period_s:
        state.loop_accumulator_s -= loop_period_s
        ran_loop = True

        pitch_deg = math.degrees(pitch)
        gyro_x_deg_s = math.degrees(pitch_rate)
        gyro_z_deg_s = math.degrees(yaw_rate)

        # Balance PD (D term reads gyro directly — spec section 3.1)
        # balance_pwm = Kp*(pitch - MIDDLE) + Kd*gyro_pitch
        state.angle_output_pwm = (
            args.threering_kp_balance
            * (pitch_deg - args.threering_middle_angle_deg)
            + args.threering_kd_balance * gyro_x_deg_s
        )

        # Velocity PI with LPF + position-form integral (spec section 3.2).
        # Anti-windup decay is OUR addition (not in the spec). The STM32
        # reference does a binary `if stopflag==1: position=0`; we use a
        # smoother multiplicative decay because in sim the wheels never go
        # exactly to zero. Without this, target=0 + tiny encoder noise =>
        # position_pulses winds up, velocity loop fights itself, runaway.
        state.speed_counter += 1
        if state.speed_counter >= args.speed_loop_divisor:
            pulses = state.left_pulse_accum + state.right_pulse_accum
            state.left_pulse_accum = 0.0
            state.right_pulse_accum = 0.0
            speed_err_raw = target_speed_pulses - pulses
            alpha = args.threering_lpf_alpha
            state.speed_filter_old = (
                alpha * state.speed_filter_old + (1.0 - alpha) * speed_err_raw
            )
            stalled = (
                abs(state.speed_filter_old) < args.threering_stall_threshold
                and abs(target_speed_pulses) < 0.5
            )
            if stalled and 0.0 < args.threering_antiwindup_decay < 1.0:
                state.position_pulses *= args.threering_antiwindup_decay
            else:
                state.position_pulses = clamp(
                    state.position_pulses + state.speed_filter_old,
                    args.threering_position_clamp,
                )
            # velocity_pwm = -Kp*err_filt - Ki*position_err (spec signs)
            state.speed_output_pwm = (
                -args.threering_kp_velocity * state.speed_filter_old
                - args.threering_ki_velocity * state.position_pulses
            )
            state.speed_counter = 0

        # Turn PD with gain scheduling (spec section 3.3)
        is_moving = abs(target_speed_pulses) > 0.01
        kd_turn_eff = args.threering_kd_turn if is_moving else 0.0
        # Spec: turn_pwm = Kp*target + Kd*gyro_yaw (D-on-measurement)
        state.turn_output_pwm = (
            args.threering_kp_turn * target_yaw_rate_deg_s
            + kd_turn_eff * gyro_z_deg_s
        )

        # Parallel + differential combine. Global negation to match the
        # asset's effort_sign convention (same as yahboom controller).
        state.pwm_left = (
            -state.angle_output_pwm - state.speed_output_pwm - state.turn_output_pwm
        )
        state.pwm_right = (
            -state.angle_output_pwm - state.speed_output_pwm + state.turn_output_pwm
        )
        pwm_lim = args.threering_pwm_limit
        state.pwm_left = clamp(state.pwm_left, pwm_lim)
        state.pwm_right = clamp(state.pwm_right, pwm_lim)
        # Spec section 2.1: fall protection at |pitch| > 40°
        if abs(pitch_deg) > args.fall_angle_deg:
            state.pwm_left = 0.0
            state.pwm_right = 0.0
            state.position_pulses = 0.0  # reset integrator after fall

    left_effort = args.effort_sign * (state.pwm_left / args.threering_pwm_limit) * args.effort_limit
    right_effort = args.effort_sign * (state.pwm_right / args.threering_pwm_limit) * args.effort_limit
    debug = {
        "ran_loop": ran_loop,
        "balance_pwm": state.angle_output_pwm,
        "velocity_pwm": state.speed_output_pwm,
        "turn_pwm": state.turn_output_pwm,
        "speed_filter": state.speed_filter_old,
        "position_pulses": state.position_pulses,
        "target_speed_pulses": target_speed_pulses,
        "target_yaw_rate_deg_s": target_yaw_rate_deg_s,
        "pwm_left": state.pwm_left,
        "pwm_right": state.pwm_right,
    }
    return left_effort, right_effort, debug



def update_legacy_controller(
    state: State,
    step: int,
    dt: float,
    pitch: float,
    pitch_rate: float,
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
) -> tuple[float, float, float, dict]:
    wheel_position_m = 0.5 * float(joint_positions[0] + joint_positions[1]) * WHEEL_RADIUS_M
    wheel_velocity_m_s = 0.5 * float(joint_velocities[0] + joint_velocities[1]) * WHEEL_RADIUS_M
    wheel_yaw_rate = (float(joint_velocities[1] - joint_velocities[0]) * WHEEL_RADIUS_M) / WHEEL_TRACK_M

    target_velocity, target_yaw_rate = command_profile(step, dt)
    state.target_position_m += target_velocity * dt

    target_pitch_limit = math.radians(args.target_pitch_limit_deg)
    position_error = state.target_position_m - wheel_position_m
    speed_error = target_velocity - wheel_velocity_m_s
    target_pitch = clamp(args.position_kp * position_error + args.speed_kp * speed_error, target_pitch_limit)
    balance_error = target_pitch - pitch
    base_effort = args.effort_sign * (args.balance_kp * balance_error - args.balance_kd * pitch_rate)
    yaw_effort = args.yaw_kp * (target_yaw_rate - wheel_yaw_rate)
    left_effort = clamp(base_effort - yaw_effort, args.effort_limit)
    right_effort = clamp(base_effort + yaw_effort, args.effort_limit)
    return left_effort, right_effort, target_pitch, {
        "target_velocity": target_velocity,
        "target_yaw_rate": target_yaw_rate,
        "wheel_yaw_rate": wheel_yaw_rate,
    }


def build_lqr_gain(dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the discrete-time LQR gain for the two-wheel balance car.

    State vector x = [theta, theta_dot, x_w, x_w_dot] where:
      theta      = pitch (rad), positive when cart tilts toward +X (forward fall direction)
      theta_dot  = pitch rate (rad/s)
      x_w        = wheel position (m) along +X
      x_w_dot    = wheel velocity (m/s) along +X

    Input u = total wheel torque (N*m). The two wheels share u/2 each (yaw is
    handled by a separate PD loop layered on top).
    """
    from scipy.linalg import expm, solve_discrete_are

    g = 9.81
    M_b = args.lqr_chassis_mass
    M_w_total = args.lqr_wheel_mass_total
    r = WHEEL_RADIUS_M
    L = args.lqr_com_height
    I_p = args.lqr_chassis_pitch_inertia
    I_w_total = args.lqr_wheel_inertia_total

    M_w_eff = M_w_total + I_w_total / (r * r)

    # Mass matrix M @ [ddot_x, ddot_theta]^T = [u/r, M_b*g*L*theta - u]^T
    #   Row 0: horizontal Newton's law (chassis + effective wheel mass)
    #   Row 1: chassis pitch about wheel axle. Gravity contributes +M_b*g*L*theta
    #          (small-angle); the motor reaction torque on the chassis is -u
    #          (the +u is applied to the wheel by the chassis stator).
    # State: x = [theta, theta_dot, x_w, x_w_dot]. Convention: theta positive
    # means cart leaning toward +X (forward fall direction).
    M_mat = np.array(
        [
            [M_w_eff + M_b, M_b * L],
            [M_b * L, I_p + M_b * L * L],
        ],
        dtype=float,
    )
    M_inv = np.linalg.inv(M_mat)

    A = np.zeros((4, 4), dtype=float)
    A[0, 1] = 1.0
    A[1, 0] = M_inv[1, 1] * M_b * g * L
    A[2, 3] = 1.0
    A[3, 0] = M_inv[0, 1] * M_b * g * L

    # B accounts for both the +u/r horizontal force at the wheel and the -u
    # motor reaction torque on the chassis (about the wheel axle).
    B = np.zeros((4, 1), dtype=float)
    B[1, 0] = M_inv[1, 0] / r + M_inv[1, 1] * (-1.0)
    B[3, 0] = M_inv[0, 0] / r + M_inv[0, 1] * (-1.0)

    AB = np.zeros((5, 5), dtype=float)
    AB[:4, :4] = A
    AB[:4, 4:] = B
    AB_d = expm(AB * dt)
    A_d = AB_d[:4, :4]
    B_d = AB_d[:4, 4:]

    Q = np.diag(
        [args.lqr_q_pitch, args.lqr_q_pitch_rate, args.lqr_q_pos, args.lqr_q_vel]
    )
    R = np.diag([args.lqr_r_torque])

    P = solve_discrete_are(A_d, B_d, Q, R)
    K = np.linalg.inv(R + B_d.T @ P @ B_d) @ (B_d.T @ P @ A_d)
    return K, A_d, B_d


def build_lqi_gain(dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the discrete LQI gain. Augmented state x_aug = [x_lqr, integral_v_err]

    The integral state accumulates the velocity error (x_w_dot - v_ref).
    Continuous augmentation: x_aug_dot = [A_x, 0; C_v, 0] x_aug + [B; 0] u
    where C_v = [0, 0, 0, 1] picks out x_w_dot (and v_ref is subtracted in the
    update loop by shifting the state reference).
    """
    from scipy.linalg import expm, solve_discrete_are

    g = 9.81
    M_b = args.lqr_chassis_mass
    M_w_total = args.lqr_wheel_mass_total
    r = WHEEL_RADIUS_M
    L = args.lqr_com_height
    I_p = args.lqr_chassis_pitch_inertia
    I_w_total = args.lqr_wheel_inertia_total
    M_w_eff = M_w_total + I_w_total / (r * r)

    M_mat = np.array(
        [[M_w_eff + M_b, M_b * L], [M_b * L, I_p + M_b * L * L]], dtype=float
    )
    M_inv = np.linalg.inv(M_mat)

    A = np.zeros((4, 4), dtype=float)
    A[0, 1] = 1.0
    A[1, 0] = M_inv[1, 1] * M_b * g * L
    A[2, 3] = 1.0
    A[3, 0] = M_inv[0, 1] * M_b * g * L

    B = np.zeros((4, 1), dtype=float)
    B[1, 0] = M_inv[1, 0] / r + M_inv[1, 1] * (-1.0)
    B[3, 0] = M_inv[0, 0] / r + M_inv[0, 1] * (-1.0)

    # Augment with integral of velocity-error state (the 5th state).
    A_aug = np.zeros((5, 5), dtype=float)
    A_aug[:4, :4] = A
    A_aug[4, 3] = 1.0  # d(integral)/dt = x_w_dot - v_ref (v_ref absorbed in state)
    B_aug = np.zeros((5, 1), dtype=float)
    B_aug[:4, :] = B

    AB = np.zeros((6, 6), dtype=float)
    AB[:5, :5] = A_aug
    AB[:5, 5:] = B_aug
    AB_d = expm(AB * dt)
    A_d = AB_d[:5, :5]
    B_d = AB_d[:5, 5:]

    Q = np.diag(
        [
            args.lqr_q_pitch,
            args.lqr_q_pitch_rate,
            args.lqr_q_pos,
            args.lqr_q_vel,
            args.lqi_q_integral,
        ]
    )
    R = np.diag([args.lqr_r_torque])

    P = solve_discrete_are(A_d, B_d, Q, R)
    K = np.linalg.inv(R + B_d.T @ P @ B_d) @ (B_d.T @ P @ A_d)
    return K, A_d, B_d


def update_lqi_controller(
    state: State,
    step: int,
    dt: float,
    pitch: float,
    pitch_rate: float,
    yaw_rate: float,
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
    K: np.ndarray,
) -> tuple[float, float, dict]:
    theta_lqr = -pitch
    theta_dot_lqr = -pitch_rate
    if args.no_encoder:
        x_w = 0.0
        x_w_dot = 0.0
    else:
        x_w = 0.5 * float(joint_positions[0] + joint_positions[1]) * WHEEL_RADIUS_M
        x_w_dot = 0.5 * float(joint_velocities[0] + joint_velocities[1]) * WHEEL_RADIUS_M

    if not state.target_position_initialised:
        state.target_position_m = x_w
        state.target_position_initialised = True

    target_velocity, target_yaw_rate = command_profile(step, dt)
    target_velocity *= args.lqr_target_velocity_scale
    state.target_position_m += target_velocity * dt

    velocity_error = x_w_dot - target_velocity
    state.lqi_velocity_error_integral = clamp(
        state.lqi_velocity_error_integral + velocity_error * dt,
        args.lqi_integral_clamp,
    )

    if args.no_encoder:
        x_state = np.array(
            [theta_lqr, theta_dot_lqr, 0.0, 0.0, state.lqi_velocity_error_integral],
            dtype=float,
        )
    else:
        x_state = np.array(
            [
                theta_lqr,
                theta_dot_lqr,
                x_w - state.target_position_m,
                x_w_dot - target_velocity,
                state.lqi_velocity_error_integral,
            ],
            dtype=float,
        )

    u_total = float(-(K @ x_state)[0])
    max_total = 2.0 * args.effort_limit
    u_total = max(-max_total, min(max_total, u_total))

    yaw_rate_cmd = target_yaw_rate * args.lqr_yaw_rate_cmd_scale
    yaw_error = yaw_rate_cmd - yaw_rate
    yaw_torque = args.lqr_yaw_kp * yaw_error

    left_effort = args.effort_sign * (u_total * 0.5 - yaw_torque)
    right_effort = args.effort_sign * (u_total * 0.5 + yaw_torque)
    left_effort = clamp(left_effort, args.effort_limit)
    right_effort = clamp(right_effort, args.effort_limit)

    debug = {
        "lqi_u_total": u_total,
        "lqi_yaw_torque": yaw_torque,
        "lqi_target_position_m": state.target_position_m,
        "lqi_target_velocity": target_velocity,
        "lqi_velocity_error": velocity_error,
        "lqi_integral": state.lqi_velocity_error_integral,
        "lqi_x_error": (x_w - state.target_position_m) if not args.no_encoder else 0.0,
        "pwm_left": (left_effort / max(args.effort_limit, 1e-9)) * args.pwm_limit,
        "pwm_right": (right_effort / max(args.effort_limit, 1e-9)) * args.pwm_limit,
    }
    return left_effort, right_effort, debug


def update_lqr_controller(
    state: State,
    step: int,
    dt: float,
    pitch: float,
    pitch_rate: float,
    yaw_rate: float,
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
    K: np.ndarray,
) -> tuple[float, float, dict]:
    """LQR controller. Converts (pitch_deg-style convention) to LQR theta convention.

    Our body_angles() returns pitch = -RotateY(rad), so pitch > 0 == cart tilted
    backward (top toward -X). The LQR linearization uses theta > 0 == cart tilted
    forward (toward +X). So theta_lqr = -pitch_world.
    """
    theta_lqr = -pitch
    theta_dot_lqr = -pitch_rate
    if args.no_encoder:
        x_w = 0.0
        x_w_dot = 0.0
    else:
        x_w = 0.5 * float(joint_positions[0] + joint_positions[1]) * WHEEL_RADIUS_M
        x_w_dot = 0.5 * float(joint_velocities[0] + joint_velocities[1]) * WHEEL_RADIUS_M

    # On the first call, anchor target_position to the current wheel position
    # so a cart spawned at a non-zero x is not commanded to drive back to 0.
    if not state.target_position_initialised:
        state.target_position_m = x_w
        state.target_position_initialised = True

    target_velocity, target_yaw_rate = command_profile(step, dt)
    target_velocity *= args.lqr_target_velocity_scale
    state.target_position_m += target_velocity * dt

    if args.no_encoder:
        # Zero out the position/velocity contribution; controller acts on
        # pitch / pitch_rate only. K is still the 4-state gain but the last
        # two state components are forced to zero, which gives the same effect
        # as truncating K to its first two columns.
        x_state = np.array([theta_lqr, theta_dot_lqr, 0.0, 0.0], dtype=float)
    else:
        x_state = np.array(
            [
                theta_lqr,
                theta_dot_lqr,
                x_w - state.target_position_m,
                x_w_dot - target_velocity,
            ],
            dtype=float,
        )

    u_total = float(-(K @ x_state)[0])  # total wheel torque
    # Clamp total torque so each wheel stays within effort_limit
    max_total = 2.0 * args.effort_limit
    u_total = max(-max_total, min(max_total, u_total))

    yaw_rate_cmd = target_yaw_rate * args.lqr_yaw_rate_cmd_scale
    yaw_error = yaw_rate_cmd - yaw_rate
    yaw_torque = args.lqr_yaw_kp * yaw_error  # differential torque between wheels

    left_effort = args.effort_sign * (u_total * 0.5 - yaw_torque)
    right_effort = args.effort_sign * (u_total * 0.5 + yaw_torque)
    left_effort = clamp(left_effort, args.effort_limit)
    right_effort = clamp(right_effort, args.effort_limit)

    debug = {
        "lqr_u_total": u_total,
        "lqr_yaw_torque": yaw_torque,
        "lqr_target_position_m": state.target_position_m,
        "lqr_target_velocity": target_velocity,
        "lqr_yaw_rate_cmd": yaw_rate_cmd,
        "lqr_theta_lqr": theta_lqr,
        "lqr_x_error": x_w - state.target_position_m,
        "pwm_left": (left_effort / max(args.effort_limit, 1e-9)) * args.pwm_limit,
        "pwm_right": (right_effort / max(args.effort_limit, 1e-9)) * args.pwm_limit,
    }
    return left_effort, right_effort, debug


def disable_joint_drives(stage) -> None:
    for joint_path in ("/World/BalanceCar/joints/left_wheel_joint", "/World/BalanceCar/joints/right_wheel_joint"):
        joint = stage.GetPrimAtPath(joint_path)
        for attr_name in (
            "drive:angular:physics:stiffness",
            "drive:angular:physics:damping",
            "drive:angular:physics:maxForce",
        ):
            attr = joint.GetAttribute(attr_name)
            if attr:
                attr.Set(0.0)


def run() -> dict:
    if not args.asset.exists():
        raise FileNotFoundError(args.asset)

    dt = 1.0 / args.physics_hz
    # rendering_dt must match physics_dt so that world.step(render=True) advances
    # the same simulated time as world.step(render=False). Otherwise the controller
    # (which assumes one step == physics_dt) sees 4x more pitch change per call in
    # GUI mode and goes unstable. See docs/AGENT_HANDOFF.md.
    world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
    world_assets = {
        "slope": (args.slope_world_asset, "/World/SlopeWorld"),
        "seesaw": (args.seesaw_world_asset, "/World/SeesawWorld"),
        "stairs": (args.stairs_world_asset, "/World/StairsWorld"),
        "gravel": (args.gravel_world_asset, "/World/GravelWorld"),
        "line": (args.line_world_asset, "/World/LineWorld"),
        "combined": (args.combined_course_asset, "/World/CombinedCourse"),
    }
    if args.world == "flat":
        world.scene.add_default_ground_plane()
    elif args.world in world_assets:
        asset_path, prim_path = world_assets[args.world]
        if not asset_path.exists():
            raise FileNotFoundError(asset_path)
        add_reference_to_stage(usd_path=str(asset_path.resolve()), prim_path=prim_path)

    # The built USD worlds do not embed a light source, so non-flat scenes
    # render pitch-black until we add one. Use a distant light + a low dome.
    if args.world != "flat":
        from pxr import UsdLux  # noqa: PLC0415

        distant = UsdLux.DistantLight.Define(world.stage, "/World/DefaultDistantLight")
        distant.CreateIntensityAttr(3000.0)
        distant.CreateAngleAttr(2.0)
        UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set((-45.0, 0.0, 30.0))

        dome = UsdLux.DomeLight.Define(world.stage, "/World/DefaultDomeLight")
        dome.CreateIntensityAttr(800.0)
    add_reference_to_stage(usd_path=str(args.asset.resolve()), prim_path="/World/BalanceCar")
    disable_joint_drives(world.stage)
    root_prim = world.stage.GetPrimAtPath("/World/BalanceCar")
    root_xformable = UsdGeom.Xformable(root_prim)
    if args.car_start_x_m or args.car_start_y_m or args.car_start_z_m:
        root_xformable.AddTranslateOp().Set(
            (float(args.car_start_x_m), float(args.car_start_y_m), float(args.car_start_z_m))
        )
    if args.car_start_yaw_deg:
        root_xformable.AddRotateZOp().Set(float(args.car_start_yaw_deg))
    if args.initial_pitch_deg:
        root_xformable.AddRotateYOp().Set(args.initial_pitch_deg)

    robot = world.scene.add(Articulation(prim_paths_expr=BASE_PATH, name="balance_car"))
    world.reset()

    # Optional: track the seesaw plank's live transform so we can log its
    # actual rotation to the CSV. Lets us definitively see whether the plank
    # is flipping (and at what tilt angle) instead of relying on visual
    # interpretation of the GUI viewport.
    seesaw_plank_prim = None
    if args.world == "combined":
        seesaw_plank_prim = world.stage.GetPrimAtPath("/World/CombinedCourse/seesaw_plank")
        if not seesaw_plank_prim or not seesaw_plank_prim.IsValid():
            seesaw_plank_prim = None

    if args.command_profile == "teleop" and not args.headless:
        setup_teleop_keyboard()
    elif args.command_profile == "teleop" and args.headless:
        print("TELEOP_REQUIRES_GUI please remove --headless to use --command-profile teleop", flush=True)

    if not args.headless:
        try:
            from isaacsim.core.utils.viewports import set_camera_view

            # For combined / line worlds the course is long along +X; pull the
            # camera back far enough to see the whole thing. For other worlds
            # frame the cart at its start.
            if args.world in ("combined", "line"):
                eye = (5.0, -8.0, 4.0)
                target = (5.0, 0.0, 0.15)
            else:
                eye = (
                    float(args.car_start_x_m) - 1.5,
                    -2.0,
                    float(args.car_start_z_m) + 0.8,
                )
                target = (
                    float(args.car_start_x_m),
                    0.0,
                    float(args.car_start_z_m) + 0.1,
                )
            set_camera_view(eye=eye, target=target)
        except Exception as cam_err:  # noqa: BLE001
            print(f"CAMERA_SETUP_WARN {cam_err}", flush=True)

    missing = [name for name in WHEEL_JOINT_NAMES if name not in list(robot.joint_names)]
    if missing:
        raise RuntimeError(f"Missing wheel joints {missing}; available joints: {list(robot.joint_names)}")

    state = State()
    # Sample IMU bias once at startup. Real MPU6050 has a persistent
    # turn-on bias that the controller must operate through. We sample
    # from N(0, bias_sigma) so each run sees a different bias, matching
    # the variability between physical units / sessions.
    _rng = np.random.default_rng()
    if args.imu_gyro_bias_deg_s > 0.0:
        bias_pitch_dps = _rng.normal(0.0, args.imu_gyro_bias_deg_s)
        bias_yaw_dps = _rng.normal(0.0, args.imu_gyro_bias_deg_s)
        state.imu_gyro_pitch_bias_rad_s = math.radians(bias_pitch_dps)
        state.imu_gyro_yaw_bias_rad_s = math.radians(bias_yaw_dps)
        print(
            f"IMU_BIAS_SAMPLED pitch_bias_dps={bias_pitch_dps:+.4f} "
            f"yaw_bias_dps={bias_yaw_dps:+.4f}",
            flush=True,
        )
    # Cache baseline values so --adaptive-slope-boost can interpolate between
    # baseline (flat-ground, safe) and boost (slope, aggressive forward lean).
    baseline_drive_pitch_offset_deg = args.drive_pitch_offset_deg
    baseline_teleop_velocity_m_s = args.teleop_velocity_m_s
    rows: list[dict] = []
    max_abs_pitch = 0.0
    max_abs_roll = 0.0
    fell_step: int | None = None
    boost_factor = 0.0

    # Streaming CSV writer — opens the file BEFORE the loop starts and flushes
    # each second so that pressing X on the GUI window doesn't lose data.
    # csv_writer / csv_file are lazily created on the first row (we need to know
    # the row dict's keys for the header).
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    csv_file = None
    csv_writer = None
    flush_every_n = max(1, int(args.physics_hz))  # flush ~once per second
    _atexit_registered = False

    def _dump_summary() -> None:
        """Best-effort write of the JSON report. Safe to call multiple times
        (later calls overwrite). Used both by the normal flow and by the
        atexit hook in case the user kills the GUI window mid-loop."""
        if not rows:
            return
        try:
            f = rows[-1]
            summary = {
                "ok": fell_step is None,
                "asset_path": str(args.asset.resolve()),
                "steps_completed": len(rows),
                "fell_step": fell_step,
                "max_abs_pitch_deg": max_abs_pitch,
                "max_abs_roll_deg": max_abs_roll,
                "final_x_m": float(f["x_m"]),
                "final_y_m": float(f["y_m"]),
                "final_z_m": float(f["z_m"]),
                "final_pitch_deg": float(f["pitch_deg"]),
                "final_roll_deg": float(f["roll_deg"]),
                "max_x_m": max(float(r["x_m"]) for r in rows),
                "controller": args.controller,
                "command_profile": args.command_profile,
                "world": args.world,
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _close_csv() -> None:
        nonlocal csv_file
        try:
            if csv_file is not None:
                csv_file.flush()
                csv_file.close()
                csv_file = None
        except Exception:
            pass

    import atexit
    atexit.register(lambda: (_close_csv(), _dump_summary()))
    lqr_gain: np.ndarray | None = None
    if args.controller == "lqr":
        lqr_gain, _, _ = build_lqr_gain(dt)
        print(f"LQR_GAIN K={lqr_gain.tolist()}", flush=True)
    elif args.controller == "lqi":
        lqr_gain, _, _ = build_lqi_gain(dt)
        print(f"LQI_GAIN K={lqr_gain.tolist()}", flush=True)
    follow_cam_func = None
    if args.follow_camera and not args.headless:
        try:
            from isaacsim.core.utils.viewports import set_camera_view as _scv
            follow_cam_func = _scv
        except Exception as cam_err:  # noqa: BLE001
            print(f"FOLLOW_CAMERA_WARN {cam_err}", flush=True)

    g_const = 9.81

    for step in range(args.steps):
        if args.command_profile == "teleop" and not args.headless:
            poll_teleop_keyboard()
        matrix = world_transform(world.stage, BASE_PATH)
        pos = matrix.ExtractTranslation()
        pitch, yaw, roll = body_angles(matrix)

        # Slope estimate.
        # 1) Heavily filtered pitch tracks the slope once the cart is ON it,
        #    but lags badly when entering a slope. So we ALSO consult the
        #    known geometry when --world combined is used.
        state.slope_est_rad = (
            state.slope_est_rad * (1.0 - args.slope_est_alpha)
            + pitch * args.slope_est_alpha
        )
        # 2) Geometry-based slope from cart x position (combined course only).
        # Matches create_combined_course.py current layout:
        #   ramp_up      x in [2.30, 2.93]  → +3°    (gentle bump up)
        #   ramp_down    x in [3.43, 4.07]  → -3°    (gentle bump down)
        #   seesaw plank x in [5.4, 6.86]   → +1.91° (climbing plank pre-pivot)
        #   seesaw plank x in [6.86, 8.36]  → -1.91° (descending plank post-flip)
        # The seesaw geom overrides matter because: when the cart sits with a
        # deliberate forward lean (drive_pitch_offset_deg), the IMU-filter
        # eventually tracks that lean, fooling the slope estimator into
        # reading a fake downslope and applying BACKWARD feed-forward — which
        # exactly cancels the forward command and stalls the cart.
        slope_geom_rad = 0.0
        if args.world == "combined":
            cart_x = float(pos[0])
            if 2.30 <= cart_x < 2.93:
                slope_geom_rad = math.radians(3.0)
            elif 3.43 <= cart_x < 4.07:
                slope_geom_rad = math.radians(-3.0)
            elif 5.40 <= cart_x < 6.86:
                slope_geom_rad = math.radians(1.91)
            elif 6.86 <= cart_x < 8.36:
                slope_geom_rad = math.radians(-1.91)
        # Effective slope: if we have a known-geometry override for the
        # cart's current x position (in the combined course's pre-mapped
        # zones), TRUST the geometry — the IMU filter is corrupted by the
        # cart's deliberate forward lean and will report a fake downslope.
        # Outside the mapped zones, fall back to the IMU filter.
        if slope_geom_rad != 0.0:
            slope_used_rad = slope_geom_rad
        else:
            slope_used_rad = state.slope_est_rad
        state.slope_used_rad = slope_used_rad

        # Adaptive slope boost: smoothly raise drive-pitch-offset and teleop
        # velocity setpoint while on a detected slope, then drop back to the
        # safer baseline on flat ground. boost_factor is a 0..1 ramp from
        # threshold to full-deg, applied to the magnitude of the slope.
        if args.adaptive_slope_boost:
            slope_abs_deg = abs(math.degrees(slope_used_rad))
            span = max(args.boost_slope_full_deg - args.boost_slope_threshold_deg, 1e-3)
            boost_factor = max(0.0, min(1.0, (slope_abs_deg - args.boost_slope_threshold_deg) / span))
            args.drive_pitch_offset_deg = (
                baseline_drive_pitch_offset_deg
                + boost_factor * (args.boost_pitch_offset_deg - baseline_drive_pitch_offset_deg)
            )
            args.teleop_velocity_m_s = (
                baseline_teleop_velocity_m_s
                + boost_factor * (args.boost_velocity_m_s - baseline_teleop_velocity_m_s)
            )

        if follow_cam_func is not None and step % 4 == 0:
            cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
            follow_cam_func(
                eye=(cx - 2.0, cy - 2.5, cz + 1.0),
                target=(cx + 0.3, cy, cz + 0.05),
            )
        if state.previous_pitch is None:
            pitch_rate = 0.0
            yaw_rate = 0.0
        else:
            pitch_rate = (pitch - state.previous_pitch) / dt
            yaw_delta = math.atan2(math.sin(yaw - state.previous_yaw), math.cos(yaw - state.previous_yaw))
            yaw_rate = yaw_delta / dt

        joint_positions = np.asarray(robot.get_joint_positions(joint_names=WHEEL_JOINT_NAMES), dtype=float).reshape(-1)
        joint_velocities = np.asarray(robot.get_joint_velocities(joint_names=WHEEL_JOINT_NAMES), dtype=float).reshape(-1)
        wheel_position_m = 0.5 * float(joint_positions[0] + joint_positions[1]) * WHEEL_RADIUS_M
        wheel_velocity_m_s = 0.5 * float(joint_velocities[0] + joint_velocities[1]) * WHEEL_RADIUS_M
        wheel_yaw_rate = (float(joint_velocities[1] - joint_velocities[0]) * WHEEL_RADIUS_M) / WHEEL_TRACK_M

        target_velocity, target_yaw_rate = command_profile(step, dt)
        target_pitch = 0.0
        control_debug: dict = {}

        # ---- Sensor noise / quantization injection (high-fidelity sim2real) ----
        # Replace the perfect ground-truth sensor reads with noisy / quantized
        # versions BEFORE the controller sees them. Keeps `pitch`, `pitch_rate`,
        # etc. as truth for logging and slope estimation; the controller gets
        # the *_obs (observed) versions. Defaults are 0/off so behaviour is
        # unchanged unless the user opts in.
        pitch_obs = pitch
        pitch_rate_obs = pitch_rate
        yaw_rate_obs = yaw_rate
        joint_positions_obs = joint_positions
        if args.imu_pitch_noise_deg > 0.0:
            pitch_obs = pitch + math.radians(
                _rng.normal(0.0, args.imu_pitch_noise_deg)
            )
        if args.imu_gyro_noise_deg_s > 0.0 or args.imu_gyro_bias_deg_s > 0.0:
            pitch_rate_obs = (
                pitch_rate
                + state.imu_gyro_pitch_bias_rad_s
                + (math.radians(_rng.normal(0.0, args.imu_gyro_noise_deg_s))
                   if args.imu_gyro_noise_deg_s > 0.0 else 0.0)
            )
            yaw_rate_obs = (
                yaw_rate
                + state.imu_gyro_yaw_bias_rad_s
                + (math.radians(_rng.normal(0.0, args.imu_gyro_noise_deg_s))
                   if args.imu_gyro_noise_deg_s > 0.0 else 0.0)
            )
        if args.encoder_quantize:
            # Hall encoder produces discrete counts. Quantize the joint
            # position (radians) to the nearest count by rounding to a
            # 2pi/CPR step. The residual carries forward so over many
            # ticks the controller sees the right average position.
            rad_per_count = 2.0 * math.pi / max(args.encoder_cpr, 1.0)
            left_raw = float(joint_positions[0])
            right_raw = float(joint_positions[1])
            left_quant = round(left_raw / rad_per_count) * rad_per_count
            right_quant = round(right_raw / rad_per_count) * rad_per_count
            joint_positions_obs = np.array([left_quant, right_quant], dtype=float)

        if args.controller == "yahboom":
            left_effort, right_effort, control_debug = update_yahboom_controller(
                state,
                step,
                dt,
                pitch_obs,
                pitch_rate_obs,
                yaw_rate_obs,
                joint_positions_obs,
            )
        elif args.controller == "threering":
            left_effort, right_effort, control_debug = update_threering_controller(
                state,
                step,
                dt,
                pitch_obs,
                pitch_rate_obs,
                yaw_rate_obs,
                joint_positions_obs,
            )
        elif args.controller == "lqr":
            assert lqr_gain is not None
            left_effort, right_effort, control_debug = update_lqr_controller(
                state,
                step,
                dt,
                pitch_obs,
                pitch_rate_obs,
                yaw_rate_obs,
                joint_positions_obs,
                joint_velocities,
                lqr_gain,
            )
        elif args.controller == "lqi":
            assert lqr_gain is not None
            left_effort, right_effort, control_debug = update_lqi_controller(
                state,
                step,
                dt,
                pitch_obs,
                pitch_rate_obs,
                yaw_rate_obs,
                joint_positions_obs,
                joint_velocities,
                lqr_gain,
            )
        else:
            left_effort, right_effort, target_pitch, control_debug = update_legacy_controller(
                state,
                step,
                dt,
                pitch_obs,
                pitch_rate_obs,
                joint_positions_obs,
                joint_velocities,
            )

        line_follow_debug = {}
        if args.line_follow:
            lateral_error = float(pos[1]) - args.line_follow_y
            correction_raw = args.line_follow_kp * lateral_error + args.line_follow_kd * yaw_rate
            # Cap the yaw correction by the balance-loop's spare effort, so the
            # symmetric pitch-control torque is preserved. If the balance loop
            # is already near saturation, the line follower steps aside.
            base_max = max(abs(left_effort), abs(right_effort))
            headroom = max(0.0, args.effort_limit - base_max)
            correction = clamp(correction_raw, headroom)
            left_effort_pre = left_effort
            right_effort_pre = right_effort
            left_effort = clamp(left_effort - correction, args.effort_limit)
            right_effort = clamp(right_effort + correction, args.effort_limit)
            line_follow_debug = {
                "line_lateral_error_m": lateral_error,
                "line_correction_nm": correction,
                "line_correction_raw_nm": correction_raw,
                "line_headroom_nm": headroom,
                "line_left_effort_pre": left_effort_pre,
                "line_right_effort_pre": right_effort_pre,
            }

        # Slope feed-forward: add a gravity-compensation torque to BOTH wheels
        # symmetrically (does not affect yaw). On a +X up-slope this pushes the
        # cart up; on a -X up-slope (cart facing wrong way) it pushes back.
        slope_ff_per_wheel = 0.0
        if args.slope_feedforward:
            slope_ff_total_torque = (
                args.slope_ff_total_mass_kg * g_const * math.sin(slope_used_rad) * WHEEL_RADIUS_M
            )
            slope_ff_per_wheel = slope_ff_total_torque * 0.5
            left_effort = clamp(left_effort + slope_ff_per_wheel, args.effort_limit)
            right_effort = clamp(right_effort + slope_ff_per_wheel, args.effort_limit)

        # Motor back-EMF damping. A real DC motor's torque drops with speed
        # (back-EMF = K * omega). Without modelling this, sim wheels accelerate
        # unrealistically fast under any commanded torque, which turns a small
        # turn-differential into a violent yaw spike that flips the cart. The
        # damping coefficient is chosen so max wheel ω ≈ 30 rad/s (~1 m/s cart
        # speed), matching the real Yahboom motor's stall behaviour.
        if args.wheel_damping_coef > 0.0:
            left_omega = float(joint_velocities[0])
            right_omega = float(joint_velocities[1])
            left_effort = clamp(left_effort - args.wheel_damping_coef * left_omega, args.effort_limit)
            right_effort = clamp(right_effort - args.wheel_damping_coef * right_omega, args.effort_limit)

        robot.set_joint_efforts(np.array([[left_effort, right_effort]], dtype=float), joint_names=WHEEL_JOINT_NAMES)
        world.step(render=not args.headless)

        pitch_deg = math.degrees(pitch)
        roll_deg = math.degrees(roll)
        max_abs_pitch = max(max_abs_pitch, abs(pitch_deg))
        max_abs_roll = max(max_abs_roll, abs(roll_deg))
        if fell_step is None and (abs(pitch_deg) > 35.0 or abs(roll_deg) > 35.0):
            fell_step = step
            if args.headless:
                break

        row = {
            "step": step,
            "time_s": step * dt,
            "x_m": float(pos[0]),
            "y_m": float(pos[1]),
            "z_m": float(pos[2]),
            "pitch_deg": pitch_deg,
            "pitch_rate_deg_s": math.degrees(pitch_rate),
            "target_pitch_deg": math.degrees(target_pitch),
            "roll_deg": roll_deg,
            "yaw_deg": math.degrees(yaw),
            "yaw_rate_deg_s": math.degrees(yaw_rate),
            "slope_est_deg": math.degrees(state.slope_est_rad),
            "slope_used_deg": math.degrees(slope_used_rad),
            "boost_factor": boost_factor,
            "effective_drive_pitch_offset_deg": args.drive_pitch_offset_deg,
            "effective_teleop_velocity_m_s": args.teleop_velocity_m_s,
            "slope_ff_per_wheel_nm": slope_ff_per_wheel,
            "wheel_position_m": wheel_position_m,
            "wheel_velocity_m_s": wheel_velocity_m_s,
            "target_position_m": state.target_position_m,
            "target_velocity_m_s": target_velocity,
            "target_yaw_rate_rad_s": target_yaw_rate,
            "mpu6050_pitch_deg": pitch_deg,
            "mpu6050_gyro_y_deg_s": math.degrees(pitch_rate),
            "hall_left_position_rad": float(joint_positions[0]),
            "hall_right_position_rad": float(joint_positions[1]),
            "hall_left_velocity_rad_s": float(joint_velocities[0]),
            "hall_right_velocity_rad_s": float(joint_velocities[1]),
            "left_effort": left_effort,
            "right_effort": right_effort,
            "controller": args.controller,
            "angle_output_pwm": float(control_debug.get("angle_output_pwm", 0.0)),
            "speed_output_pwm": float(control_debug.get("speed_output_pwm", 0.0)),
            "turn_output_pwm": float(control_debug.get("turn_output_pwm", 0.0)),
            "speed_filter_pulses": float(control_debug.get("speed_filter_pulses", 0.0)),
            "position_pulses": float(control_debug.get("position_pulses", 0.0)),
            "pwm_left": float(control_debug.get("pwm_left", left_effort / max(args.effort_limit, 1.0e-9) * args.pwm_limit)),
            "pwm_right": float(control_debug.get("pwm_right", right_effort / max(args.effort_limit, 1.0e-9) * args.pwm_limit)),
            # Actual drive_offset USED by the yahboom controller this step
            # (0 when no W/S held; ±drive_pitch_offset_deg when held). Distinct
            # from effective_drive_pitch_offset_deg which is just the CLI param.
            "yahboom_drive_offset_deg": float(control_debug.get("drive_offset_deg", 0.0)),
            # Teleop key state at this physics tick — lets us see whether the
            # user's WASD inputs were actually being received by the polling
            # loop (vs. being eaten by Isaac Sim's hotkey system).
            "key_forward": int(TELEOP_KEYS.get("forward", False)) if args.command_profile == "teleop" else 0,
            "key_back": int(TELEOP_KEYS.get("back", False)) if args.command_profile == "teleop" else 0,
            "key_left": int(TELEOP_KEYS.get("left", False)) if args.command_profile == "teleop" else 0,
            "key_right": int(TELEOP_KEYS.get("right", False)) if args.command_profile == "teleop" else 0,
        }

        # Seesaw plank state (combined course only). Reads the plank's live
        # world transform every tick so we can definitively log whether the
        # plank flipped, at what angle, and when. plank_pitch_deg > 0 means
        # exit-side DOWN (flipped state); < 0 means entry-side DOWN (resting).
        if seesaw_plank_prim is not None:
            try:
                plank_xformable = UsdGeom.Xformable(seesaw_plank_prim)
                plank_local_to_world = plank_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                # Translation: column 3 of the row-major 4x4 (USD convention).
                plank_world_z = float(plank_local_to_world[3][2])
                # Rotation: extract Y-axis pitch from the rotation matrix.
                # USD row-vector RotateY by theta puts cos(theta) at [0][0]
                # and sin(theta) at [2][0]. So pitch = atan2(m[2][0], m[0][0]).
                m00 = float(plank_local_to_world[0][0])
                m20 = float(plank_local_to_world[2][0])
                plank_pitch_rad = math.atan2(m20, m00)
                row["plank_pitch_deg"] = math.degrees(plank_pitch_rad)
                row["plank_world_z"] = plank_world_z
            except Exception:
                row["plank_pitch_deg"] = 0.0
                row["plank_world_z"] = 0.0
        else:
            row["plank_pitch_deg"] = 0.0
            row["plank_world_z"] = 0.0
        rows.append(row)

        # Stream the row to disk so closing the GUI window doesn't lose data.
        if csv_file is None:
            csv_file = args.csv.open("w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
            csv_writer.writeheader()
        csv_writer.writerow(row)
        if step % flush_every_n == 0:
            csv_file.flush()
            # Also refresh the JSON summary so an external watcher / the user
            # who killed the window can read where the cart ended up.
            _dump_summary()
            # Diagnostic heartbeat so we can see if teleop keys are being
            # received without spamming every tick.
            if args.command_profile == "teleop" and not args.headless:
                pressed = [k for k, v in TELEOP_KEYS.items() if v]
                print(
                    f"HEARTBEAT t={step*dt:.1f}s x={float(pos[0]):.2f} y={float(pos[1]):.3f} "
                    f"pitch={pitch_deg:+.1f} roll={roll_deg:+.1f} vel={wheel_velocity_m_s:+.3f} "
                    f"keys={pressed if pressed else 'NONE'}",
                    flush=True,
                )

        state.previous_pitch = pitch
        state.previous_yaw = yaw

    # Normal end of loop: close the streaming file. The atexit hook will also
    # call _close_csv() but calling it here gives a clean shutdown path.
    _close_csv()

    final = rows[-1]
    result = {
        "ok": fell_step is None,
        "asset_path": str(args.asset.resolve()),
        "steps_requested": args.steps,
        "steps_completed": len(rows),
        "physics_hz": args.physics_hz,
        "fell_step": fell_step,
        "max_abs_pitch_deg": max_abs_pitch,
        "max_abs_roll_deg": max_abs_roll,
        "controller": {
            "type": f"pure_physics_{args.controller}_pid_effort",
            "pose_assist": False,
            "mode": args.controller,
            "effort_limit": args.effort_limit,
            "effort_sign": args.effort_sign,
            "balance_kp": args.balance_kp,
            "balance_kd": args.balance_kd,
            "speed_kp": args.speed_kp,
            "position_kp": args.position_kp,
            "yaw_kp": args.yaw_kp,
            "loop_period_ms": args.loop_period_ms,
            "speed_loop_divisor": args.speed_loop_divisor,
            "turn_loop_divisor": args.turn_loop_divisor,
            "encoder_cpr": args.encoder_cpr,
            "angle_kp": args.angle_kp,
            "angle_kd": args.angle_kd,
            "speed_p": args.speed_p,
            "speed_i": args.speed_i,
            "turn_p": args.turn_p,
            "turn_d": args.turn_d,
            "pwm_limit": args.pwm_limit,
            "fall_angle_deg": args.fall_angle_deg,
        },
        "final": final,
        "csv_path": str(args.csv.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


try:
    result = run()
    status = "PID_EFFORT_OK" if result["ok"] else "PID_EFFORT_FELL"
    print(
        f"{status} steps={result['steps_completed']} max_abs_pitch={result['max_abs_pitch_deg']:.2f} "
        f"final_pitch={result['final']['pitch_deg']:.2f} fell_step={result['fell_step']}",
        flush=True,
    )
finally:
    simulation_app.close()
