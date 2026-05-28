"""Inspect drive_demo CSV at key timestamps."""
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sim/output/pid_smoke_drive_3000.csv")
rows = list(csv.DictReader(open(path)))
times = (0.5, 1.5, 2.5, 3.5, 5.5, 7.0, 8.5, 11.0)
for t_target in times:
    s = int(t_target * 240)
    if s >= len(rows):
        continue
    r = rows[s]
    print(
        f"t={float(r['time_s']):5.2f}s "
        f"pitch={float(r['pitch_deg']):+7.3f} "
        f"tgt_p={float(r['target_pitch_deg']):+6.3f} "
        f"x={float(r['x_m']):+7.4f} "
        f"yaw={float(r['yaw_deg']):+7.3f} "
        f"wv={float(r['wheel_velocity_m_s']):+7.4f} "
        f"pos_p={float(r['position_pulses']):+7.2f} "
        f"A_pwm={float(r['angle_output_pwm']):+8.3f} "
        f"S_pwm={float(r['speed_output_pwm']):+8.3f} "
        f"T_pwm={float(r['turn_output_pwm']):+8.3f} "
        f"pwmL={float(r['pwm_left']):+7.2f} "
        f"pwmR={float(r['pwm_right']):+7.2f}"
    )

# Also report extremes during each window
windows = {
    "forward(1-3)": (1.0, 3.0),
    "back(4.5-6.5)": (4.5, 6.5),
    "turnL(6.5-8)": (6.5, 8.0),
    "turnR(8-9.5)": (8.0, 9.5),
}
print()
for name, (a, b) in windows.items():
    seg = [r for r in rows if a <= float(r["time_s"]) <= b]
    if not seg:
        continue
    xs = [float(r["x_m"]) for r in seg]
    yaws = [float(r["yaw_deg"]) for r in seg]
    wvs = [float(r["wheel_velocity_m_s"]) for r in seg]
    pp = [float(r["position_pulses"]) for r in seg]
    pwm_l = [float(r["pwm_left"]) for r in seg]
    s_pwm = [float(r["speed_output_pwm"]) for r in seg]
    print(
        f"{name:14s} "
        f"x_range=[{min(xs):+.4f}, {max(xs):+.4f}] "
        f"yaw_range=[{min(yaws):+.2f}, {max(yaws):+.2f}] "
        f"wv_max={max(wvs, key=abs):+.4f} "
        f"pos_p_max={max(pp, key=abs):+.2f} "
        f"S_pwm_max={max(s_pwm, key=abs):+.3f} "
        f"pwmL_max={max(pwm_l, key=abs):+.2f}"
    )
