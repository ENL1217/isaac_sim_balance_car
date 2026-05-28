"""Run an honest pass/fail matrix for each obstacle on combined_course.

For each (controller, obstacle) pair: spawn cart at the start of the obstacle,
drive forward for fixed time, and record whether the cart reached the other
side WITHOUT falling. Output a single JSON / markdown summary.

Run with:
    D:\\isaac\\isaacsim\\python.bat sim\\scripts\\run_obstacle_matrix.py

This script invokes play_pid_effort.py as a subprocess for each test so we get
a clean Isaac Sim instance per case (no shared state pollution).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("D:/isaac/isaacsim/python.bat")
PLAY = REPO / "scripts" / "play_pid_effort.py"
OUT = REPO / "output" / "obstacle_matrix"
OUT.mkdir(parents=True, exist_ok=True)

# Obstacle definitions: starting position, target x, max duration.
OBSTACLES = [
    {
        "name": "flat_drive",
        "start_x": -1.0,
        "start_z": 0.0,
        "target_x": -0.5,
        "max_t": 12.0,
        "comment": "stand + brief drive on start_zone",
        "command_profile": "drive_demo",
    },
    {
        "name": "climb_slope_15deg",
        "start_x": -0.5,
        "start_z": 0.0,
        "target_x": 0.8,  # past the ramp (ends ~0.56) onto top platform
        "max_t": 25.0,
        "comment": "climb 15 deg ramp (uniform slope, bevel == main angle)",
        "command_profile": "climb",
    },
    {
        "name": "descend_stairs",
        "start_x": 1.5,
        "start_z": 0.14,  # top platform z=0.136
        "target_x": 3.5,
        "max_t": 20.0,
        "comment": "from top platform, descend 4 stairs of 0.034 m",
        "command_profile": "climb",
    },
    {
        "name": "cross_seesaw",
        "start_x": 4.3,
        "start_z": 0.0,
        "target_x": 6.7,
        "max_t": 25.0,
        "comment": "drive onto tilted plank (4.6 deg tilt), cross pivot",
        "command_profile": "climb",
    },
    {
        "name": "traverse_gravel",
        "start_x": 7.2,
        "start_z": 0.0,
        "target_x": 10.4,
        "max_t": 25.0,
        "comment": "drive across 3m of gravel (max stone 12 mm)",
        "command_profile": "climb",
    },
    {
        "name": "full_course",
        "start_x": -1.0,
        "start_z": 0.0,
        "target_x": 12.0,
        "max_t": 90.0,
        "comment": "end-to-end traversal",
        "command_profile": "climb",
    },
]

CONTROLLERS = [
    {
        "name": "yahboom_ff",
        "args": [
            "--controller", "yahboom",
            "--drive-pitch-offset-deg", "4.0",
            "--slope-feedforward",
            "--slope-ff-total-mass-kg", "2.5",
        ],
    },
    {
        "name": "lqr_ff",
        "args": [
            "--controller", "lqr",
            "--lqr-target-velocity-scale", "0.4",
            "--lqr-q-vel", "30.0",
            "--slope-feedforward",
            "--slope-ff-total-mass-kg", "2.0",
        ],
    },
    {
        "name": "lqi_ff",
        "args": [
            "--controller", "lqi",
            "--lqr-target-velocity-scale", "0.4",
            "--lqr-q-vel", "30.0",
            "--lqi-q-integral", "3.0",
            "--lqi-integral-clamp", "1.0",
            "--slope-feedforward",
            "--slope-ff-total-mass-kg", "2.0",
        ],
    },
]


def run_one(ctrl: dict, obs: dict) -> dict:
    physics_hz = 240
    steps = int(obs["max_t"] * physics_hz)
    tag = f"{ctrl['name']}__{obs['name']}"
    report = OUT / f"{tag}.json"
    csv_path = OUT / f"{tag}.csv"
    cmd = [
        str(PYTHON),
        str(PLAY),
        "--headless",
        "--world", "combined",
        "--car-start-x-m", str(obs["start_x"]),
        "--car-start-z-m", str(obs["start_z"]),
        "--command-profile", obs["command_profile"],
        "--steps", str(steps),
        "--report", str(report),
        "--csv", str(csv_path),
    ] + ctrl["args"]
    print(f"-- {tag}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    summary = {
        "controller": ctrl["name"],
        "obstacle": obs["name"],
        "target_x": obs["target_x"],
        "stdout_last": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "",
        "rc": proc.returncode,
    }
    if report.exists():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            summary.update({
                "steps_completed": data.get("steps_completed"),
                "max_abs_pitch": data.get("max_abs_pitch_deg"),
                "max_abs_roll": data.get("max_abs_roll_deg"),
                "fell_step": data.get("fell_step"),
                "final_x": data.get("final", {}).get("x_m"),
                "final_z": data.get("final", {}).get("z_m"),
            })
        except Exception as e:
            summary["json_err"] = str(e)
    # Read CSV to find peak x to test if target reached
    try:
        import csv as csvmod
        rows = list(csvmod.DictReader(open(csv_path, encoding="utf-8")))
        if rows:
            xs = [float(r["x_m"]) for r in rows]
            summary["peak_x"] = max(xs)
            summary["min_x"] = min(xs)
            summary["passed"] = summary["peak_x"] >= obs["target_x"] and summary.get("fell_step") is None
    except Exception as e:
        summary["csv_err"] = str(e)
        summary["passed"] = False
    return summary


def main() -> None:
    results: list[dict] = []
    for obs in OBSTACLES:
        for ctrl in CONTROLLERS:
            try:
                r = run_one(ctrl, obs)
            except subprocess.TimeoutExpired:
                r = {"controller": ctrl["name"], "obstacle": obs["name"], "error": "timeout"}
            results.append(r)
            tag = f"{r['controller']}__{r['obstacle']}"
            passed = r.get("passed")
            fell = r.get("fell_step")
            peak = r.get("peak_x")
            print(f"   -> passed={passed} fell={fell} peak_x={peak}", flush=True)

    summary_path = OUT / "_matrix.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Print markdown table
    print()
    print("| Obstacle | Controller | Passed | Fell | Peak x | Max pitch | Max roll |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        passed = "Y" if r.get("passed") else "N"
        fell = r.get("fell_step", "-")
        peak = r.get("peak_x")
        peak_str = f"{peak:+.2f}" if isinstance(peak, (int, float)) else "-"
        mp = r.get("max_abs_pitch")
        mr = r.get("max_abs_roll")
        mp_str = f"{mp:.1f}" if isinstance(mp, (int, float)) else "-"
        mr_str = f"{mr:.1f}" if isinstance(mr, (int, float)) else "-"
        print(f"| {r['obstacle']} | {r['controller']} | {passed} | {fell} | {peak_str} | {mp_str} | {mr_str} |")


if __name__ == "__main__":
    main()
