"""Diagnose direction sign: apply constant +effort to both wheels, see which
way cart moves. Bypasses ALL policy/reward code."""

from __future__ import annotations
import sys
from pathlib import Path

from isaacsim import SimulationApp
_sim_app = SimulationApp({"headless": True})

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from balance_env import BalanceCarEnv, BalanceEnvConfig, WHEEL_RADIUS_M

REPO = Path(__file__).resolve().parents[1]
ASSET = REPO / "sim" / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda"


def main():
    cfg = BalanceEnvConfig(
        asset_path=ASSET,
        world_asset_path=None,
        use_default_ground=True,
        max_episode_steps=600,
        init_pitch_range_deg=(0.0, 0.0),  # ZERO init pitch
        init_x_range_m=(0.0, 0.0),         # ZERO init position
        target_velocity_range_m_s=(0.0, 0.0),
    )
    env = BalanceCarEnv(cfg)
    print("DIAG_ENV_READY", flush=True)

    # Test 1: constant +0.5 effort on both wheels (positive action)
    obs, _ = env.reset(seed=0)
    print(f"DIAG_T1_START obs={[float(x) for x in obs]}", flush=True)
    pos_action = np.array([+0.5, +0.5], dtype=np.float32)
    for step in range(500):
        obs, _, term, trunc, info = env.step(pos_action)
        if step in (1, 10, 50, 100, 200, 300, 499) or term:
            print(f"DIAG_T1 step={step} x_w={info['x_w']:+.4f} pitch={info['pitch_deg']:+.2f} fell={info['fell']}", flush=True)
        if term or trunc:
            break

    # Test 2: constant -0.5 effort on both wheels
    obs, _ = env.reset(seed=0)
    print(f"DIAG_T2_START obs={[float(x) for x in obs]}", flush=True)
    neg_action = np.array([-0.5, -0.5], dtype=np.float32)
    for step in range(500):
        obs, _, term, trunc, info = env.step(neg_action)
        if step in (1, 10, 50, 100, 200, 300, 499) or term:
            print(f"DIAG_T2 step={step} x_w={info['x_w']:+.4f} pitch={info['pitch_deg']:+.2f} fell={info['fell']}", flush=True)
        if term or trunc:
            break

    print("DIAG_DONE", flush=True)


try:
    main()
finally:
    _sim_app.close()
