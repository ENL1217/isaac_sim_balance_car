"""Sanity test: with velocity action mode, does positive action drive cart +X?"""
from __future__ import annotations
import sys
from pathlib import Path

from isaacsim import SimulationApp
_sim_app = SimulationApp({"headless": True})

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from balance_env import BalanceCarEnv, BalanceEnvConfig

REPO = Path(__file__).resolve().parents[1]
ASSET = REPO / "sim" / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda"


def main():
    cfg = BalanceEnvConfig(
        asset_path=ASSET,
        world_asset_path=None,
        use_default_ground=True,
        max_episode_steps=500,
        init_pitch_range_deg=(0.0, 0.0),
        init_x_range_m=(0.0, 0.0),
        target_velocity_range_m_s=(0.0, 0.0),
        fixed_target_velocity=0.0,
        action_mode="velocity",
    )
    env = BalanceCarEnv(cfg)
    print("DIAG_ENV_READY mode=velocity", flush=True)

    # Constant +0.3 action (target wheel vel = +0.3*30 = +9 rad/s = +0.306 m/s)
    print("\n--- T1: action=+0.3 (target ~+0.3 m/s) ---", flush=True)
    obs, _ = env.reset(seed=0)
    print("T1_RESET_OK", flush=True)
    pos_action = np.array([+0.3, +0.3], dtype=np.float32)
    for step in range(60):
        obs, _, term, trunc, info = env.step(pos_action)
        print(f"step={step:3d} x_w={info['x_w']:+.4f} pitch={info['pitch_deg']:+.2f} roll={info['roll_deg']:+.2f} fell={info['fell']}", flush=True)
        if term:
            break

    # Constant -0.3 action
    print("\n--- T2: action=-0.3 (target ~-0.3 m/s) ---", flush=True)
    obs, _ = env.reset(seed=0)
    neg_action = np.array([-0.3, -0.3], dtype=np.float32)
    for step in range(300):
        obs, _, term, trunc, info = env.step(neg_action)
        if step in (5, 20, 50, 100, 200, 299) or term:
            print(f"step={step:3d} x_w={info['x_w']:+.4f} pitch={info['pitch_deg']:+.2f} roll={info['roll_deg']:+.2f} fell={info['fell']}", flush=True)
        if term:
            break

    # Zero action — should stay still (modulo balance)
    print("\n--- T3: action=0 (target 0) ---", flush=True)
    obs, _ = env.reset(seed=0)
    zero_action = np.array([0.0, 0.0], dtype=np.float32)
    for step in range(300):
        obs, _, term, trunc, info = env.step(zero_action)
        if step in (5, 50, 200, 299) or term:
            print(f"step={step:3d} x_w={info['x_w']:+.4f} pitch={info['pitch_deg']:+.2f} fell={info['fell']}", flush=True)
        if term:
            break

    print("\nDIAG_DONE", flush=True)


try:
    main()
finally:
    _sim_app.close()
