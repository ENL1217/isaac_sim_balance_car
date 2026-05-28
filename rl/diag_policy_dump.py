"""Dump what the trained policy actually outputs for various obs."""
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
        max_episode_steps=600,
        init_pitch_range_deg=(0.0, 0.0),
        init_x_range_m=(0.0, 0.0),
        target_velocity_range_m_s=(0.10, 0.10),
        fixed_target_velocity=0.10,
    )
    env = BalanceCarEnv(cfg)
    print("DIAG_ENV_READY", flush=True)

    from stable_baselines3 import PPO
    model = PPO.load(str(REPO / "rl" / "diag_fixed_pos.zip"), env=None)
    print("MODEL_LOADED", flush=True)

    # Test 1: cart at rest, target=+0.10. What does policy output?
    obs, _ = env.reset(seed=0)
    print(f"INIT_OBS={[float(x) for x in obs]}", flush=True)
    for step in range(20):
        action, _ = model.predict(obs, deterministic=True)
        print(f"step={step} obs7(target)={obs[7]:+.3f} obs0(pitch)={obs[0]:+.3f} obs6(x_dot)={obs[6]:+.3f} action=[{action[0]:+.3f},{action[1]:+.3f}]", flush=True)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break

    # Test 2: manually feed positive obs[7] target with zeroed cart state
    print("\n--- TEST 2: manual obs ---", flush=True)
    for tv_norm in (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75):
        # Construct fake obs: zero state, varying target
        fake_obs = np.zeros(8, dtype=np.float32)
        fake_obs[7] = tv_norm
        action, _ = model.predict(fake_obs, deterministic=True)
        print(f"target_norm={tv_norm:+.2f} (raw target={tv_norm*0.2:+.3f}) → action=[{action[0]:+.3f},{action[1]:+.3f}]", flush=True)


try:
    main()
finally:
    _sim_app.close()
