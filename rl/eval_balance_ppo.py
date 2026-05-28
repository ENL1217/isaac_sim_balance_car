"""Evaluate a trained PPO model in Isaac Sim 5.0 standalone.

Run:
    D:\\isaac\\isaacsim\\python.bat rl\\eval_balance_ppo.py --model rl\\ppo_balance.zip --episodes 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_parser = argparse.ArgumentParser()
_parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "ppo_balance.zip")
_parser.add_argument("--world", choices=("flat", "combined"), default="flat")
_parser.add_argument("--episodes", type=int, default=5)
_parser.add_argument("--gui", action="store_true")
_parser.add_argument("--max-episode-steps", type=int, default=2400)
_parser.add_argument("--target-velocity", type=float, default=None,
                     help="Override the random target velocity (e.g. 0.15 for a steady forward drive).")
_cli_args = _parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402
_sim_app = SimulationApp({"headless": not _cli_args.gui})

sys.path.insert(0, str(Path(__file__).resolve().parent))
from balance_env import BalanceCarEnv, BalanceEnvConfig  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ASSET = REPO / "sim" / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda"
COMBINED = REPO / "sim" / "assets" / "combined_course" / "combined_course.usda"


def main() -> None:
    if _cli_args.world == "flat":
        cfg = BalanceEnvConfig(
            asset_path=ASSET, world_asset_path=None, use_default_ground=True,
            max_episode_steps=_cli_args.max_episode_steps,
        )
    else:
        cfg = BalanceEnvConfig(
            asset_path=ASSET, world_asset_path=COMBINED, world_prim_path="/World/CombinedCourse",
            use_default_ground=False, max_episode_steps=_cli_args.max_episode_steps,
        )

    env = BalanceCarEnv(cfg)
    print(f"EVAL_ENV_READY world={_cli_args.world}", flush=True)

    from stable_baselines3 import PPO
    model = PPO.load(str(_cli_args.model), env=None)
    print(f"EVAL_MODEL_LOADED path={_cli_args.model}", flush=True)

    rewards = []
    survival = []
    finals_x = []
    for ep in range(_cli_args.episodes):
        obs, _ = env.reset(seed=1000 + ep)
        if _cli_args.target_velocity is not None:
            env._target_velocity = _cli_args.target_velocity  # type: ignore[attr-defined]
            obs[7] = _cli_args.target_velocity
        ep_reward = 0.0
        ep_steps = 0
        last_info = {}
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            last_info = info
            if term or trunc:
                break
        rewards.append(ep_reward)
        survival.append(ep_steps)
        finals_x.append(float(last_info.get("x_w", 0.0)))
        print(
            f"EP {ep:2d} reward={ep_reward:+.2f} steps={ep_steps} final_x={last_info.get('x_w', 0.0):+.3f} "
            f"pitch={last_info.get('pitch_deg', 0.0):+.2f} roll={last_info.get('roll_deg', 0.0):+.2f} "
            f"fell={last_info.get('fell', False)}",
            flush=True,
        )

    print(
        f"EVAL_SUMMARY mean_reward={np.mean(rewards):+.2f} mean_steps={np.mean(survival):.0f} "
        f"mean_final_x={np.mean(finals_x):+.3f}",
        flush=True,
    )


try:
    main()
finally:
    _sim_app.close()
