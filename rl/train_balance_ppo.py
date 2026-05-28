"""Train PPO on the BalanceCarEnv inside Isaac Sim 5.0 standalone.

Single env, CPU-only torch (Isaac Sim 5.0 ships torch 2.5.1+cpu).

Quick sanity: 10k steps will take ~5 minutes; cart should at least learn
not to fall immediately. Real training needs 100k+ steps.

Run:
    D:\\isaac\\isaacsim\\python.bat rl\\train_balance_ppo.py --steps 10000
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# IMPORTANT: launch SimulationApp BEFORE importing anything that touches kit.
import argparse as _argparse_local

_parser = _argparse_local.ArgumentParser()
_parser.add_argument("--steps", type=int, default=10000)
_parser.add_argument("--world", choices=("flat", "combined"), default="flat")
_parser.add_argument("--save", type=Path, default=Path(__file__).resolve().parent / "ppo_balance.zip")
_parser.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "logs")
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("--max-episode-steps", type=int, default=1200)
_parser.add_argument("--resume-from", type=Path, default=None,
                     help="Load a pretrained PPO model and continue training.")
_parser.add_argument("--fixed-target", type=float, default=None,
                     help="Override env target_velocity to this fixed value (for diagnostic single-target training).")
_parser.add_argument("--action-mode", choices=("torque", "velocity"), default="torque",
                     help="torque: RL action is wheel effort (N*m). velocity: RL action is target wheel velocity (rad/s) tracked by PhysX PD. velocity is the Flamingo-baseline-recommended mode.")
_cli_args = _parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

_sim_app = SimulationApp({"headless": True})

# Now safe to import everything else
sys.path.insert(0, str(Path(__file__).resolve().parent))
from balance_env import BalanceCarEnv, BalanceEnvConfig  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ASSET = REPO / "sim" / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda"
COMBINED = REPO / "sim" / "assets" / "combined_course" / "combined_course.usda"


def main() -> None:
    if _cli_args.world == "flat":
        cfg = BalanceEnvConfig(
            asset_path=ASSET,
            world_asset_path=None,
            use_default_ground=True,
            max_episode_steps=_cli_args.max_episode_steps,
            fixed_target_velocity=_cli_args.fixed_target,
            action_mode=_cli_args.action_mode,
        )
    else:
        # Mixed-terrain spawn: cart starts at one of several key positions
        # along the combined course each episode. Prevents the policy from
        # specialising to (and over-fitting on) any single section.
        init_positions = (
            (-1.5, 0.0),    # start zone (flat) - balance practice
            (-0.8, 0.0),    # near ramp approach
            (-0.3, 0.0),    # right before ramp
            (+1.0, 0.272),  # top platform
            (+3.5, 0.0),    # flat after stairs
            (+4.7, 0.0),    # seesaw approach
            (+7.0, 0.0),    # flat before gravel
            (+8.5, 0.0),    # inside gravel patch
            (+11.0, 0.0),   # finish zone
        )
        cfg = BalanceEnvConfig(
            asset_path=ASSET,
            world_asset_path=COMBINED,
            world_prim_path="/World/CombinedCourse",
            use_default_ground=False,
            max_episode_steps=_cli_args.max_episode_steps,
            init_positions=init_positions,
        )

    print("RL_BUILD_ENV", flush=True)
    env = BalanceCarEnv(cfg)
    print("RL_ENV_READY obs_dim=8 action_dim=2", flush=True)

    # Quick smoke: random rollout to check env works
    obs, _ = env.reset(seed=_cli_args.seed)
    print(f"RL_RESET_OK obs[:4]={obs[:4]}", flush=True)
    rng = np.random.default_rng(_cli_args.seed)
    for _ in range(50):
        a = rng.uniform(-1, 1, size=2).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
    print("RL_SMOKE_OK", flush=True)

    # Train PPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    _cli_args.log_dir.mkdir(parents=True, exist_ok=True)
    env_wrapped = Monitor(env, str(_cli_args.log_dir / "monitor.csv"))

    if _cli_args.resume_from is not None and _cli_args.resume_from.exists():
        model = PPO.load(
            str(_cli_args.resume_from),
            env=env_wrapped,
            device="cpu",
            tensorboard_log=str(_cli_args.log_dir / "tb"),
        )
        print(f"RL_PPO_LOADED from={_cli_args.resume_from}", flush=True)
    else:
        model = PPO(
            "MlpPolicy",
            env_wrapped,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=1,
            device="cpu",
            seed=_cli_args.seed,
            tensorboard_log=str(_cli_args.log_dir / "tb"),
        )
        print("RL_PPO_BUILT", flush=True)

    model.learn(total_timesteps=_cli_args.steps, progress_bar=False)
    model.save(_cli_args.save)
    print(f"RL_TRAINING_DONE saved={_cli_args.save}", flush=True)


try:
    main()
finally:
    _sim_app.close()
