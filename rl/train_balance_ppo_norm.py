"""Train PPO on the NORMALISED balance car env. Use if balance_env.py policy
fails to learn target-velocity tracking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_parser = argparse.ArgumentParser()
_parser.add_argument("--steps", type=int, default=500000)
_parser.add_argument("--world", choices=("flat", "combined"), default="flat")
_parser.add_argument("--save", type=Path, default=Path(__file__).resolve().parent / "ppo_norm.zip")
_parser.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "logs_norm")
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("--max-episode-steps", type=int, default=1200)
_parser.add_argument("--resume-from", type=Path, default=None)
_cli = _parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402
_sim_app = SimulationApp({"headless": True})

sys.path.insert(0, str(Path(__file__).resolve().parent))
from balance_env_norm import BalanceCarEnvNorm, BalanceEnvNormConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ASSET = REPO / "sim" / "assets" / "two_wheel_balance_car" / "two_wheel_balance_car.usda"
COMBINED = REPO / "sim" / "assets" / "combined_course" / "combined_course.usda"


def main() -> None:
    cfg = BalanceEnvNormConfig(
        asset_path=ASSET,
        world_asset_path=COMBINED if _cli.world == "combined" else None,
        use_default_ground=(_cli.world == "flat"),
        max_episode_steps=_cli.max_episode_steps,
    )
    env = BalanceCarEnvNorm(cfg)
    print(f"NORM_ENV_READY world={_cli.world}", flush=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    _cli.log_dir.mkdir(parents=True, exist_ok=True)
    env_wrapped = Monitor(env, str(_cli.log_dir / "monitor.csv"))

    if _cli.resume_from is not None and _cli.resume_from.exists():
        model = PPO.load(
            str(_cli.resume_from),
            env=env_wrapped,
            device="cpu",
            tensorboard_log=str(_cli.log_dir / "tb"),
        )
        print(f"NORM_PPO_LOADED from={_cli.resume_from}", flush=True)
    else:
        model = PPO(
            "MlpPolicy",
            env_wrapped,
            learning_rate=3e-4,
            n_steps=1024,  # bigger rollout buffer with normalised obs
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.005,
            verbose=1,
            device="cpu",
            seed=_cli.seed,
            tensorboard_log=str(_cli.log_dir / "tb"),
        )
        print("NORM_PPO_BUILT", flush=True)

    model.learn(total_timesteps=_cli.steps, progress_bar=False)
    model.save(_cli.save)
    print(f"NORM_TRAINING_DONE saved={_cli.save}", flush=True)


try:
    main()
finally:
    _sim_app.close()
