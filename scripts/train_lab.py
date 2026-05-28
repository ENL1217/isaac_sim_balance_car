"""Wrapper to train our balance car task with Isaac Lab's rsl_rl pipeline.

Usage:
    cd D:/isaac/IsaacLab-2.0.0
    ./isaaclab.bat -p "D:/AI Project/two_wheel_balance/scripts/train_lab.py" \
        --task TwoWheel-Balance-Direct-v0 --num_envs 4096 --headless

This script:
1. Adds our project root to sys.path so `isaaclab_task` is importable
2. Imports our task module → triggers gym.register
3. Defers to Isaac Lab's standard rsl_rl train.py logic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path FIRST so we can import isaaclab_task
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

# Lab AppLauncher must be the first Isaac import
from isaaclab.app import AppLauncher  # noqa: E402

# Locate Lab's rsl_rl scripts dir to reuse cli_args
_LAB_RSL_DIR = Path("D:/isaac/IsaacLab-2.0.0/scripts/reinforcement_learning/rsl_rl")
sys.path.insert(0, str(_LAB_RSL_DIR))
import cli_args  # noqa: E402  type: ignore

# CLI
parser = argparse.ArgumentParser(description="Train RL agent (balance car) with rsl_rl on Isaac Lab.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="TwoWheel-Balance-Direct-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--pretrained", type=str, default=None,
                    help="Path to a previous .pt checkpoint to warm-start the policy. "
                         "Useful for cross-run resume (e.g. flat baseline → terrain fine-tune).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Now safe to import everything else."""
import gymnasium as gym  # noqa: E402
import os  # noqa: E402
from datetime import datetime  # noqa: E402

import torch  # noqa: E402

# Register Lab's built-in tasks (needed for hydra)
import isaaclab_tasks  # noqa: F401, E402

# Register OUR task
import isaaclab_task  # noqa: F401, E402

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402

from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    # Multi-GPU not used. Single GPU training.
    log_root = os.path.abspath(
        os.path.join(_PROJECT_ROOT, "rl", "lab_logs", agent_cfg.experiment_name)
    )
    log_root = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_root, exist_ok=True)
    print(f"[INFO] Logging to: {log_root}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    dump_yaml(os.path.join(log_root, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root, "params", "agent.pkl"), agent_cfg)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_root, device=agent_cfg.device)
    if args_cli.pretrained is not None:
        pretrained = os.path.abspath(args_cli.pretrained)
        print(f"[INFO] Warm-starting from pretrained checkpoint: {pretrained}", flush=True)
        runner.load(pretrained)
    elif args_cli.resume:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading model checkpoint: {resume_path}", flush=True)
        runner.load(resume_path)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
