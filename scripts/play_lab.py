"""Eval a trained Isaac Lab balance car policy.

Records trajectory at intermediate steps so we see motion regardless of episode
resets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

_LAB_RSL_DIR = Path("D:/isaac/IsaacLab-2.0.0/scripts/reinforcement_learning/rsl_rl")
sys.path.insert(0, str(_LAB_RSL_DIR))
import cli_args  # noqa: E402  type: ignore

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="TwoWheel-Balance-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--target_vel", type=float, default=0.10)
parser.add_argument("--rollout_seconds", type=float, default=5.0,
                    help="Run policy for this many sim-seconds (no reset constraint).")
cli_args.add_rsl_rl_args(parser)
parser.add_argument("--ckpt", type=Path, required=True, dest="checkpoint")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaaclab_task  # noqa: F401, E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # Override episode length so we get one long rollout per env
    env_cfg.episode_length_s = float(args_cli.rollout_seconds) + 1.0

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    u = env.unwrapped

    # Set viewport camera near the first env so the cart is actually visible
    # in GUI mode. Without this the default camera is far away and the cart
    # is a tiny speck in the middle of an empty-looking procedural terrain.
    if not args_cli.headless:
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            origin = u.scene.env_origins[0].cpu().numpy()
            set_camera_view(
                eye=(origin[0] + 1.5, origin[1] - 1.5, origin[2] + 0.8),
                target=(origin[0], origin[1], origin[2] + 0.05),
            )
            print(f"[EVAL] Camera set near env 0 origin {tuple(origin)}", flush=True)
        except Exception as e:
            print(f"[EVAL] Could not set viewport camera ({e}); using default.", flush=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args_cli.checkpoint))
    print(f"[EVAL] Loaded checkpoint: {args_cli.checkpoint}", flush=True)

    policy = runner.get_inference_policy(device=agent_cfg.device)

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    # Force fixed target_vel
    u._target_vel_x[:] = float(args_cli.target_vel)

    # Decimation = 2 → control rate = sim_dt * decimation
    sim_dt = u.cfg.sim.dt
    dec = u.cfg.decimation
    control_rate = 1.0 / (sim_dt * dec)
    num_steps = int(args_cli.rollout_seconds * control_rate)
    print(f"[EVAL] target_vel={args_cli.target_vel:+.3f} steps={num_steps} envs={args_cli.num_envs}", flush=True)

    # Record x position at start and every step
    env_origins = u.scene.env_origins.clone()
    x_history = []
    pitch_history = []
    fell_envs = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=agent_cfg.device)

    with torch.inference_mode():
        for step in range(num_steps):
            actions = policy(obs)
            obs, reward, done, info = env.step(actions)
            # Force target velocity after each step (in case reset resampled it)
            u._target_vel_x[:] = float(args_cli.target_vel)

            x = u.scene.articulations["robot"].data.root_pos_w[:, 0] - env_origins[:, 0]
            root_quat = u.scene.articulations["robot"].data.root_quat_w
            w, qx, qy, qz = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
            sinp = (2.0 * (w * qy - qz * qx)).clamp(-1.0, 1.0)
            pitch = torch.asin(sinp)

            x_history.append(x.cpu())
            pitch_history.append(pitch.cpu())

            # Mark envs that ever fell (terminated)
            terminated = u.reset_terminated
            fell_envs |= terminated

    import torch as T
    x_stack = T.stack(x_history)   # (T, N)
    pitch_stack = T.stack(pitch_history)

    # Per env final x
    final_x = x_stack[-1]
    # Per env max abs pitch
    max_abs_pitch = pitch_stack.abs().max(dim=0).values

    print(f"[EVAL] final_x: mean={final_x.mean():+.3f} m, stdev={final_x.std():.3f}, min={final_x.min():+.3f} max={final_x.max():+.3f}", flush=True)
    avg_vel = (x_stack[-1] - x_stack[0]) / args_cli.rollout_seconds
    print(f"[EVAL] avg_velocity: mean={avg_vel.mean():+.3f} m/s (target {args_cli.target_vel:+.3f})", flush=True)
    print(f"[EVAL] max_abs_pitch: mean={torch.rad2deg(max_abs_pitch).mean():.1f} deg", flush=True)
    print(f"[EVAL] ever_fell: {int(fell_envs.sum())}/{args_cli.num_envs} envs", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
