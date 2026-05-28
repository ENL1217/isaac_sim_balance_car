@echo off
REM ============================================================
REM  Train PPO on the ENCODER balance car variant.
REM
REM  Observation: 8-dim (5 IMU + wheel_pos + wheel_vel + target_vel).
REM  Action: 2-dim (left/right wheel torques).
REM
REM  4096 parallel envs, 1500 iter, ~15-20 min on RTX 4070.
REM  Logs to rl/lab_logs/two_wheel_balance/<timestamp>/.
REM
REM  HONEST CAVEAT: with the current reward shaping, PPO tends to
REM  converge to "stand still" rather than learning velocity tracking
REM  (standing is the easiest stable mode). Comparing your fresh run's
REM  final-iter checkpoint to launch_eval_rl_encoder.bat's reference
REM  checkpoint should give similar pitch stability (0/N envs fall)
REM  but similar velocity tracking failure (target ≠ actual).
REM
REM  If you want a policy that actually drives, raise rew_track_lin_vel
REM  in isaaclab_task/balance_car/env.py:BalanceCarEnvCfg and retrain.
REM
REM  Requires Isaac Lab 2.0 installed.
REM ============================================================

cd /d "%~dp0"

if "%ISAACLAB_PATH%"=="" set "ISAACLAB_PATH=D:\isaac\IsaacLab-2.0.0"
if not exist "%ISAACLAB_PATH%\isaaclab.bat" (
  echo ERROR: Cannot find Isaac Lab at "%ISAACLAB_PATH%"
  echo Set ISAACLAB_PATH to your Isaac Lab 2.0 install dir.
  echo Download from https://github.com/isaac-sim/IsaacLab
  pause
  exit /b 1
)

"%ISAACLAB_PATH%\isaaclab.bat" -p "%~dp0scripts\train_lab.py" ^
  --task TwoWheel-Balance-Direct-v0 ^
  --num_envs 4096 ^
  --max_iterations 1500 ^
  --headless

echo.
echo ============================================================
echo  Training finished. Checkpoint saved under rl/lab_logs/.
echo ============================================================
pause >nul
