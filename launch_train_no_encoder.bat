@echo off
REM ============================================================
REM  Train PPO on the NO-ENCODER balance car variant.
REM
REM  Observation space is reduced from 8 -> 6 dims (drops the two
REM  encoder-derived columns x_w and x_w_dot). The policy must
REM  balance and track velocity using IMU + target only.
REM
REM  Compare with TwoWheel-Balance-Direct-v0 (encoder version):
REM  expect significantly worse velocity tracking and a drifting
REM  rest pose because the policy can no longer observe its own
REM  position/velocity error.
REM
REM  Requires Isaac Lab 2.0 installed at the path below (edit if
REM  yours is elsewhere).
REM ============================================================

cd /d "%~dp0"

if "%ISAACSIM_PATH%"=="" set "ISAACSIM_PATH=D:\isaac\isaacsim"
if "%ISAACLAB_PATH%"=="" set "ISAACLAB_PATH=D:\isaac\IsaacLab-2.0.0"

if not exist "%ISAACLAB_PATH%\isaaclab.bat" (
  echo ERROR: Cannot find Isaac Lab at "%ISAACLAB_PATH%"
  echo Set ISAACLAB_PATH to your Isaac Lab 2.0 install dir.
  echo Download from https://github.com/isaac-sim/IsaacLab
  pause
  exit /b 1
)

REM Training defaults: 4096 envs, ~1500 iterations (~2 M steps), headless.
REM Logs go to rl/lab_logs/. Use --resume or --pretrained to warm-start
REM from an existing checkpoint.
"%ISAACLAB_PATH%\isaaclab.bat" -p "%~dp0scripts\train_lab.py" ^
  --task TwoWheel-Balance-NoEncoder-Direct-v0 ^
  --num_envs 4096 ^
  --max_iterations 1500 ^
  --headless

echo.
echo ============================================================
echo  Training finished. Checkpoint saved under rl/lab_logs/.
echo ============================================================
pause >nul
