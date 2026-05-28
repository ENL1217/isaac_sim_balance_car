@echo off
REM ============================================================
REM  Evaluate trained PPO policy (encoder version).
REM
REM  Loads the encoder PPO checkpoint and runs it in Isaac Sim with
REM  rendering on. The policy sees pitch / gyro / wheel encoders /
REM  target velocity (8-dim observation).
REM
REM  HONEST CAVEAT (verified by running it):
REM    These trained policies learn to STAND UPRIGHT well (0/32 envs
REM    fall, max_pitch < 21 deg) but do NOT track the velocity command
REM    (target 0.3 m/s → actual 0 m/s). The reward shaping made
REM    standing-still the easiest stable mode, and the policy converged
REM    there. To get a policy that actually drives, you would need to
REM    retrain with a stronger velocity-tracking reward or shorter
REM    "ok-to-stand" reward window.
REM
REM  In the GUI you will see the cart sitting upright not moving.
REM
REM  Requires Isaac Lab 2.0 installed.
REM ============================================================

cd /d "%~dp0"

if "%ISAACLAB_PATH%"=="" set "ISAACLAB_PATH=D:\isaac\IsaacLab-2.0.0"
if not exist "%ISAACLAB_PATH%\isaaclab.bat" (
  echo ERROR: Cannot find Isaac Lab at "%ISAACLAB_PATH%"
  pause
  exit /b 1
)

REM ---- Verified-working encoder checkpoint ----
REM Tested 2026-05-28: 0/32 envs fell, max_pitch=20.7 deg.
REM The 2026-05-26_14-06-51 run (model_2498) over-trained and falls 64/64.
set "CKPT=%~dp0rl\lab_logs\two_wheel_balance\2026-05-26_13-54-37\model_1999.pt"
if not exist "%CKPT%" (
  echo ERROR: Checkpoint not found at "%CKPT%"
  echo Train an encoder policy first via the Isaac Lab train script.
  pause
  exit /b 1
)

"%ISAACLAB_PATH%\isaaclab.bat" -p "%~dp0scripts\play_lab.py" ^
  --task TwoWheel-Balance-Direct-v0 ^
  --num_envs 4 ^
  --target_vel 0.3 ^
  --rollout_seconds 30 ^
  --ckpt "%CKPT%"

echo.
echo ============================================================
echo  Eval finished. Press any key to close this window.
echo ============================================================
pause >nul
