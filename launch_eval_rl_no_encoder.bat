@echo off
REM ============================================================
REM  Evaluate trained PPO policy (NO-ENCODER version).
REM
REM  Loads a no-encoder PPO checkpoint (6-dim observation: IMU + target,
REM  no wheel pos/vel). Compare with launch_eval_rl_encoder.bat to see
REM  how much the encoder feedback matters.
REM
REM  HONEST CAVEAT (verified by running it):
REM    Trained 1500 iter / 4096 envs / mean reward 3837.
REM    Result: 0/32 envs fall, max_pitch=3.7 deg (excellent balance!)
REM    BUT velocity tracking failed: target=0.3 m/s -> actual=0 m/s.
REM    Same failure mode as the encoder version — policy converged to
REM    "stand still" because that was the easiest stable reward mode.
REM    The cart will sit perfectly upright in the GUI but won't move.
REM
REM  This is actually a teaching insight: even with 4096-env PPO, if
REM  your reward shaping lets standing-still be a local optimum, the
REM  policy will lock onto it. Compare with the rl/ folder's standalone
REM  PPO experiments which used different reward weights.
REM ============================================================

cd /d "%~dp0"

if "%ISAACLAB_PATH%"=="" set "ISAACLAB_PATH=D:\isaac\IsaacLab-2.0.0"
if not exist "%ISAACLAB_PATH%\isaaclab.bat" (
  echo ERROR: Cannot find Isaac Lab at "%ISAACLAB_PATH%"
  pause
  exit /b 1
)

REM ---- The no-encoder training run we shipped this with ----
set "CKPT=%~dp0rl\lab_logs\two_wheel_balance\2026-05-28_13-35-39\model_1499.pt"
REM Change the timestamp if you retrain (see rl/lab_logs/two_wheel_balance/ for runs)
REM ------------------------------------------------------------

if not exist "%CKPT%" (
  echo ERROR: Checkpoint not found at "%CKPT%"
  echo.
  echo Available runs under rl/lab_logs/two_wheel_balance/:
  dir /b "%~dp0rl\lab_logs\two_wheel_balance\" 2>nul
  echo.
  echo Edit this file (launch_eval_rl_no_encoder.bat) and replace the
  echo NO_ENCODER_RUN_TIMESTAMP segment with the actual folder name above.
  pause
  exit /b 1
)

"%ISAACLAB_PATH%\isaaclab.bat" -p "%~dp0scripts\play_lab.py" ^
  --task TwoWheel-Balance-NoEncoder-Direct-v0 ^
  --num_envs 4 ^
  --target_vel 0.3 ^
  --rollout_seconds 30 ^
  --ckpt "%CKPT%"

echo.
echo ============================================================
echo  Eval finished. Press any key to close this window.
echo ============================================================
pause >nul
