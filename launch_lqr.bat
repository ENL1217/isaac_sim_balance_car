@echo off
REM ============================================================
REM  Full-state LQR controller on combined course
REM
REM  LQR (Linear Quadratic Regulator) uses a state-space model:
REM    state = [pitch, pitch_rate, wheel_pos, wheel_vel]
REM    gain matrix K derived from solving the algebraic Riccati equation
REM  with Q (state cost) and R (control cost) weights.
REM
REM  Compared to PID: works better on slopes because gravity feed-forward
REM  is part of the linearization, not a hand-tuned correction.
REM ============================================================

cd /d "%~dp0"

if "%ISAACSIM_PATH%"=="" set "ISAACSIM_PATH=D:\isaac\isaacsim"
set "ISAACSIM_PY=%ISAACSIM_PATH%\python.bat"
if not exist "%ISAACSIM_PY%" (
  echo ERROR: Cannot find Isaac Sim at "%ISAACSIM_PY%"
  echo Set ISAACSIM_PATH. See docs/QUICKSTART.md.
  pause
  exit /b 1
)

"%ISAACSIM_PY%" "sim\scripts\play_pid_effort.py" ^
  --world combined ^
  --car-start-x-m -1.0 ^
  --car-start-z-m 0.10 ^
  --controller lqr ^
  --command-profile teleop ^
  --effort-limit 2.17 ^
  --wheel-damping-coef 0.0 ^
  --slope-feedforward ^
  --adaptive-slope-boost ^
  --steps 60000 ^
  --lqr-q-pitch 5000 ^
  --lqr-q-pitch-rate 100 ^
  --lqr-q-pos 1.0 ^
  --lqr-q-vel 0.5 ^
  --lqr-r-torque 0.1 ^
  --teleop-velocity-m-s 0.6

echo.
echo ============================================================
echo  Isaac Sim has closed. Press any key to close this window.
echo ============================================================
pause >nul
