@echo off
REM ============================================================
REM  LQI (LQR + Integral state) controller on combined course
REM
REM  Like LQR but adds an extra integral state on velocity error.
REM  Helps eliminate steady-state velocity offset on slopes where
REM  pure LQR drifts.
REM
REM  Best classical controller in our HONEST_REPORT.md matrix for
REM  velocity tracking on flat + gentle slopes.
REM ============================================================

cd /d "%~dp0"

if "%ISAACSIM_PATH%"=="" set "ISAACSIM_PATH=D:\isaac\isaacsim"
set "ISAACSIM_PY=%ISAACSIM_PATH%\python.bat"
if not exist "%ISAACSIM_PY%" (
  echo ERROR: Cannot find Isaac Sim at "%ISAACSIM_PY%"
  pause
  exit /b 1
)

"%ISAACSIM_PY%" "sim\scripts\play_pid_effort.py" ^
  --world combined ^
  --car-start-x-m -1.0 ^
  --car-start-z-m 0.10 ^
  --controller lqi ^
  --command-profile teleop ^
  --effort-limit 2.17 ^
  --wheel-damping-coef 0.0 ^
  --slope-feedforward ^
  --adaptive-slope-boost ^
  --steps 60000 ^
  --lqi-q-integral 20 ^
  --lqi-integral-clamp 2.0 ^
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
