@echo off
REM ============================================================
REM  No-encoder dual-loop test
REM
REM  Hardware reality: many cheap balance cars use only an IMU
REM  (pitch + gyro), no wheel encoders. The controller then has
REM  only two feedback loops:
REM    - Balance loop: pitch -> wheel PWM
REM    - Turn loop:    A/D buttons -> wheel-differential PWM
REM
REM  No velocity loop -> cart drifts when W not pressed.
REM  No position loop -> cart accelerates indefinitely under W.
REM
REM  Compare with launch_seesaw_test.bat (which uses encoders) to
REM  see how much wheel-position feedback matters for sim2real
REM  stability.
REM ============================================================

cd /d "%~dp0"

if "%ISAACSIM_PATH%"=="" set "ISAACSIM_PATH=D:\isaac\isaacsim"
set "ISAACSIM_PY=%ISAACSIM_PATH%\python.bat"
if not exist "%ISAACSIM_PY%" (
  echo ERROR: Cannot find Isaac Sim at "%ISAACSIM_PY%"
  echo Set ISAACSIM_PATH to your Isaac Sim install dir. See docs/QUICKSTART.md.
  pause
  exit /b 1
)

REM Combined obstacle course shows how badly the cart fails without
REM encoders — under W the cart accelerates uncontrollably (no velocity
REM feedback to brake), then tips at the first obstacle. Use flat by
REM editing --world below if you'd rather isolate the pure-drift case.
"%ISAACSIM_PY%" "sim\scripts\play_pid_effort.py" ^
  --world combined ^
  --car-start-x-m -1.0 ^
  --car-start-z-m 0.10 ^
  --controller yahboom ^
  --command-profile teleop ^
  --no-encoder ^
  --effort-limit 2.17 ^
  --wheel-damping-coef 0.0 ^
  --steps 60000 ^
  --drive-pitch-offset-deg 8.0 ^
  --bluetooth-direction-magnitude 30 ^
  --turn-kd-pwm-per-dps 0.364 ^
  --teleop-velocity-m-s 0.6

echo.
echo ============================================================
echo  Isaac Sim has closed. Press any key to close this window.
echo ============================================================
pause >nul
