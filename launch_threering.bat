@echo off
REM ============================================================
REM  GUI test using the W13 three-ring parallel PID controller
REM  (Balance PD + Velocity PI + Turn PD with gain-scheduled Kd)
REM
REM  Controls (click Isaac Sim viewport FIRST to give it focus):
REM    W = forward    S = backward    A = turn left    D = turn right
REM
REM  Spec gain defaults: Kp_balance=250, Kd_balance=5,
REM                      Kp_velocity=0.7, Ki_velocity=0.01,
REM                      Kp_turn=0.5, Kd_turn=0.3
REM ============================================================

cd /d "%~dp0"

REM Locate Isaac Sim's python.bat (honor ISAACSIM_PATH env var).
if "%ISAACSIM_PATH%"=="" set "ISAACSIM_PATH=D:\isaac\isaacsim"
set "ISAACSIM_PY=%ISAACSIM_PATH%\python.bat"
if not exist "%ISAACSIM_PY%" (
  echo ERROR: Cannot find Isaac Sim at "%ISAACSIM_PY%"
  echo Set ISAACSIM_PATH to your Isaac Sim install dir. See docs/QUICKSTART.md.
  pause
  exit /b 1
)

REM Keep the spec's gain values verbatim (Kp_balance=250 etc.), but
REM rescale the PWM saturation limit from STM32-style 7200/3000 down to
REM 500 — that brings the spec's gain-to-effort mapping in line with
REM our 0.6 N*m simulated motor. Result: 1 deg pitch -> 0.3 N*m balance
REM correction (saturates near 1.7 deg). Velocity loop steady-state
REM ~25 mN*m, enough to drag the cart forward without overpowering
REM balance.

"%ISAACSIM_PY%" "sim\scripts\play_pid_effort.py" ^
  --world combined ^
  --car-start-x-m -1.0 ^
  --car-start-z-m 0.10 ^
  --controller threering ^
  --command-profile teleop ^
  --effort-limit 0.6 ^
  --wheel-damping-coef 0.005 ^
  --steps 60000 ^
  --threering-pwm-limit 500 ^
  --threering-position-clamp 1000

echo.
echo ============================================================
echo  Isaac Sim has closed. Press any key to close this window.
echo ============================================================
pause >nul
