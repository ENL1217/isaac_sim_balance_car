@echo off
REM ============================================================
REM  PID controller on the combined obstacle course (GUI teleop).
REM
REM  This is the project's "main" controller — three loops in parallel:
REM    - Balance: PD on pitch (keeps cart upright)
REM    - Speed:   PI on wheel velocity (closes velocity loop via encoder)
REM    - Turn:    Direct PWM differential + Kd damping on yaw rate
REM
REM  The exact gains and architecture are ported from a real Yahboom
REM  STM32 balance-car firmware (see docs/CONTROL_METHODS.md for the
REM  full mapping). With encoder feedback the cart can cross stairs,
REM  ramps, and the seesaw.
REM
REM  Controls inside Isaac Sim (click the 3D viewport first to
REM  give the window keyboard focus):
REM    W = forward    A = turn left
REM    S = backward   D = turn right
REM
REM  Close the Isaac Sim window when you're done.
REM ============================================================

REM Move to the project root (this .bat's own directory)
cd /d "%~dp0"

REM Locate Isaac Sim's python.bat. Honor ISAACSIM_PATH env var if set,
REM otherwise fall back to the developer's machine default.
REM Set ISAACSIM_PATH in setup_env.bat or in Windows System Properties.
if "%ISAACSIM_PATH%"=="" set "ISAACSIM_PATH=D:\isaac\isaacsim"
set "ISAACSIM_PY=%ISAACSIM_PATH%\python.bat"
if not exist "%ISAACSIM_PY%" (
  echo ERROR: Cannot find Isaac Sim's python.bat at "%ISAACSIM_PY%"
  echo Please set ISAACSIM_PATH to your Isaac Sim 5.0 install directory.
  echo See docs/QUICKSTART.md for setup instructions.
  pause
  exit /b 1
)

REM --- v2 cart parameters (real Yahboom-aligned, C3 commit c41e276) ---
REM   effort-limit 2.17                 = real GB37 stall torque (was 0.6)
REM   wheel-damping-coef 0.0            = back-EMF lives in the USD joint now
REM   drive-pitch-offset-deg 8.0        = stronger lean to compensate low CoM
REM   bluetooth-direction-mag 30        = real STM32 Turn_Amplitude*Turn_Kp/Flag_velocity
REM                                        = 54*42/2 = 1134 PWM = 16.4% of 6900 PWM
REM                                        in real, equivalent to ~42 of our 255.
REM                                        We keep 30 (11.8%) here because PhysX
REM                                        single-axle roll is more sensitive than
REM                                        real-world roll (verified: removing the
REM                                        speed-taper + bumping to 42 rolls the
REM                                        cart over in stress_teleop). 30 + taper
REM                                        + Kd damping matches the real cart's
REM                                        perceived turn rate without tipping.
REM   turn-kd-pwm-per-dps 0.364         = real STM32 Turn_Kd=0.6 LSB^-1, scaled to
REM                                        our PWM 255 from real 6900:
REM                                        0.6 * 16.4 LSB/dps * 255/6900 ≈ 0.364.
REM                                        Only active when W/S held (Flag_front||
REM                                        Flag_back), mirroring real gain-scheduling
REM                                        on B570/control.c:168. Adds yaw-rate
REM                                        damping that the real cart relies on to
REM                                        stop the instant 0→±27 Turn_Target step
REM                                        from over-shooting.
REM   teleop-velocity-m-s 0.6           = matches the cart's actual top speed

"%ISAACSIM_PY%" "sim\scripts\play_pid_effort.py" ^
  --world combined ^
  --car-start-x-m -1.0 ^
  --car-start-z-m 0.10 ^
  --controller yahboom ^
  --command-profile teleop ^
  --effort-limit 2.17 ^
  --wheel-damping-coef 0.0 ^
  --steps 60000 ^
  --drive-pitch-offset-deg 8.0 ^
  --bluetooth-direction-magnitude 30 ^
  --turn-kd-pwm-per-dps 0.364 ^
  --bluetooth-speed-magnitude 100 ^
  --teleop-velocity-m-s 0.6
REM IMU noise + encoder quantize available via:
REM   --imu-pitch-noise-deg 0.05 --imu-gyro-noise-deg-s 0.5
REM   --imu-gyro-bias-deg-s 0.05 --encoder-quantize
REM They improve sim2real fidelity in HEADLESS (C7: 300 s stand stable)
REM but accelerate lateral drift in GUI (PhysX-GUI specific). Default off
REM for GUI; turn on for RL training / batch evaluation.

echo.
echo ============================================================
echo  Isaac Sim has closed. Press any key to close this window.
echo ============================================================
pause >nul
