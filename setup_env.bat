@echo off
REM ============================================================
REM  One-time environment setup for the two-wheel balance car project.
REM
REM  Run this ONCE per machine to set the ISAACSIM_PATH env var permanently.
REM  After that, launch_seesaw_test.bat and friends will work without
REM  needing this script.
REM
REM  If you installed Isaac Sim somewhere other than the default path
REM  below, edit the SET line first, then run this .bat.
REM ============================================================

REM ---- EDIT THIS LINE TO POINT AT YOUR Isaac Sim 5.0 install ----
set "ISAACSIM_PATH=D:\isaac\isaacsim"
REM ---------------------------------------------------------------

if not exist "%ISAACSIM_PATH%\python.bat" (
  echo.
  echo ERROR: "%ISAACSIM_PATH%\python.bat" does not exist.
  echo.
  echo Please:
  echo   1. Install Isaac Sim 5.0 from
  echo      https://developer.nvidia.com/isaac/sim
  echo   2. Edit this file (setup_env.bat) and change ISAACSIM_PATH
  echo      to your install directory (the one that contains python.bat).
  echo   3. Re-run this file.
  echo.
  pause
  exit /b 1
)

echo Found Isaac Sim at: %ISAACSIM_PATH%
echo Setting ISAACSIM_PATH as a user environment variable...
setx ISAACSIM_PATH "%ISAACSIM_PATH%"
echo.
echo Done. You can now double-click launch_seesaw_test.bat from File Explorer.
echo (You may need to close and reopen any command prompts to see the change.)
echo.
pause
