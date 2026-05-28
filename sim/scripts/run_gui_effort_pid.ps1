$ErrorActionPreference = "Stop"
Set-Location "D:\AI Project\two_wheel_balance"

& "D:\isaac\isaacsim\python.bat" `
  "D:\AI Project\two_wheel_balance\sim\scripts\play_pid_effort.py" `
  --steps 36000 `
  --command-profile drive_demo `
  --report "D:\AI Project\two_wheel_balance\sim\output\pid_effort_gui_result.json" `
  --csv "D:\AI Project\two_wheel_balance\sim\output\pid_effort_gui_timeseries.csv"

Read-Host "Pure physics PID run finished. Press Enter to close this window"
