$ErrorActionPreference = "Stop"
Set-Location "D:\AI Project\two_wheel_balance"
Start-Transcript -Path "D:\AI Project\two_wheel_balance\sim\output\pid_gui_wander_launcher.log" -Force

& "D:\isaac\isaacsim\python.bat" `
  "D:\AI Project\two_wheel_balance\sim\scripts\play_pid.py" `
  --steps 36000 `
  --command-profile wander `
  --assist-upright `
  --report "D:\AI Project\two_wheel_balance\sim\output\pid_gui_wander_result.json" `
  --csv "D:\AI Project\two_wheel_balance\sim\output\pid_gui_wander_timeseries.csv"

Stop-Transcript
Read-Host "Isaac Sim run finished. Press Enter to close this window"
