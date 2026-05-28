# Quick Start Guide / 詳細安裝指引

This is the detailed setup guide. For the 4-step overview see the top-level [README](../README.md).

> 這是完整的環境設定步驟。如果你只想快速跑起來，回去看 [README](../README.md)。

---

## 1. System requirements / 系統需求

| Item | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11 64-bit | Windows 11 |
| GPU | NVIDIA RTX 2060 (8 GB) | RTX 4070 (12 GB) |
| RAM | 16 GB | 32 GB |
| Disk | 50 GB free | 100 GB free (for Isaac Sim + caches) |
| NVIDIA Driver | 525.x or newer | Latest production driver |

This project is **Windows-tested**. Isaac Sim 5.0 runs on Linux too but the launcher `.bat` files are Windows-specific — you'd need to translate them to shell scripts.

Linux 也可以跑，但要自己改 .sh。

---

## 2. Install Isaac Sim 5.0 / 安裝 Isaac Sim

### Download

1. Go to <https://developer.nvidia.com/isaac/sim>.
2. Click **Download** (you may need to sign in with a free NVIDIA Developer account).
3. Choose **Isaac Sim 5.0.0** (this project is tested against 5.0; later versions may also work but parameters might need re-tuning).
4. Run the installer. ~30 GB download, ~50 GB installed.

### Verify the install

After install, find the install directory. It's usually:

- `C:\Users\<you>\AppData\Local\ov\pkg\isaac-sim-5.0.0\` (default Omniverse Launcher path), **or**
- `D:\isaac\isaacsim\` (custom install path).

That directory **must contain `python.bat`** at its top level. Verify:

```cmd
dir "<your_isaacsim_path>\python.bat"
```

If `python.bat` isn't there, the install didn't finish or you're looking at the wrong directory.

### Run Isaac Sim once standalone

Just to confirm it works:

```cmd
cd /d "<your_isaacsim_path>"
isaac-sim.bat
```

Isaac Sim should open. Close it when you've confirmed it launches.

---

## 3. (Optional) Install Isaac Lab 2.0 / 安裝 Isaac Lab

Only needed if you want to **train your own PPO policy**. Skip this if you just want to run the pre-built scenes.

```cmd
git clone -b v2.0.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
isaaclab.bat --install
```

See <https://isaac-sim.github.io/IsaacLab/> for full instructions.

---

## 4. Clone this repo / 下載這個專案

```cmd
git clone https://github.com/<your-username>/two_wheel_balance.git
cd two_wheel_balance
```

The repo is ~20 MB. Pre-generated USDs are included, so you don't have to wait for Isaac Sim to regenerate them on first run.

---

## 5. Configure ISAACSIM_PATH / 設定 ISAACSIM_PATH

### Easy method (recommended) / 簡單方法

1. Open `setup_env.bat` in any text editor (Notepad works).
2. Find this line:
   ```
   set "ISAACSIM_PATH=D:\isaac\isaacsim"
   ```
3. Change the path to your Isaac Sim install directory.
4. Save and close.
5. **Double-click** `setup_env.bat` in File Explorer to run it.

You should see "Done." and a green checkmark-feeling message. The env var is now persistent — you don't need to do this again.

### Manual method / 手動方法

If you prefer setting environment variables yourself:

1. Win+R, type `sysdm.cpl`, press Enter.
2. Advanced tab → Environment Variables.
3. Under "User variables", New:
   - Name: `ISAACSIM_PATH`
   - Value: your Isaac Sim install path (e.g., `D:\isaac\isaacsim`)
4. OK out. Close any open command prompts (they need to be relaunched to pick up the change).

### Verify

Open a new command prompt and:

```cmd
echo %ISAACSIM_PATH%
dir "%ISAACSIM_PATH%\python.bat"
```

The `echo` should show your path. The `dir` should find `python.bat`.

---

## 6. First run / 第一次執行

Double-click `launch_pid.bat`. You should see:

- A command prompt window opens with the Isaac Sim startup log scrolling.
- After ~25 seconds, a separate Isaac Sim window opens showing a 3D viewport.
- The cart appears on the elevated start platform.
- Physics runs (cart visibly balancing).

**To control the cart**: click anywhere on the 3D viewport (NOT on the menu bar) to give it keyboard focus, then:

- **W**: forward
- **S**: backward
- **A**: turn left
- **D**: turn right

**To quit**: close the Isaac Sim window. The command prompt stays open so you can read any messages.

### Expected timeline

- 0–2 s: cart settles on the elevated platform.
- Press W: cart drives down the stairs (4 × 25 mm drops — may take a couple tries; this is a known challenging case for classical controllers).
- Cart drives across the flat section.
- Press W more: cart climbs the orange ramp, crosses the apex, descends the blue ramp.
- Cart approaches the yellow seesaw, climbs the entry bevel.
- Once cart is ~16 cm past the seesaw pivot, **the yellow plank should visibly flip** (entry side goes up, exit side touches floor).
- Cart drives off the seesaw onto the green finish zone.

If any of these stages doesn't work, see [Troubleshooting](#troubleshooting) below.

---

## 7. Inspect telemetry / 看 CSV log

Every run writes a CSV to `sim/output/pid_effort_timeseries.csv` (overwritten each run). Each row is one physics tick (~1 kHz). Useful columns:

| Column | Meaning |
|---|---|
| `time_s` | sim time in seconds |
| `x_m`, `y_m`, `z_m` | cart world position |
| `pitch_deg`, `roll_deg`, `yaw_deg` | cart orientation |
| `wheel_velocity_m_s` | average wheel ground speed |
| `left_effort`, `right_effort` | torque commands (N·m) |
| `plank_pitch_deg`, `plank_world_z` | seesaw plank state |
| `key_forward`, `key_back`, `key_left`, `key_right` | which keys were held this tick |

Read in Excel, Python (pandas), or any CSV viewer.

---

## 8. Try other scenes / 試其他場景

Edit `launch_pid.bat` and change `--world combined` to one of:

- `flat` — just a flat ground plane (no obstacles).
- `slope` — single 30° ramp (hard for PID, doable for LQR with FF).
- `stairs` — descending stair sequence.
- `seesaw` — standalone seesaw only.
- `gravel` — uneven terrain (no controller passes this yet).
- `line` — straight track with IR line.
- `combined` (default) — full obstacle course.

---

## 9. Try other controllers / 試其他控制器

Edit `launch_pid.bat` and change `--controller yahboom` to:

- `yahboom` (default) — three-loop PID ported from real STM32 firmware (see CONTROL_METHODS.md).
- `legacy` — simpler PID, older implementation.
- `lqr` — full-state LQR. Add `--slope-feedforward` for ramps.
- `lqi` — LQR + velocity integrator.
- `threering` — three-ring parallel PID per course-material spec. Or just use `launch_seesaw_threering.bat`.

See `docs/CONTROL_METHODS.md` for what each one does.

---

## Troubleshooting / 疑難排解

### "Cannot find Isaac Sim at ..."

Your `ISAACSIM_PATH` isn't pointing at the right directory. Run `setup_env.bat` again and double-check the path. The directory must contain `python.bat` directly (not in a subfolder).

### Isaac Sim window doesn't open after 30 seconds

Check the command prompt for error messages. Common issues:

- **"CUDA not available"**: Update your NVIDIA driver to the latest production version.
- **"Out of memory"**: Close other GPU-heavy apps. Isaac Sim wants ≥ 6 GB free VRAM.
- **"Warp CUDA error 36"**: Driver mismatch. Usually harmless for our use case (we use PhysX, not Warp).

### Cart immediately falls

- **Falls forward as soon as physics starts**: Check `--effort-limit` — default is 2.17 N·m (real GB37 stall torque). If accidentally lowered to < 0.5, the cart's motor can't hold balance.
- **Falls sideways**: Single-axle roll instability. Try with `--world flat` first to confirm controller works on level ground.

### Cart doesn't respond to W/A/S/D

- Did you click the **3D viewport** first? (Not the menu bar, not the property panel.)
- Check the CSV `key_forward` column. If it's `0` while you're holding W, Isaac Sim's hotkey system might be eating the key. Try clicking deeper into the viewport area.

### Seesaw plank doesn't flip

This was an extensively debugged bug in C13. The fix is in the current code. If it regresses:

1. Open `sim/output/pid_effort_timeseries.csv`.
2. Look at `plank_pitch_deg` over time.
3. If it stays at `-3.13°` (resting) while cart is on the plank's exit side, you have a friction-lock issue: a static collider (likely entry_bevel) is touching the plank's underside, creating a permanent contact constraint with full static friction that prevents joint rotation. Look at `sim/scripts/create_combined_course.py` near `bevel_z_gap_m` — keep a 5 mm vertical gap between static decorative geometry and the dynamic plank.

### Performance is slow

- Isaac Sim at 1 kHz physics runs in real-time on RTX 4070. On weaker GPUs you may see wall-clock slowdown.
- For RL training, headless mode (no GUI) is 5-10× faster. Use `--headless` flag with `play_pid_effort.py`.

### File `External_refs` not in repo

You may have seen references in older docs to an `external_refs/` directory bundling other repos. This was removed for the public release (licensing). The upstream URLs are in `docs/REFERENCES.md` — go to those repos directly.

---

## Regenerate USD scenes / 重新生成 USD

The repo ships with pre-generated USD scenes in `sim/assets/`. If you change parameters in `sim/scripts/create_*.py`, regenerate the corresponding USD:

```cmd
"%ISAACSIM_PATH%\python.bat" sim\scripts\create_combined_course.py
"%ISAACSIM_PATH%\python.bat" sim\scripts\create_balance_car_usd.py
```

The scripts overwrite the USDs in place.

---

## Next steps / 接下來可以做什麼

- Read `docs/SIM_VS_REAL.md` to understand the physics parameters (what's matched to real hardware, what's sim-only).
- Read `docs/CONTROL_METHODS.md` for a student-friendly walkthrough of each controller (PID, LQR, LQI, three-ring, PPO).
- Read `docs/CONTROL_ARCHITECTURE.md` for sign conventions and the controller block diagram.
- Try `sim/scripts/create_balance_board_usd.py` — the **minimal standalone seesaw scene** — when you want to learn USD joint setup without the full obstacle course.
- Open the Isaac Lab task `isaaclab_task/balance_car/` to study how the RL training is structured.

Have fun. Open an issue on GitHub if something's broken or unclear.
