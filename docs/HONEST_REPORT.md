# Honest Status Report

> Last verified: 2026-05-28 against the shipped checkpoints + bat files.

This document supersedes any marketing language elsewhere. Read it before
believing any "controller X works" claim.

---

## TL;DR

| Capability | PID | LQR | LQI | 3-ring | PPO (encoder) | PPO (no-enc) |
|---|---|---|---|---|---|---|
| Stand still 30 s | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Drive forward on flat | ✓ | partial | ✓ | ✗ (tuned for stand) | ✗ (stuck still) | ✗ (stuck still) |
| Press A/D briefly | ✓ | ✓ | ✓ | ✓ | n/a | n/a |
| 3° ramp (combined course) | ✓ | ✗ (stalls) | partial | ✗ | ✗ | ✗ |
| 25 mm stair drop | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Seesaw flip-cross | partial (rolls after) | ✗ | ✗ | ✗ | ✗ | ✗ |
| Gravel | ✗ (geometric, single-axle roll) | ✗ | ✗ | ✗ | ✗ | ✗ |

**Honest bottom line**: no controller in this repo traverses the full
combined obstacle course end-to-end without failing somewhere. Classical
PID gets furthest (across the seesaw, then rolls in the finish zone).
The shipped RL checkpoints learn balance but **do not learn velocity
tracking** — they converge to standing still.

---

## Verified results (headless smoke tests)

All on `combined_course`, `--command-profile climb` (W held), RTX 4070.

### Classical (15 000 steps ≈ 30 s sim)

| Controller | max_x reached | Fall? | Notes |
|---|---|---|---|
| PID (`launch_pid.bat`) | **14.6 m** | t=27.5 s, roll −34.2° in finish zone | Only controller that crosses the seesaw |
| LQR (`launch_lqr.bat`) | 3.1 m | no fall | Stalls before the orange ramp |
| LQI (`launch_lqi.bat`) | 6.9 m | no fall | Stalls at apex |
| Three-ring (`launch_threering.bat`) | 0.05 m | no fall | Tuned for stationary balance, doesn't drive |
| PID + no-encoder (`launch_no_encoder.bat`) | 12.9 m | t=9.1 s, pitch −36° | No velocity loop → runaway speed → tips |

### PPO RL (1500–2000 iter, 4096 envs, eval on 32 envs @ target_vel=0.3, 8 s)

| Checkpoint | Pitch stable? | Vel tracking | Verdict |
|---|---|---|---|
| Encoder | ✓ (0/32 fall, max 20.7°) | ✗ (target 0.3 → actual 0 m/s) | Stands well, refuses to drive |
| No-encoder | ✓ (0/32 fall, max 3.7°) | ✗ (target 0.3 → actual 0 m/s) | Stands extremely well, refuses to drive |

GUI shows the cart sitting upright not moving.

**Why**: current reward weights make standing the easiest stable mode.
The agent learns balance + flat-orientation + alive bonus without risking
falling by accelerating. To get a driving policy, weight
`rew_track_lin_vel` higher (currently 5, try 30+) or change to a
line-following task — see `docs/CONTROL_METHODS.md` "已知 limitation".

---

## Hardware limitations no controller can fix

1. **Single-axle two-wheel chassis has no roll authority.** Lateral
   perturbations can't be corrected by motor action. Gravel and uneven
   terrain will tip eventually regardless of controller. Geometric, not
   a tuning problem.

2. **PhysX cylinder-collider edge contact** generates small vertical
   impulses at slab joints. Mitigation: thin plates (< 10 % wheel
   radius). Already done in the combined course.

3. **Motor torque ceiling = real GB37 stall (2.17 N·m).** Some obstacle
   transitions need transient torque above this. The cart "gives up" on
   the stairs because the motor saturates.

---

## What's solid

- **Physics realism**: motor params, plate thickness, IMU noise scales
  match real hardware. See `docs/SIM_VS_REAL.md`.
- **Combined obstacle course**: stairs/ramps/seesaw load and respond
  correctly. The seesaw flips at the predicted force-balance point
  (verified in C13).
- **CSV instrumentation**: `play_pid_effort.py` logs 45 columns per
  tick, including plank pitch. Use this to diagnose any "did X happen?"
  question.
- **Bat launchers**: every shipped bat has been headless-tested.

## What's NOT solid

- **PPO reward shaping** — policy collects reward by not moving. Treat
  shipped RL as "balance can be learned" demo, not a working controller.
- **Three-ring forward-drive tuning** — gains follow W13 spec but motor
  model differs, so default config doesn't drive.
- **Sim2real transfer** — no real-cart validation in this session. All
  claims are sim-only.

---

## Verify any claim yourself

```cmd
"%ISAACSIM_PATH%\python.bat" sim\scripts\play_pid_effort.py ^
  --world combined --controller <name> --command-profile climb ^
  --headless --steps 15000 --csv sim\output\my_test.csv
```

Then read max_x and fall point from the CSV (see `docs/QUICKSTART.md`).
File an issue with CSV attached if your numbers disagree with this report.
