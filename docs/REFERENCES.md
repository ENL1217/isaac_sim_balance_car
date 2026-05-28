# External References / 外部參考

The implementations and design choices in this repo draw from the following open-source projects. We **do not redistribute** their source — please go to the upstream repos.

這份專案參考過的開源實作。我們**不重新發佈**它們的 source code，請去上游 repo 看。

---

## Robot / RL references

### WheeledLab
- **Repo**: <https://github.com/UWRobotLearning/WheeledLab>
- **What we use**: PhysX wheel-collider parameter patterns (`maxDepenetrationVelocity`, `enableGyroscopicForces`, `solverPositionIterationCount`). Their MUSHR RC car setup is the reference we matched on the physical settings front.
- **Their license**: MIT.

### Isaac-RL-Two-wheel-Legged-Bot (Flamingo Edu v1)
- **Repo**: <https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot>
- **What we use**: PPO reward shaping for a 2-wheel-2-leg balance robot in Isaac Lab. Our `isaaclab_task/balance_car/env.py` reward function structure mirrors their `flamingo_edu_v1` setup.
- **Important caveat**: Their robot has hip joints; ours doesn't. We removed the leg-related reward terms.

### Isaac Lab
- **Repo**: <https://github.com/isaac-sim/IsaacLab>
- **Used version**: 2.0.0 (we tested against this; later versions may need API updates).
- **What we use**: `ManagerBasedRLEnv`, the 4096-env parallel training pipeline, rsl-rl PPO trainer.

---

## Hardware references

### Yahboom STM32 平衡小車
- **Product page**: <https://category.yahboom.net/products/yahboom-stm32-self-balancing-car> (or search "Yahboom STM32 balance car").
- **What we model**: chassis dimensions, GB37 motor specs, MPU6050 placement, 18650 battery layout.
- **Datasheet snapshots**: see `docs/HARDWARE_NOTES.md` for the numerical summary we use in `sim/scripts/create_balance_car_usd.py`.

### GB37 motor (DC + encoder)
- **Datasheet PDF**: not redistributed. Search "GB37 DC gear motor with encoder" for Yahboom / generic vendor datasheets.
- **Numbers we use**: stall torque 2.17 N·m, no-load speed 34.6 rad/s, 1:30 gear ratio, K_e ≈ 0.0111 V·s/rad.
- **Where they're used**: `--effort-limit 2.17`, `motor_drive_damping=0.063` in `create_balance_car_usd.py`.

### MPU6050 IMU
- **Datasheet**: <https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Datasheet1.pdf>
- **What we use**: gyro full-scale ±2000 dps (16.4 LSB/dps sensitivity), accel ±2g, both at 1 kHz sample rate.
- **Where it's used**: `--imu-pitch-noise-deg`, `--imu-gyro-noise-deg-s`, `--imu-gyro-bias-deg-s` flags in `play_pid_effort.py`.

---

## Firmware references

### Real STM32 firmware (Yahboom control.c)
- **Source**: Taiwan vocational embedded-systems course material (W13 三環並聯 PID), not redistributed.
- **Equivalent open repos** that implement similar STM32 balance-car firmware:
  - <https://github.com/Lichifeng/Arduino-Balance-Car> — Arduino version of the same architecture.
  - <https://github.com/wzhengsen/stm32f103-self-balancing> — STM32 + MPU6050 + L298N.
  - <https://github.com/topics/balance-car?l=c> — many similar reference implementations.
- **What we port**: `Turn()` function structure (Turn_Target × Kp + gyro × Kd, with Kd gain-scheduled on forward/back). See `update_yahboom_controller()` in `sim/scripts/play_pid_effort.py`.
- **Original gains** (when scaled by 100): Balance_Kp = 255, Balance_Kd = 1.35, Velocity_Kp = 160, Velocity_Ki = 0.8, Turn_Kp = 42, Turn_Kd = 0.6.

---

## Course / academic references

### W13 三環並聯 PID 教材 (Taiwan vocational course)
- **Source**: course material from a Taiwan vocational embedded-systems class. Not public.
- **What we port**: three-ring (balance PD + velocity PI + turn PD) controller structure.
- **Where**: `update_threering_controller()` in `play_pid_effort.py`, plus the `--threering-*` flags.

### LQR for inverted pendulum
- **Reference text**: any modern control textbook (Ogata, "Modern Control Engineering"; or Franklin, Powell, Emami-Naeini, "Feedback Control of Dynamic Systems").
- **Our implementation**: `update_lqr_controller()` in `play_pid_effort.py`. Q/R weights derived empirically — see `docs/lqr_tuning_notes.md` (in memory).

---

## How we keep this honest

We don't include downloaded copies of these references in the repo because:

1. **License compatibility** — many of those repos have GPL or "for educational use only" licenses, incompatible with this repo's MIT.
2. **Repo size** — a 270 MB external_refs directory would make `git clone` painful.
3. **Drift** — bundled snapshots get stale. Going upstream means you read the current version.

If you find we've used a pattern from somewhere not credited here, please file an issue — we'll add it.
