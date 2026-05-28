# SIM_VS_REAL.md — sim 跟真實 Yahboom STM32 平衡小車的對齊紀錄

> 寫作日期：2026-05-27（C3/C4 commit 期間）
> 目的：把「sim 模型的每個物理數字來自哪份真實規格」記下來，避免後續再
> 邊寫邊猜。任何修改 USD 或控制器都先回到本文件確認來源。
> 配對文件：`docs/HONEST_REPORT.md`（誠實的測試結果）、
> `sim/scripts/create_balance_car_usd.py`（USD 產生器，本文件的權威實作）。

---

## 0. 為何需要這份文件

C0–C2 期間 (2026/05 前半) 我們的 USD 是「手寫 + 一邊測試一邊改」做出來
的。結果是 sim 跟真實行為有 4–10 倍差距，我們用「軟體 ramp / brake / gate /
anti-windup」一層層補丁去 cover 物理上的不正確，最後拉到一個能跑但跟真實
完全不像的狀態。

C3 (commit `c41e276`) 重新從真實規格 datasheet 出發重建 USD，目前在常用
場景達到 ≈ 89% 行為對齊。本文件記每個數字怎麼來。

如果未來要改任何物理量，先檢查這裡有沒有引用某份真實 source；如果有，請
連 source 一起改。

---

## 1. 真實 Yahboom 規格來源

| 規格 | 來源 |
|---|---|
| 產品 spec 圖（總重 942g、尺寸 194×84×139.59 mm、馬達 333±10 rpm、編碼器 11 線、續航 20h/5h、車材 金屬+PCB+亞克力、車身組合圖） | Yahboom 產品頁照片，使用者 2026-05-27 在 chat 上傳 |
| 馬達 datasheet（GB37 直流減速電機，12V、330rpm 減速前、250mA 空轉、4.8W、6.5A 堵轉、1:30 齒輪、22mm 齒輪箱） | `Y:\DellNB\D\BaiduNetdiskDownload\亚博智能 STM32平衡车\亚博智能 STM32平衡小车20220803\STM32\3.硬體資料\6.GB37電機\GB37 帶測速碼盤.pdf` |
| 編碼器 spec（30×11×2 = 780 pulses/輪轉，A 相單獨 = 330） | 同 GB37 datasheet |
| STM32 控制律參考碼 | `sim/output/_ref_upstandingcar.c`（從同一個 Yahboom 教材 ZIP 抽出） |
| MPU6050 IMU | 產品 spec 圖上的 "MPU6050 六軸傳感器模塊" 標示 |
| 控制電路板架構（OLED, Bluetooth 5.0, drive board, MPU6050, 超聲波, 大摩擦力防滑輪胎, 封閉電池倉, 金屬底盤） | 產品 spec 爆炸圖 |
| 馬達 stall torque 算出 | 用 datasheet 數字反推 DC 馬達常數，見 §2.2 |

---

## 2. 物理數字推導

### 2.1 質量分配（總 942 g）

從產品 spec 直接給的車重 = **942 g**。爆炸圖顯示主要組件：
封閉式電池倉、PCB 驅動板、MPU6050、超聲波模塊、OLED、亞克力上下蓋板、
金屬底盤、雙馬達、雙輪。

對應 sim：

| 組件 | sim 數字 | 依據 |
|---|---|---|
| `chassis_mass_kg` | **0.822** | 942g 總重 − 2×60g 輪子 |
| `wheel_mass_kg` (each) | **0.060** | 橡膠胎 + 塑膠輪殼 + 軸接頭典型重量 |
| sim 總重 | **0.942 kg** | 對齊真實 ✓ |

### 2.2 馬達特性（GB37 datasheet 反推）

從 GB37 datasheet：
- 額定電壓 V = 12 V
- 減速前轉速（即輸出端轉速） ω_no_load = 330 rpm = **34.6 rad/s**
- 空轉電流 I_no_load = 0.25 A
- 堵轉電流 I_stall = 6.5 A
- 齒輪比 N = 1:30（馬達 → 輸出端）

標準 DC 馬達公式 V = K_e·ω_motor + I·R + L·dI/dt（穩態忽略 L）：

| 推導 | 算式 | 結果 |
|---|---|---|
| 內阻 R | V / I_stall = 12 / 6.5 | **1.85 Ω** |
| 馬達側無載 ω | 330 rpm × 30 × 2π/60 | **1037 rad/s** |
| 反電動勢常數 K_e (馬達側) | (V − I_no_load·R) / ω_motor_max = (12 − 0.463) / 1037 | **0.0111 V·s/rad** |
| 馬達側堵轉扭矩 | K_e × I_stall = 0.0111 × 6.5 | **0.0723 N·m** |
| **輸出端堵轉扭矩** | × 30（齒輪倍率） | **2.17 N·m** |
| **輸出端無載 ω** | 1037 / 30 | **34.6 rad/s** |
| 反電動勢係數（輸出端） | stall / no_load_ω = 2.17 / 34.6 | **0.063 N·m·s/rad** |
| 額定持續扭矩（≈ 4.8W 處） | P_rated / ω_rated ≈ 4.8 / 24 | **~0.2 N·m** |

對應 USD：

| sim 量 | 數值 | 對齊 |
|---|---|---|
| `motor_drive_max_force`（joint maxForce）| **2.17 N·m** | = GB37 輸出端 stall ✓ |
| `motor_drive_damping`（joint damping）| **0.063** | = K_back_EMF，讓 joint 自然生成 linear 馬達曲線 ✓ |
| 內建頂速 | 自動 34.6 rad/s | back-EMF 在 ω=34.6 抵銷 stall ✓ |

**為何用 joint damping 模擬 back-EMF**：PhysX joint drive 的力學是
`τ = stiffness·(target_pos − pos) + damping·(target_vel − vel)`。
我們設 `stiffness=0`, `target_vel=0`，於是 drive 永遠施加
`τ_drive = −damping × ω`。這就是 back-EMF。
當外部 controller 命令 +2.17 N·m，joint drive 應用 −0.063×ω，淨扭矩 = 2.17 − 0.063×ω。
ω=34.6 時淨扭矩 = 0，車不能再加速。**這完美對應 DC 馬達的 linear 扭矩-速度曲線**。

### 2.3 幾何尺寸

從產品 spec 圖（dimension drawing）：

| 量 | 真實 | sim USD 對應 | 對齊 |
|---|---|---|---|
| 整體寬（沿輪軸 Y，含輪子）| 194 mm | wheel_track 0.170 + wheel_width 0.030 × 2 = 0.230... 實際輪子在邊上突出，視覺對齊只到車身範圍 | 近似 |
| 車身寬 Y（兩輪內側之間） | ~140 mm | plate_xyz_m[Y] = 0.140 | ✓ |
| 車身深 X（前後） | 84 mm | plate_xyz_m[X] = 0.084 | ✓ |
| 車身高 Z（地面 → 頂蓋） | 139.59 mm | upper_plate_z (rel) 0.106 + axle_z 0.0335 = 0.140 | ✓ |
| 輪 OD | 67 mm | wheel_radius_m × 2 = 67 mm | ✓ |
| 輪寬（沿軸 Y） | ~30 mm | wheel_width_m = 0.030 | ✓ |

**plate 方向**：C3 commit 把 plate_xyz_m 設 (0.140, 0.084) 即 X=140, Y=84，
**這是錯的**（外型旋轉 90°）。C4 commit 修正為 (0.084, 0.140) 對齊真實。

### 2.4 質心高度（CoM）

C0 USD 設 CoM 在 chassis-local (0, 0, 0.105) → 絕對 z = 0.0335 + 0.105 = **139 mm**。
這比車身頂蓋還高，物理不可能。

爆炸圖顯示電池倉在 **底層**（馬達高度上方一點），PCB 驅動板在中層，OLED + 超聲波
在上層。電池是車內最重的單一組件（2200mAh 12.6V 鋰電池組 ≈ 180–220 g），dominate CoM。

C3 USD 設 CoM 在 chassis-local (0, 0, 0.021) → 絕對 z = **54.5 mm**。
這跟以「battery 在底」估算的真實 CoM 一致。

**影響**：CoM 從 139 mm 降到 54.5 mm，gravity tipping torque 從 0.014 N·m/° 降到 0.003 N·m/°
（5× 變小）。車**變得很穩**，但相對的，給定 lean angle 推進力也變小。所以
C3 把 `drive_pitch_offset_deg` 從 3° 拉到 8° 補回 lean torque。

### 2.5 摩擦係數

C0 USD 設 μ_static=1.6, μ_dynamic=1.4。沒有 source — 試出來能跑就用。
這超過大部分「橡膠對地」的真實值。

C3 USD 改 μ_static=**0.9**, μ_dynamic=**0.8**。對齊
`campus-wheel-legged-robot/SDD.md §3.5`（ground_friction 0.8 baseline，DR 0.3–1.0）。
真實 Yahboom 在磁磚/木板上實測大約這個範圍。

### 2.6 編碼器

從 datasheet：「30×11×2 = 780 pulses」per 輪轉。對應 STM32 firmware 的
A+B 兩相邊緣計數方式（不是完整 quadrature）。

sim `--encoder-cpr 780` = 對齊 ✓

注意：純 quadrature 計數會是 30×11×4 = **1320**。如果想對齊「完整 quadrature」
要把 `--encoder-cpr` 改成 1320。但 STM32 firmware 是 780 模式，我們對齊它。

### 2.7 IMU

真實：MPU6050（6 軸：3 軸加速度 + 3 軸陀螺）。MPU6050 dataset 給的雜訊：
- gyro bias: 約 ±0.05 °/s（25°C）
- gyro 雜訊 PSD: ~0.005 °/s/√Hz
- 量化：16-bit @ ±2000 °/s 範圍

sim **目前無 IMU 模型**。我們直接從 PhysX rigid body matrix 抽 pitch/yaw/roll，
這是 ground truth 完美值。差距見 §4。

---

## 3. 控制律對齊

### 3.1 W13 教材三環並聯 vs 我們的 Yahboom controller

| 環 | 教材 spec (`IsaacSim_平衡車控制_v1_三環並聯.md`) | 我們 yahboom controller | 對齊 |
|---|---|---|---|
| Balance | PD on (pitch_deg, gyro_x_deg/s), D 直讀 gyro | 同（line 690–693 in play_pid_effort.py） | ✓ |
| Velocity | PI on encoder pulses + LPF α=0.86 + position-form integral + clamp | PI + LPF α=0.7 + stall-decay anti-windup | 結構同，係數略不同 |
| Turn | PD with gain-scheduled Kd | 直接 PWM 差速 + smooth speed taper | 結構不同（Yahboom STM32 reference 是這樣） |

教材跟 STM32 reference 對 turn 的處理不同：教材寫 PD with gain-scheduling，
reference C 代碼是直接 PWM 差速。我們 yahboom controller 跟 STM32 reference 對齊。
三環版本（`--controller threering`）跟教材對齊。

### 3.2 簽號慣例

USD asset 的 pitch 慣例：`body_angles()` 回傳 pitch = -RotateY(degrees)。
**負 pitch = 車前傾**（top 向 +X）。這跟「一般直覺 positive pitch = forward」相反。

Yahboom controller 在合成 motor PWM 時用 global negation 修正：
`pwm_left = -angle - speed - turn`。

如果未來換 controller，請先確認 pitch sign 慣例，否則第一步就會把車推倒。
memory note：`memory/drive_pitch_offset_decision.md`

### 3.3 drive_pitch_offset 是 sim hack

真實 Yahboom 不需要 `drive_pitch_offset_deg`。它的速度回授靠：
- 馬達 stall torque 提供推進權威
- 編碼器量化雜訊隱性穩定 PI 積分
- 操作者持續調整指令

sim 在 C0–C2 期間 PID 達不到 drive authority，所以加 drive_pitch_offset 偏移
angle setpoint 強制車前傾。詳細推理在 `memory/drive_pitch_offset_decision.md`。

C3 後馬達曲線對齊真實，drive_pitch_offset 仍然需要（因為 sim 仍無 sensor noise
+ 真實摩擦變動）。長期目標是加 DR 後可以拿掉這個 hack。

---

## 4. 目前剩餘的 sim ≠ real 差距

對應 §1–§3，已對齊的我們不重複。**還沒對齊**的：

| 項目 | 真實 | sim | 對 sim2real 的影響 | 修法 |
|---|---|---|---|---|
| IMU 雜訊 | MPU6050 chip 雜訊 + bias drift | **C7 已加**：`--imu-pitch-noise-deg`, `--imu-gyro-noise-deg-s`, `--imu-gyro-bias-deg-s`。Headless 5 分鐘 stand 穩。**GUI mode 反而會加速 lateral drift**（PhysX GUI solver 跟 noise 互動異常），所以 GUI default 關閉、headless/RL 訓練建議打開 | 已實作但 GUI 暫不啟用 |
| 編碼器量化 | ±1 count | **C7 已加**：`--encoder-quantize` 把 joint 位置 round 到整數 Hall count | 已實作 |
| 馬達 dead-band | PWM < ~10% 不動 | **C5 已加**：`--pwm-dead-band` 預設 8/255 ≈ 3%。模擬 TB6612FNG H-bridge 啟動閾值 | 已實作 |
| 馬達 thermal limit | 持續 stall ≈ 78W 熱衰減 | 永遠 2.17 N·m | sim 可以無限 stall | thermal model（可選） |
| 控制迴路 delay | 真實 200 Hz STM32 主迴路 + CAN/I2C 通訊 ~5 ms 延遲 | sim 即時 | sim 響應比真實快一點 | DR：observation/action delay 10 ms |
| Wheel rolling friction | 真實有 motor 軸承 + 齒輪滯後 ~0.02 N·m | sim 馬達 joint damping 已模擬 back-EMF，但沒額外軸承摩擦 | 微差 | 加 joint armature + friction |
| 場景隨機化 | 真實地面有不平、塵土、邊緣 | 完美平面 | RL policy 過擬合 sim | 加 ground 高度 noise + 邊緣磨損 |

完整 sim2real 對齊（≥ 95%）需要至少加 IMU/encoder noise + motor dead-band，
這在路線圖上是「短期可做、單一 PR 完成」。

---

## 5. Sim → Real 對齊評分

對齊度自評（基於 §1–§4）：

| 維度 | C3 | C7 (現況) |
|---|---|---|
| 質量 / 慣量 / 幾何 | 100% | 100% |
| 馬達靜態特性（torque, ω_max） | 100% | 100% |
| 馬達動態特性（dead-band, thermal） | 60% | **85%**（dead-band 加在 C5） |
| 感測（pitch, gyro, encoder） | 40% | **85%**（noise + quantize 加在 C7） |
| 摩擦 | 80% | 80% |
| 控制律邏輯 | 80% | 80% |
| 物理數值穩定性（GUI 站不倒） | 不適用 | C6 ~180s（40mm 輪），C7 加 noise 反而 GUI 30s（headless 5min+） |
| **綜合 sim2real readiness** | ~75% | **~85%** |

C3 commit message 寫的是「~90% behavioral parity on flat ground」，
那是針對特定測試（從 spawn 衝刺到 seesaw）的行為對齊度。
本表是整體 sim2real readiness，更嚴格。

要達 95% 還需：
1. Mass/CoM/friction domain randomization（每 reset 抽不同值）
2. Control loop delay 模擬（10–50 ms 延遲）
3. 為什麼 IMU noise 在 GUI 反而加速 drift（debug 中）— 可能要降低 default σ 或改用不同 PhysX solver param

---

## 6. 變更紀錄

| 日期 | commit | 主要對齊改動 |
|---|---|---|
| 2026-05-27 | bd87024 (C1) | 起始狀態：手寫 USD，無 sim2real 對齊意識 |
| 2026-05-27 | 21a021d (C2) | 加 W13 三環 PID（gain 對齊不正確） |
| 2026-05-27 | c41e276 (C3) | 重做 USD：質量、CoM、馬達曲線、摩擦、編碼器對齊真實 |
| 2026-05-27 | 1b74263 (C4) | 修 plate XY 方向；explicit inertia tensor；turn-gate 改 smooth taper |
| 2026-05-27 | 7bec0fc (C5) | Joint Coulomb friction + PWM dead-band + 短 release tau |
| 2026-05-27 | d2b83a5 (C6) | 輪胎 30→40 mm 解 PhysX 接觸 numerical instability（GUI 站不倒從 16s 拉到 180s） |
| 2026-05-27 | baa7492 (C7) | IMU noise + encoder quantize 旗標（GUI default 關，headless/RL 推薦開） |
| 未來 | – | DR (mass/CoM/friction) + control delay → 預計達 95% sim2real readiness |
