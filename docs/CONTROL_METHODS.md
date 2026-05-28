# 控制方法總覽 / Control Methods Reference

> For students: 5 種控制方法，從最簡單的 PID 到 PPO 強化學習。每個都附了「**它在做什麼**」「**程式碼在哪**」「**何時用**」「**它失敗時長怎樣**」。

---

## Launcher 對照表（雙擊即可跑）

| 控制方法 | Launcher (`.bat`) | 用途 |
|---|---|---|
| **PID（預設，3-loop）** | **`launch_pid.bat`** | 起點，最直觀的控制器 |
| PID（沒編碼器版） | `launch_no_encoder.bat` | 教學「沒編碼器會怎樣」 |
| LQR | `launch_lqr.bat` | 學狀態空間最優控制 |
| LQI | `launch_lqi.bat` | LQR + integral，補 steady-state 偏差 |
| 三環並聯 PID（W13 教材版） | `launch_threering.bat` | 跟 W13 課程教材逐字對照（教學用，gain 還沒完全調好） |
| 訓練 PPO（有編碼器版） | `launch_train_encoder.bat` | 自己跑 RL 訓練（~15 分鐘） |
| 訓練 PPO（沒編碼器版） | `launch_train_no_encoder.bat` | 同上但沒編碼器（教學對照） |
| 跑 PPO 訓練好的（有編碼器） | `launch_eval_rl_encoder.bat` | 看 RL 表現 |
| 跑 PPO 訓練好的（沒編碼器） | `launch_eval_rl_no_encoder.bat` | 對比沒編碼器的 RL |

所有 launcher 都在 repo 根目錄。雙擊即可（前提是已執行過 `setup_env.bat`）。

**建議第一次就跑 `launch_pid.bat`**，之後再回來讀下面細節。

---

## 控制問題本身

平衡車是 **倒立擺 + 雙輪驅動**，狀態變數有 4 個：

| 符號 | 意義 | 來源 |
|---|---|---|
| θ | Pitch 傾角（rad）— 車身偏離垂直的角度 | IMU (MPU6050) |
| θ̇ | Pitch 角速度（rad/s）| IMU gyro_y |
| x | 車輪累積位置（m）| 編碼器（GB37 motor encoder）|
| ẋ | 車輪速度（m/s）| 編碼器（微分）|

控制目標：

1. **平衡**：θ → 0 （車不倒）
2. **速度**：ẋ → target_velocity（W 鍵命令前進、S 鍵命令後退）
3. **轉向**：ψ̇ → target_yaw_rate（A/D 鍵命令轉向）

輸入：左右輪的馬達 PWM（→ 力矩）。控制器要決定每個瞬間的 PWM 值。

---

## 方法 1：PID（三環並聯）— 預設

**Launcher**：`launch_pid.bat` → `--controller yahboom`（標準版，有編碼器）
**Variant**：`launch_no_encoder.bat`（同 controller 但 `--no-encoder`，把 wheel pos/vel feedback 歸零，變成「IMU only」雙環版）
**程式碼**：`sim/scripts/play_pid_effort.py:update_yahboom_controller()`

### 架構（從真實 STM32 韌體 port）

三個獨立的 P/PI/PD 環，相加後輸出 PWM：

```
Balance loop  (PD on pitch):
  angle_output_pwm = Kp_balance * (pitch + offset) + Kd_balance * gyro_y

Speed loop    (PI on wheel position + velocity):
  position_pulses += speed_filter  (積分)
  speed_output_pwm = Ki_speed * (0 - position_pulses) + Kp_speed * (0 - speed_filter)

Turn loop     (direct PWM differential + Kd damping on yaw rate):
  turn_output_pwm = magnitude × turn_scale − Kd_turn × gyro_z

Combine:
  pwm_left  = -angle_output_pwm - speed_output_pwm - turn_output_pwm
  pwm_right = -angle_output_pwm - speed_output_pwm + turn_output_pwm
```

### 預設增益（從真實韌體拿來除以 100）

```
Kp_balance = 255    Kd_balance = 1.35
Kp_speed   = 160    Ki_speed   = 0.8
Kp_turn    = 42     Kd_turn    = 0.6
```

### 何時用 PID

- **絕大多數教學情境** — 它最直觀，每個環的物理意義清楚
- 平地、緩坡、簡單轉向 — 表現很穩
- 需要對應到真實小車控制器 — 我們的 PID 直接 port 自實機韌體

### 失敗時長怎樣

- 樓梯：car 過台階時 angle PID 會 saturate，pitch 偏離後拉不回來 → 翻車
- 蹺蹺板：能過支點翻板，但翻完出口側的「突然下落」會讓 cart 過衝 → 偶爾 roll 翻
- Gravel 地形：單軸 roll 不穩，物理層級就過不了

### 進階參數

PID 有很多可調 flag（在 `launch_pid.bat` 裡）：

| Flag | 預設 | 意義 |
|---|---|---|
| `--effort-limit` | 2.17 | 馬達最大力矩（N·m）— 真實 GB37 stall torque |
| `--drive-pitch-offset-deg` | 8 | 按 W 時讓車「主動前傾」幾度才會走 |
| `--bluetooth-direction-magnitude` | 30 | A/D 轉向時的 PWM 差分量 |
| `--turn-kd-pwm-per-dps` | 0.364 | yaw rate 阻尼（μ rad/s 變多少 PWM）|

---

## 方法 2：LQR — 最優線性控制

**Launcher**：`launch_lqr.bat` → `--controller lqr`
**程式碼**：`sim/scripts/play_pid_effort.py:update_lqr_controller()`

### 架構

不是分環，而是把車當成 **線性化的狀態空間系統**：

```
state = [pitch, pitch_rate, wheel_pos_error, wheel_vel_error]
control = K × state    # K 是 4×1 矩陣（每個 state 一個 gain）
```

K 矩陣從 **Algebraic Riccati Equation** 解出來，最小化代價函式：

```
J = ∫ (state^T Q state + control^T R control) dt
```

`Q` 跟 `R` 是學生可調的「想多在乎哪個 state」「想多省力」權重。我們的預設：

```
Q_pitch     = 5000   ← 最在乎 pitch（不能倒）
Q_pitch_rate = 100   ← 次在乎角速度
Q_pos       = 1      ← 位置誤差小事
Q_vel       = 0.5    ← 速度誤差更小事
R_torque    = 0.1    ← 用力的代價
```

### 何時用 LQR

- **教控制理論** — 它直接對應書本上的「state-space + Riccati」章節
- **斜坡** — LQR 自動含 gravity 補償（線性化模型內建），比 PID + feed-forward 更乾淨
- 想看「不調 PID 也能站」 — LQR 給你一組「數學最優」的增益

### 失敗時長怎樣

- **沒有 integrator** → 斜坡 steady-state 有偏差，車會慢慢漂走
- 障礙物：linearization 假設小角度，過樓梯時 pitch 太大就失效
- 程式碼裡用 `--slope-feedforward` flag 才能爬陡坡，否則卡住

---

## 方法 3：LQI — LQR 加 integral

**Launcher**：`launch_lqi.bat` → `--controller lqi`
**程式碼**：`sim/scripts/play_pid_effort.py:update_lqi_controller()`

### 跟 LQR 差別

加了一個 **integral state**：

```
state = [pitch, pitch_rate, wheel_pos_error, wheel_vel_error, ∫vel_error dt]
```

那個多出來的 ∫vel_error 就是「速度誤差累積多少」，讓 K 矩陣可以「記得」過去的偏差，自動把 steady-state error 拉到 0。

### 何時用 LQI

- 你跑 LQR 發現「會慢慢飄」就用 LQI
- 想教 PI control 的「integral action 為什麼有用」最好的對比

### 失敗時長怎樣

- Integral windup：integrator 撞到 clamp 後反應慢
- 樓梯、蹺蹺板：跟 LQR 一樣 linearization 不夠

---

## 方法 4：三環並聯 PID（W13 教材版）

**Launcher**：`launch_threering.bat` → `--controller threering`
**程式碼**：`sim/scripts/play_pid_effort.py:update_threering_controller()`

### 跟方法 1 的 PID 差別

兩個都是**三環並聯 PID**，數學上 ~99% 等價。差別在來源跟細節：

| 項目 | 方法 1 (`launch_pid.bat`) | 方法 4 (`launch_threering.bat`) |
|---|---|---|
| 來源 | 真實 STM32 韌體 port | W13 課程教材 spec 逐字實作 |
| Kp_balance | 255 | 250 |
| 速度環 anti-windup | stalled 判斷（看 wheel 是否真的在動） | stopflag（看是否還在按 W） |
| PWM saturation | 255 | 3000（縮放後） |
| Turn 公式 | direct PWM + Kd*gyro 阻尼 | PD on (target_yaw_rate − gyro_z) |

### 何時用三環版

- **跟 W13 課程簡報逐字對照**做習題時
- 比較「同樣架構、不同細節」會不會有差

### 何時不用

- **想實際跑障礙物**：用方法 1（threering 還沒完整調好，W 不太走）
- **學基本控制**：方法 1 更穩，先學那個

### 失敗時長怎樣

- 跟方法 1 一樣的物理限制
- 額外問題：目前 gain 偏保守，按 W 推力不夠 → cart 幾乎不前進（task #56 正在處理）

---

## 方法 5：PPO（強化學習）

**Launcher**：
- 訓練：`launch_train_no_encoder.bat`（或在 Isaac Lab 跑訓練 task）
- 評估：`launch_eval_rl_encoder.bat` / `launch_eval_rl_no_encoder.bat`

**程式碼**：`isaaclab_task/balance_car/env.py` + Isaac Lab rsl_rl pipeline

### 架構

不是寫公式，而是讓 RL agent 自己學：

```
Observation (encoder version, 8-dim):
  [pitch, pitch_rate, roll, yaw_rate, roll_rate, x_wheel, x_wheel_dot, target_vel]

Action (2-dim):
  [left_wheel_torque, right_wheel_torque]

Reward:
  + 速度跟蹤獎勵 (track linear vel)
  + 方向匹配獎勵 (dir match)
  − 角度偏離扣分 (flat orientation)
  − 翻倒扣分 (terminated)
  + 還活著的小獎勵 (alive bonus)
```

PPO 用 **4096 個平行模擬環境**（每個有不同地形）同時訓練，~20-30 分鐘在 RTX 4070 可以收斂。

### 何時用 PPO

- **想過所有障礙** — PPO 是唯一能穩定過樓梯+蹺蹺板的方法
- 教學機器學習如何「自己發明控制律」
- 想看「資料驅動 vs 模型驅動」的對比

### 失敗時長怎樣

- 訓練前 100k steps 看起來像隨機亂動（policy 還沒學會平衡）
- 沒有 encoder 訓練的版本（`--no-encoder` 變體）— velocity tracking 變差
- 跨 domain 遷移（sim → real）需要 domain randomization 才行

### No-encoder 變體

我們提供兩種 PPO 訓練：

| 版本 | Observation | Task ID |
|---|---|---|
| **有 encoder** | 8 dim（含 wheel_pos + wheel_vel） | `TwoWheel-Balance-Direct-v0` |
| **沒 encoder** | 6 dim（純 IMU + target）| `TwoWheel-Balance-NoEncoder-Direct-v0` |

對比目的：證明 RL 也跟 classical 一樣 — **沒有編碼器就學不會穩定的位置控制**。教材重點。

---

## 怎麼選

```
我要：
├── 學基本控制 → PID (方法 1)
├── 學最優控制理論 → LQR (方法 2)
├── 學 integrator 為什麼有用 → LQI (方法 3)
├── 跟著 W13 教材對齊 → 三環 (方法 4)
└── 看 ML 怎麼學控制 → PPO (方法 5)
```

實驗順序建議：

1. **方法 1 PID** — 在 `flat` 地形跑通 → 在 `combined` 地形看撞到障礙倒
2. **方法 2 LQR** — 同樣兩個情境跑一次 → 對比 PID 的差別（LQR 在斜坡更穩）
3. **方法 3 LQI** — 看 integrator 如何修正 steady-state 偏差
4. **方法 5 PPO** — 訓練一個，眼見為憑「資料驅動真的能過所有障礙」
5. **方法 4 三環** — 想精確跟教材對照再回來看

---

## 看 controller 跑得如何

每跑一次任何 launcher，CSV 寫到 `sim/output/pid_effort_timeseries.csv`，主要欄位：

| Column | 意義 |
|---|---|
| `time_s` | sim 時間 |
| `x_m` / `z_m` | 車的世界位置（z 突然降表示掉下去）|
| `pitch_deg` / `roll_deg` | 車身姿態（±35° 觸發 fall）|
| `wheel_velocity_m_s` | 平均輪速 |
| `left_effort` / `right_effort` | 馬達實際力矩（N·m）|
| `angle_output_pwm` | balance 環輸出 |
| `speed_output_pwm` | speed 環輸出 |
| `turn_output_pwm` | turn 環輸出 |
| `controller` | 用哪個 controller |
| `plank_pitch_deg` | 蹺蹺板實際角度（驗證有沒有翻）|

用 Excel / pandas 打開分析。**每次跑完先看 CSV 再看視覺**，視覺有時會騙人。

---

## 共同的「邊界條件」

不論哪個方法，這些是物理硬限制 —— **調 controller 解不開**：

1. **單軸雙輪天生 roll 不穩** — 碎石地形不可能撐久，這是幾何限制
2. **馬達飽和** — `--effort-limit 2.17` 是真實 GB37 stall torque；超出這個力矩需要的場景就過不了
3. **PhysX cylinder collider 邊緣 trip** — 過尖銳邊角會跳，用薄板（plate < 10% wheel radius）緩解
4. **沒編碼器 → 沒位置/速度回授** — 任何 controller（含 RL）都會漂移

詳見 `docs/HONEST_REPORT.md` — 我們把所有 controller × scene 跑過一遍寫成 matrix。

---

## 已知 limitation / 待做：line-following RL

**目前所有 PPO 都學會「站著不動」**（無論有沒有編碼器）。原因：
- 我們的 reward shaping 有獎勵速度跟蹤，但**站著比較容易拿到平衡獎勵**
- Policy 收斂到「站著」這個 local optimum 就不想動了

要修這個有兩條路：

**Path A**：把 `rew_track_lin_vel` 從 5 拉到 50+，逼 policy 必須移動才能拿夠 reward。簡單但暴力。

**Path B（更有教育意義）**：改成「**循跡 + 到達終點**」任務 — observation 加 lateral_error (cart 偏離黑線多少) + yaw_error，reward 改成「跟蹤黑線 + 越靠近終點越多」。
- 學生看得到 cart 沿黑線跑，比「跟蹤抽象的 velocity 命令」直觀
- 站著不動 = 拿不到終點獎勵，policy 一定要前進
- 對應到 `--line-follow` flag（classical controller 已支援）

**目前狀態**：B 還沒實作。如果你想做，我可以加：
- 新 task `TwoWheel-Balance-LineFollow-Direct-v0`
- `launch_train_linefollow.bat` + `launch_eval_rl_linefollow.bat`
- 訓練環境用 `line_world` 或 `combined_course` 上的黑線

工程量：~2-3 小時實作 + 15-30 分鐘訓練。

---

## 學生小作業（建議）

如果你想動手練，這幾個是好的：

1. **改 PID 的 Kp_balance** — 從 255 改成 100、500，看效果有什麼差
2. **改 LQR 的 Q matrix** — Q_pitch 從 5000 改成 1000，看 cart 在斜坡的反應
3. **訓練自己的 PPO** — 改 reward 權重重新訓練，看 policy 學出什麼
4. **加新的 controller** — 例如 MPC（模型預測控制），plug-in 到 `play_pid_effort.py` 的 controller dispatch
5. **加新的 scene** — 仿照 `sim/scripts/create_*.py` 寫一個自己的障礙場景

每個小作業都附 CSV 證明你的改動有效。
