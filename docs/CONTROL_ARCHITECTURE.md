# Control Architecture

## Concept

The balancing car should be built around a replaceable controller interface:

```text
state + command -> controller -> left/right motor command
```

This lets the project compare PID, LQR, MPC, and RL on the same robot.

## State Vector

Initial common state:

```text
pitch
pitch_rate
left_wheel_position
right_wheel_position
left_wheel_velocity
right_wheel_velocity
target_velocity
target_yaw_rate
```

Optional:

```text
battery_voltage
estimated_body_velocity
estimated_position
motor_current_left
motor_current_right
```

## Motor Command

Initial simulation command:

```text
left_motor_command
right_motor_command
```

Interpretation by backend:

- Isaac Sim early version: torque command or normalized effort.
- Motor model version: PWM-like command with saturation, dead zone, delay.
- Real STM32 version: TB6612FNG direction pins plus PWM duty.

## PID Structure

Yahboom vendor-controller structure:

```text
5 ms interrupt:
  read MPU6050 ax/ay/az/gx/gy/gz
  estimate pitch with Kalman filter
  angle_output = kp_angle * pitch + kd_angle * gyro_x
  accumulate left/right Hall encoder pulses

~40 ms speed loop:
  speed = left_pulses + right_pulses
  speed_filter = 0.7 * old_speed_filter + 0.3 * speed
  position += speed_filter + forward_command + back_command
  speed_output = ki_speed * (target_position - position)
               + kp_speed * (target_speed - speed_filter)

~25-40 ms turn loop:
  turn_output = -turn_accumulator * kp_turn - gyro_z * kd_turn

motor mixing:
  left_pwm  = -angle_output - speed_output - turn_output
  right_pwm = -angle_output - speed_output + turn_output
  clamp pwm to [-255, 255]
  cut motor if abs(pitch) > 30 deg
```

Vendor default gains in the Arduino reference:

```text
kp_angle = 38
kd_angle = 0.58
kp_speed = 3.8
ki_speed = 0.11
kp_turn = 28
kd_turn = 0.29
```

Simple normalized simulation starting point:

```text
balance_error = target_pitch - pitch
balance_output = PID(balance_error)

speed_error = target_velocity - wheel_velocity_average
speed_output = PID(speed_error)

yaw_error = target_yaw_rate - yaw_rate_estimate
yaw_output = PID(yaw_error)

base_command = balance_output + speed_output
left_motor = base_command - yaw_output
right_motor = base_command + yaw_output
```

The intended real-car mapping is:

```text
MPU6050 attitude estimate:
  pitch, pitch_rate
  -> inner upright loop

Hall encoder wheel feedback:
  left/right wheel position
  left/right wheel velocity
  -> speed, position, yaw outer loops
```

Manual PID tuning and automatic PID gain search can run in Isaac Sim standalone scripts. Isaac Lab is not required just to learn `Kp`, `Ki`, and `Kd`; it is better reserved for later RL, parallel environments, domain randomization, and curriculum learning.

## LQR Structure

Candidate state:

```text
x = [pitch, pitch_rate, wheel_position, wheel_velocity]
u = motor_torque
```

For differential drive:

```text
u_balance = LQR(x)
u_yaw = yaw_controller(target_yaw_rate)
left = u_balance - u_yaw
right = u_balance + u_yaw
```

The model should be linearized around upright equilibrium.

## MPC Structure

MPC can use the same state as LQR but adds constraints:

- Maximum motor command.
- Maximum pitch.
- Maximum wheel velocity.
- Soft velocity tracking.

MPC is optional after PID/LQR.

## RL Structure

Start safer:

```text
RL action = target_pitch_offset or motor command correction
STM32/PID/LQR remains safety stabilizer
```

More aggressive later:

```text
RL action = left/right motor command
```

Do not deploy direct raw RL commands to the real car without safety gating.

## Backend Targets

Simulation:

```text
Python controller -> Isaac Sim articulation command
```

Real car:

```text
Jetson/ROS2 high-level command
  -> STM32 controller mode/gains/target
  -> TB6612FNG PWM
  -> GM37 motors
```

The STM32 remains responsible for hard safety.
