# Hardware Notes

## Known Hardware

The real car is a Yahboom/Arduino-style two-wheel balancing car.

Known components from user:

- Motor driver: TB6612FNG.
- Motors: dual DC gear motors with AB incremental Hall encoders.
- IMU: MPU6050.
- Battery: 3x 18650 plus battery holder.
- Existing controller code: STM32 firmware available from user.

## Yahboom Reference Notes

Checked official Yahboom/YahboomTechnology references for the Arduino balance car. The visual/mechanical baseline should be the Yahboom BST-ABC three-layer balance-car chassis, not a generic inverted-pendulum block.

Official reference photo saved locally:

```text
hardware\references\yahboom_balance_robot_official.jpg
```

Source:

```text
https://www.yahboom.net/study/bc-uno
https://github.com/YahboomTechnology/Arduino-Balance-Car
https://github.com/YahboomTechnology/Arduino-Balance-Car/blob/master/balance_robot.jpg
```

Reference structure to model first:

- White stacked chassis plates.
- Brass standoffs between plates.
- Battery holder/cell pack mounted in the upper stack.
- Arduino UNO/shield or Arduino-style controller board mounted in the middle/lower stack.
- Black anti-skid tires with blue hubs.
- Silver gear motors mounted under the lower plate, inboard of the wheels.
- Both wheels must be on one transverse axle under the chassis; any front/back offset makes the model no longer a two-wheel self-balancing car.
- Encoder/motor wiring and MPU6050/driver board details to be refined from real photos.

Current USD consistency check:

- `sim\assets\two_wheel_balance_car\two_wheel_balance_car.usda` uses two wheel links only, left/right, on one transverse `Y` axis.
- The USD visual style is broadly consistent with the Yahboom official photo: three white plates, brass standoffs, blue wheel hubs, black tires, inboard motor cans, upper battery block, and middle controller board.
- The USD is still approximate: tire tread, exact plate holes, board/shield geometry, wire routing, motor bracket shapes, and mass/inertia values need real measurements or CAD if precision is required.

Yahboom Arduino code reference:

```text
external_refs\Arduino-Balance-Car\3.Arduino balance car code_2019_2_14\bst_abc
```

Important implementation details from the vendor code:

- `MsTimer2::set(5, inter)` runs the main control interrupt every `5 ms`.
- MPU6050 is sampled in the interrupt with `getMotion6`.
- Angle loop runs every interrupt using Kalman-filtered pitch and gyro X.
- Speed PI loop runs every `8` interrupts, about `40 ms`.
- Turn PD loop runs every `>4` interrupts, about `25-40 ms` depending on the counter condition.
- Left/right Hall encoder pulses are counted through Arduino interrupt / pin-change interrupt.
- Motor output is TB6612FNG-style direction pins plus PWM clamped to `[-255, 255]`.
- Fall safety cuts PWM when pitch exceeds about `+/-30 deg`.

## Photo Record

The user provided a chat screenshot of the intended CAD-style vehicle layout, showing a Yahboom/Arduino-style stacked-plate two-wheel balance car. No local file path for the user's physical-car photos is attached or linked in this workspace yet.

Current true Isaac/Omniverse 3D model snapshot:

```text
sim\output\media\balance_car_isaac_3d_snapshot.png
```

This is a generated Yahboom-style primitive USD render, not a real-car photo or official CAD export.

Older annotated model image:

```text
sim\output\media\balance_car_model_photo.png
```

This older PNG is a 2D communication/debug drawing from `render_uncontrolled_fall_media.py`; do not treat it as the 3D model picture.

When the user provides images, record:

- Front, side, top, and bottom views.
- Close-up of motor labels and encoder wiring.
- Close-up of STM32 board, TB6612FNG, MPU6050, and battery placement.
- Local file paths for each photo.

## Modeling Assumptions

Initial geometry can be approximate.

Suggested first assumptions:

| Item | Initial assumption |
| --- | ---: |
| Wheel diameter | `68 mm` |
| Wheel width | `26 mm` |
| Wheel center-to-center track | `168 mm` in the current USD |
| Plate size | Estimate from kit photo until measured |
| Plate thickness | `3-4 mm` |
| Total mass | `0.8-1.2 kg` |
| Battery mass | `135-150 g` cells plus holder |
| Motor gear ratio | Verify, often around `1:30` |

Current simulation-specific assumption:

- The balance axis is forward/back pitch around the wheel axle (`RotateY` in the USD frame).
- The base-link center of mass is explicitly set above the axle at `z = 0.105 m` so the uncontrolled vehicle falls like a two-wheel inverted pendulum.

## Data To Ask User For

Ask for these before precision tuning:

- Actual wheel diameter and width.
- Distance between wheel centers.
- Total vehicle mass with battery.
- Plate length, width, thickness.
- Battery position.
- Motor label or datasheet.
- Encoder counts per motor revolution and per wheel revolution.
- Existing STM32 firmware.
- Real PID gains that can balance the car.

## TB6612FNG Model

For the first simulation:

- Treat motor command as normalized effort `[-1, 1]`.
- Add saturation.
- Later add PWM dead zone and voltage scaling.

Real driver behavior to model later:

- Direction pins.
- PWM duty.
- Supply voltage.
- Dead zone.
- Motor back EMF.
- Current/torque limits.

## MPU6050 Model

For the first simulation:

- Use clean pitch and pitch-rate from simulation.

Later add:

- Noise.
- Bias.
- Low-pass filter.
- Complementary filter or Kalman filter equivalent to firmware.

## Encoder Model

For the first simulation:

- Use wheel joint position and velocity directly.

Later add:

- Counts per revolution.
- Quantization.
- Direction sign.
- Velocity estimation window.
- Missed pulses/noise if needed.
