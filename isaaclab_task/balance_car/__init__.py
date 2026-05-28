"""Balance car Direct RL task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="TwoWheel-Balance-Direct-v0",
    entry_point=f"{__name__}.env:BalanceCarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:BalanceCarEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceCarPPORunnerCfg",
    },
)

# No-encoder variant: the policy only sees IMU state (pitch / roll / 3 gyros)
# plus the target velocity. Wheel position and wheel velocity observations
# are removed (observation_space drops from 8 to 6). Simulates a cheap
# balance car with only MPU6050, no wheel encoders.
# Expected outcome: pitch balance still learnable, but velocity tracking
# and stationary stability much worse than the encoder variant — the policy
# has no way to observe its own position/velocity error, so it must rely on
# implicit dynamics from IMU readings alone.
gym.register(
    id="TwoWheel-Balance-NoEncoder-Direct-v0",
    entry_point=f"{__name__}.env:BalanceCarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:BalanceCarNoEncoderEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceCarPPORunnerCfg",
    },
)
