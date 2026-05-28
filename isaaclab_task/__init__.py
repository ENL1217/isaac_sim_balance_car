"""Isaac Lab task package for the two-wheel balance car.

Importing this package side-effect registers gym environments. The
companion train wrapper in `scripts/train_lab.py` imports this before
calling Isaac Lab's training pipeline.
"""

from . import balance_car  # noqa: F401  # triggers gym.register
