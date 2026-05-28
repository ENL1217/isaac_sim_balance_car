"""Procedural terrain config sized for the two-wheel balance car.

Scaled DOWN from Flamingo / Anymal because our wheel radius is only 34 mm.
Step heights & slope angles are bounded so the cart's wheels can physically
roll over the geometry.

- pyramid_stairs (down): step 5-25 mm, < wheel radius
- pyramid_stairs_inv (up): same range
- boxes (random grid): 5-20 mm bumps, like cobblestones
- random_rough: small noise terrain
- hf_pyramid_slope (down) / inv (up): 0-17 deg slopes (max 0.3 rad)
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


BALANCE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=8,
    num_cols=16,
    horizontal_scale=0.05,
    vertical_scale=0.002,
    slope_threshold=0.5,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.005, 0.025),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.005, 0.025),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.30,
            grid_height_range=(0.005, 0.020),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.005, 0.020),
            noise_step=0.005,
            border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.0, 0.3),  # 0 - 17 deg
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
