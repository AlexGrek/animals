"""Custom SB3 feature extractor for the grid-based observations.

Each actor's observation is a flat vector that actually contains two co-located
8x8 spatial grids (an entity/terrain grid and a grass-health grid) plus a few
scalar features wedged between/after them. The default SB3 `FlattenExtractor`
throws away that spatial structure — the MLP has to relearn from scratch that
cell (r, c) in one grid is the same location as cell (r, c) in the other.

`GridCnnExtractor` reshapes the two grids into a 2-channel 8x8 image, runs a
small CNN over them, and concatenates the raw scalar features. The result is fed
to the usual `net_arch` pi/vf heads. Grid slices come from `constants.py` and
mirror the Rust obs builders in `animals_engine/src/game.rs`.

The snake observation additionally carries a third grid (`grid3`, see
`SNAKE_GRID3`): an 8x8 coarse visitation-recency grid at a different spatial
scale (8 tiles/cell vs 1 tile/cell for grid1/grid2), so it's run through its
own small conv branch rather than stacked as a channel of the fine-grid image
— stacking would spatially misalign the two scales. Prey/amphibia models don't
pass `grid3` and get the original two-grid behavior unchanged.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from learner.constants import GRID_H, GRID_W


class GridCnnExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        grid1: tuple,
        grid2: tuple,
        grid3: tuple | None = None,
        cnn_features: int = 128,
    ):
        obs_dim = int(observation_space.shape[0])

        # Indices covered by the spatial grids; everything else is a scalar.
        grid_idx = set(range(grid1[0], grid1[1])) | set(range(grid2[0], grid2[1]))
        if grid3 is not None:
            grid_idx |= set(range(grid3[0], grid3[1]))
        scalar_idx = [i for i in range(obs_dim) if i not in grid_idx]
        num_scalars = len(scalar_idx)

        cells = GRID_H * GRID_W
        assert grid1[1] - grid1[0] == cells, "grid1 must be exactly one 8x8 grid"
        assert grid2[1] - grid2[0] == cells, "grid2 must be exactly one 8x8 grid"
        if grid3 is not None:
            assert grid3[1] - grid3[0] == cells, "grid3 must be exactly one 8x8 grid"

        # features_dim is the total output width handed to the pi/vf MLP heads.
        # The coarse branch (when present) is folded into `self.linear` below,
        # so this stays cnn_features regardless of grid3.
        super().__init__(observation_space, features_dim=cnn_features + num_scalars)

        self._grid1 = grid1
        self._grid2 = grid2
        self._grid3 = grid3
        self.register_buffer("_scalar_idx", torch.tensor(scalar_idx, dtype=torch.long))

        # 3x3 convs with padding keep the 8x8 resolution; the grid is small, so
        # spatial downsampling would throw away most of the signal.
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 2, GRID_H, GRID_W)
            cnn_flat = self.cnn(dummy).shape[1]

        if grid3 is not None:
            self.cnn_coarse = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy_coarse = torch.zeros(1, 1, GRID_H, GRID_W)
                coarse_flat = self.cnn_coarse(dummy_coarse).shape[1]
            self.linear = nn.Sequential(nn.Linear(cnn_flat + coarse_flat, cnn_features), nn.ReLU())
        else:
            self.cnn_coarse = None
            self.linear = nn.Sequential(nn.Linear(cnn_flat, cnn_features), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        g1 = observations[:, self._grid1[0]:self._grid1[1]]
        g2 = observations[:, self._grid2[0]:self._grid2[1]]
        # (B, 2, 8, 8): channel 0 = entity/terrain, channel 1 = grass health.
        img = torch.stack(
            [g1.view(-1, GRID_H, GRID_W), g2.view(-1, GRID_H, GRID_W)], dim=1
        )
        fine_out = self.cnn(img)
        if self.cnn_coarse is not None:
            g3 = observations[:, self._grid3[0]:self._grid3[1]]
            coarse_img = g3.view(-1, 1, GRID_H, GRID_W)
            coarse_out = self.cnn_coarse(coarse_img)
            cnn_out = self.linear(torch.cat([fine_out, coarse_out], dim=1))
        else:
            cnn_out = self.linear(fine_out)
        scalars = observations.index_select(1, self._scalar_idx)
        return torch.cat([cnn_out, scalars], dim=1)
