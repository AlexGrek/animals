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
        cnn_features: int = 128,
    ):
        obs_dim = int(observation_space.shape[0])

        # Indices covered by the two spatial grids; everything else is a scalar.
        grid_idx = set(range(grid1[0], grid1[1])) | set(range(grid2[0], grid2[1]))
        scalar_idx = [i for i in range(obs_dim) if i not in grid_idx]
        num_scalars = len(scalar_idx)

        cells = GRID_H * GRID_W
        assert grid1[1] - grid1[0] == cells, "grid1 must be exactly one 8x8 grid"
        assert grid2[1] - grid2[0] == cells, "grid2 must be exactly one 8x8 grid"

        # features_dim is the total output width handed to the pi/vf MLP heads.
        super().__init__(observation_space, features_dim=cnn_features + num_scalars)

        self._grid1 = grid1
        self._grid2 = grid2
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

        self.linear = nn.Sequential(nn.Linear(cnn_flat, cnn_features), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        g1 = observations[:, self._grid1[0]:self._grid1[1]]
        g2 = observations[:, self._grid2[0]:self._grid2[1]]
        # (B, 2, 8, 8): channel 0 = entity/terrain, channel 1 = grass health.
        img = torch.stack(
            [g1.view(-1, GRID_H, GRID_W), g2.view(-1, GRID_H, GRID_W)], dim=1
        )
        cnn_out = self.linear(self.cnn(img))
        scalars = observations.index_select(1, self._scalar_idx)
        return torch.cat([cnn_out, scalars], dim=1)
