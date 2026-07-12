"""Shared constants mirrored from the Rust engine (`animals_engine/src/lib.rs`).

Keep these in sync with `SNAKE_OBS_SIZE`, `PREY_OBS_SIZE`, `HUNGER_LIMIT`, and
`SMELL_RANGE` there — changing the observation size invalidates saved
checkpoints in `learner/models/` (SB3 load fails on shape mismatch).
"""

# 8x8 grid (64) + smelled-prey unit-direction (2) + normalized distance (1)
# + hunger (1) + own length (1) + grass health grid (64)
# + 8x8 coarse visitation-recency grid (64).
SNAKE_OBS_SIZE = 197

# 8x8 grid (64) + nearest-snake-head unit-direction (2) + normalized distance (1)
# + reproduction-progress scalar (1) + grass health grid (64).
# Shared by Prey and Amphibia (terrain values are species-relative).
PREY_OBS_SIZE = 132

# Steps without eating before a snake dies of hunger.
HUNGER_LIMIT = 1200

# A snake only smells prey within this torus-wrapped Manhattan distance.
SMELL_RANGE = 60

# 3x3 grid (9) + nearest-corpse unit-direction (2) + normalized distance (1) + points (1)
CORPSEFAG_OBS_SIZE = 18

# Prey / amphibia / corpsefag discrete action space: 0 Stand, 1 Up, 2 Right, 3 Down, 4 Left.
PREY_NUM_ACTIONS = 5
CORPSEFAG_NUM_ACTIONS = 5
# Snake discrete action space: 0 Straight, 1 Turn Right, 2 Turn Left.
SNAKE_NUM_ACTIONS = 3

# --- Observation grid layout (for the CNN feature extractor, `policy.py`) ---
# Each observation holds two co-located 8x8 grids fed as a 2-channel image plus
# a handful of scalar features between/after them. These slices mirror the write
# order in `animals_engine/src/game.rs` (get_relative_observation / get_prey_observation)
# and MUST stay in sync with the Rust obs builders.
GRID_H = 8
GRID_W = 8
GRID_CELLS = GRID_H * GRID_W  # 64

# Snake: terrain/entity grid [0:64), scalars [64:69) (smell dir/dist, hunger,
# length), grass-health grid [69:133), coarse visitation-recency grid [133:197).
# Pinned explicitly (not derived from SNAKE_OBS_SIZE) since the visitation
# grid was appended after grass-health, so "last GRID_CELLS floats" no longer
# identifies a single slice.
SNAKE_GRID1 = (0, GRID_CELLS)
SNAKE_GRID2 = (69, 133)
SNAKE_GRID3 = (133, 197)

# Prey/Amphibia: terrain grid [0:64), scalars [64:68) (snake dir/dist,
# reproduction progress), grass-health grid [68:132).
PREY_GRID1 = (0, GRID_CELLS)
PREY_GRID2 = (PREY_OBS_SIZE - GRID_CELLS, PREY_OBS_SIZE)  # (68, 132)
