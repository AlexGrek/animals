"""Shared constants mirrored from the Rust engine (`animals_engine/src/lib.rs`).

Keep these in sync with `SNAKE_OBS_SIZE`, `PREY_OBS_SIZE`, `HUNGER_LIMIT`, and
`SMELL_RANGE` there — changing the observation size invalidates saved
checkpoints in `learner/models/` (SB3 load fails on shape mismatch).
"""

# 8x8 grid (64) + smelled-prey unit-direction (2) + normalized distance (1)
# + hunger (1) + own length (1).
SNAKE_OBS_SIZE = 69

# 8x8 grid (64) + nearest-snake-head unit-direction (2) + normalized distance (1).
# Shared by Prey and Amphibia (terrain values are species-relative).
PREY_OBS_SIZE = 67

# Steps without eating before a snake dies of hunger.
HUNGER_LIMIT = 1200

# A snake only smells prey within this torus-wrapped Manhattan distance.
SMELL_RANGE = 30

# Prey / amphibia discrete action space: 0 Stand, 1 Up, 2 Right, 3 Down, 4 Left.
PREY_NUM_ACTIONS = 5
# Snake discrete action space: 0 Straight, 1 Turn Right, 2 Turn Left.
SNAKE_NUM_ACTIONS = 3
