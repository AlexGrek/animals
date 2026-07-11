# Reinforcement Learning Details

The RL system trains three independent policies in a predator/prey loop: **Snake** (predator, self-play across snake slots), **Prey** (land-favoring herbivore), and **Amphibia** (water-favoring herbivore, same observation layout as Prey but a different terrain-speed profile — see `Species::speed_on` in `animals_engine/src/species.rs`). Each is trained against a *frozen* snapshot of its counterpart(s): the snake env loads the current `prey_model`/`amphibia_model` as opponents, and the prey/amphibia envs load `snake_model` as the predator. When a counterpart checkpoint is missing or its observation shape doesn't match (see `learner/src/learner/model_utils.py`), it falls back to action `0` (stand still / go straight) rather than failing, so the pipeline can bootstrap from nothing.

## Observation Space

### Snake (`SNAKE_OBS_SIZE = 133` floats, `animals_engine/src/game.rs::get_relative_observation`)
- `[0..64)` — 8x8 grid in the snake's own rotated frame (4 cells ahead, 3 behind, 4 right, 3 left of the head):
  - **1.0**: Prey (either species)
  - **-1.0**: Wall / rock / own body
  - **-0.8**: Enemy snake head (the part that kills on collision or head-to-head)
  - **-0.5**: Enemy snake body
  - otherwise **`Species::Snake.speed_on(terrain) * 0.5`** (passable terrain, weighted by how fast a snake moves there)
- `[64]`/`[65]` — **unit** direction to the nearest prey the snake can *smell* (forward/right components in the snake's frame); zero if nothing is smelled. A snake only smells prey within `SMELL_RANGE = 30` torus-wrapped Manhattan cells of its head (`GameState::update_targets`) — it has no knowledge of prey farther away, however close it may appear on an absolute map view. A unit vector keeps the heading signal equally strong at any range within that radius, unlike the old `dx / max_dim` encoding which shrank to ~0.01 for a prey a few cells away.
- `[66]` — distance to that prey, normalized by `SMELL_RANGE` (`1.0` if nothing is smelled).
- `[67]` — hunger: `steps_since_last_eat / HUNGER_LIMIT` (see below).
- `[68]` — own length `/ 100`, capped at 1.
- `[69..133)` — 8x8 grass-health grid over the *same* rotated cells as `[0..64)`: `grass_health` in `[0, 1]` per cell (1.0 = full grass, 0.0 = grazed bare / non-grass). Lets the snake read where prey have recently fed and head toward likely prey.

### Prey / Amphibia (`PREY_OBS_SIZE = 131` floats, `animals_engine/src/game.rs::get_prey_observation`, shared by both species)
- `[0..64)` — 8x8 grid in the absolute frame (up is always north):
  - **-1.0**: Out of bounds / rock
  - **-0.8**: Snake head (the lethal part — snakes eat in a 3×3 radius around their head)
  - **-0.5**: Snake body
  - otherwise **`prey.species.speed_on(terrain) * 0.5`** — this is species-relative, so the *same* map cell reads differently to the two species (water ≈ 0.1 to Prey, ≈ 0.5 to Amphibia; grass ≈ 0.4 to Prey, ≈ 0.3 to Amphibia)
- `[64]`/`[65]` — unit direction (east/north) to the nearest **alive** snake head; zero if no snake is alive.
- `[66]` — distance to that head, normalized by the larger grid dimension (`1.0` if no snake is alive).
- `[67..131)` — 8x8 grass-health grid over the *same* absolute cells as `[0..64)`: `grass_health` in `[0, 1]` per cell — where the food is, so prey can graze toward full grass (which also drives reproduction once `grass_eaten ≥ 8`).

This global threat vector exists because a prey's local 8x8 patch (roughly a 7-8 cell radius) is often too small to see an oncoming snake in time. Note this is asymmetric with the snake's sense of prey: prey always see the globally nearest snake head, while a snake only *smells* prey within `SMELL_RANGE` (see above) — deliberate, so a snake must explore to find prey rather than beelining toward one anywhere on the map.

## Reward Functions (`animals_simulation/src/lib.rs::step`)

### Snake
- **Death**: `-5.0` if by hunger, else `-3.0` (wall, self, opponent, or head-to-head collision).
- **Kill** (opponent collides into you): `+50.0 * Δkills` — additive with eating, not `else if`, so a same-tick kill-and-eat is fully credited.
- **Eat** (prey within a 3×3 radius of the head): `+30.0 * Δscore`.
- **Mitosis** (body reached the split threshold this tick — see below): `+60.0 * mitosis_count`, added on top of any death/eat/kill/shaping. Kept as the single largest reward (the reproduction goal) but only just above a kill, since each mitosis already rides on ~6-8 eats worth of `+30`; a larger spike mostly inflates value-function variance.
- Otherwise (no kill/eat this tick):
  - **Smell shaping**: `0.15 * clamp(prev_dist_to_smelled_prey - curr_dist, -2.0, 2.0)`, gated to prey within `SMELL_RANGE` torus-wrapped Manhattan cells (`min_dist_to_smelled_prey` in `animals_simulation/src/lib.rs`). If either side of the delta has nothing in smell range (prey just entered/left range, or none exists), no shaping is applied that tick — the reward never leaks information the policy can't observe. The distance itself is torus-wrapped, unlike the pre-existing (buggy) unwrapped version.
  - **Hunger penalty**: `-0.01 * steps_since_last_eat / (HUNGER_LIMIT / 4)`.
  - **Exploration bonus**: `+0.05` for entering a not-yet-visited 4×4-coarse grid cell, applied **only when nothing is currently smelled** (smell → pursue via shaping; no smell → explore). Visited-cell state is per-snake, held in the PyO3 `Simulation` struct (not the engine), and is cleared whenever the snake dies or eats — it only tracks "new ground since the last meal, this life."

### Prey / Amphibia
- **Death** (eaten this tick): `-10.0`.
- **Survive**: `0.1` base, plus threat shaping `0.1 * clamp(curr_dist_to_nearest_alive_snake_head - prev_dist, -2.0, 2.0)` — reward for increasing distance from the closest predator. For Amphibia this naturally rewards retreating into water, where snakes crawl at 0.2 speed but amphibia swim at 1.0.
- When training with multiple prey/amphibia per game, a surviving individual also gets `+2.0` for each sibling eaten that tick (the predator is occupied elsewhere — a genuinely safer state), applied in the Python env (`prey_environment.py` / `amphibia_environment.py`), not the Rust reward.

## Hunger and Eating

- `HUNGER_LIMIT = 1200` steps without eating kills a snake (`animals_engine/src/lib.rs`).
- Snakes eat any prey within a 3×3 radius of their head (Chebyshev distance ≤ 1), not just an exact cell match — this makes eating slightly forgiving of the 1-cell-per-tick grid movement.

## Episode Termination: Per-Snake and Per-Prey Respawn

Snakes do **not** share a single game-over condition. `GameState::step()` never sets `game_over` on death; when a snake dies it is immediately respawned by `GameState::respawn_dead()` (fresh body of length 1 at a spawn position, score/kills/death flags reset).

Prey and amphibia respawn the same way but through a separate, explicit call: `GameState::respawn_dead_preys()`. It is **not** called automatically inside `step()` — the training simulation (`animals_simulation/src/lib.rs`) calls `get_prey_observation` for every prey that died *before* calling `respawn_dead_preys()`, so it can capture the true pre-respawn terminal observation; only after that does it respawn and compute the fresh post-respawn observation. Earlier, prey respawned inside `step()` itself, so every consumer (Python envs and `test.py`) was reporting the post-respawn (fresh-spawn) observation as if it were the terminal one — corrupting the PPO value function's bootstrap on death. The Bevy visualizer, which doesn't need terminal observations, just calls `respawn_dead_preys()` immediately after `step()` each tick.

The PyO3 `Simulation.step()` returns a 12-tuple:
```
(obs, rewards, dones, terminal_obs,
 prey_obs, prey_rewards, prey_dones,
 amphibia_obs, amphibia_rewards, amphibia_dones,
 prey_terminal_obs, amphibia_terminal_obs)
```
`dones[i]` / `prey_dones[i]` / `amphibia_dones[i]` are true exactly on the tick that actor died. The `*_obs` arrays are always post-respawn (next-episode) observations; the `*_terminal_obs` arrays are the true pre-respawn observations, meaningful only where the matching `dones` entry is true.

Head-to-head collisions (two snakes' heads landing on the same cell in the same tick) kill **both** snakes — computed from a pre-step snapshot of alive snakes and their next head positions.

The Bevy visualizer (`animals_game`) still wants a classic "game over, press Space to restart" experience for manual/AI-watch play: it detects any snake death itself after calling `engine.step()` and sets `GameState.game_over`.

## The Vector Environment Trick & Mixed-Model / Mixed-Species Training

Stable-Baselines3 natively only supports single-agent environments. To enable MARL without migrating to heavy libraries like PettingZoo, we built three custom `VecEnv`s, one per trained policy:

- **`RustMultiSnakeVecEnv`** (`environment.py`) — trains the snake policy. Spawns multiprocessing workers, each managing multiple PyO3 `Simulation` instances (`preys_per_game` land prey + `amphibias_per_game` amphibia per instance, both driven by frozen opponent models). Randomly assigns snake slots to either the model actively being trained or one or more frozen "existing" past snake checkpoints (self-play across generations), exposing only the training slots to SB3.
- **`RustPreyVecEnv`** (`prey_environment.py`) and **`RustAmphibiaVecEnv`** (`amphibia_environment.py`) — mirror structure, training the land/water herbivore policy against a frozen snake model. Both use the true pre-respawn `prey_terminal_obs`/`amphibia_terminal_obs` from `Simulation.step()` for SB3's `infos["terminal_observation"]`.

All three batch their counterpart's action prediction: they gather every game's observations for that counterpart into one array and call `model.predict()` once per step instead of once per agent (`learner/src/learner/model_utils.py::predict_actions`), which matters because with 16+ games per process each step would otherwise trigger dozens of single-row PyTorch forward passes.

## Neural Network Architecture

Every observation carries **two co-located 8×8 grids** — an entity/terrain grid and a
grass-health grid (the latter lets a snake infer where prey have been feeding, since grazed
cells read as depleted). Rather than flatten them into the MLP (which discards their spatial
structure), all three policies use a shared custom feature extractor,
`GridCnnExtractor` (`learner/src/learner/policy.py`): it reshapes the two grids into a
2-channel 8×8 image, runs two padded 3×3 convs (2→16→32 channels) + a linear projection to 128
features, and concatenates the raw scalar features (smell/threat direction+distance, hunger,
length). The grid/scalar index slices live in `learner/src/learner/constants.py`
(`SNAKE_GRID1/2`, `PREY_GRID1/2`) and mirror the write order in `animals_engine/src/game.rs`.

On top of that extractor:
- **Snake**: MLP `pi=[256, 256]`, `vf=[256, 256]`.
- **Prey / Amphibia**: MLP `pi=[128, 128]`, `vf=[128, 128]` — simpler action space (5 discrete
  moves vs 3 turns), so a smaller head is sufficient and faster to train.
- Framework: PyTorch via Stable-Baselines3, Algorithm: PPO.

## PPO Hyperparameters & CPU Throughput

Training runs on `device="cpu"` (the policy MLPs are small enough that GPU host↔device transfer/launch overhead exceeds the compute it would save). On CPU, PPO's optimizer step count dominates wall-clock far more than environment rollout speed: with SB3's defaults (`batch_size=64`, `n_steps=2048`) and 16 parallel training envs, each policy update does `(2048*16/64) = 512` minibatches × 10 epochs = 5,120 tiny optimizer steps, versus rollout collection alone running at ~60,000 steps/s.

We instead use:
- Snake: `batch_size=4096`, `n_steps=512`, `ent_coef=0.01` (measured ~14,000 steps/s, a ~4.5x wall-clock speedup over SB3 defaults).
- Prey / Amphibia: `batch_size=2048`, `n_steps=512`, `ent_coef=0.02` — lower than the snake's exploration needs less encouragement now that the reward includes dense threat-distance shaping rather than only sparse survive/death.

Changing any observation size invalidates saved checkpoints in `learner/models/` (SB3 `.load()` fails on shape mismatch) — retrain or delete them. See `CLAUDE.md` for the full list of files that must stay in sync.
