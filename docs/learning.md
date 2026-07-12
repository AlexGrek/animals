# Reinforcement Learning Details

The RL system trains three independent policies in a predator/prey loop: **Snake** (predator, self-play across snake slots), **Prey** (land-favoring herbivore), and **Amphibia** (water-favoring herbivore, same observation layout as Prey but a different terrain-speed profile — see `Species::speed_on` in `animals_engine/src/species.rs`). Each is trained against a *frozen* snapshot of its counterpart(s): the snake env loads the current `prey_model`/`amphibia_model` as opponents, and the prey/amphibia envs load `snake_model` as the predator. When a counterpart checkpoint is missing or its observation shape doesn't match (see `learner/src/learner/model_utils.py`), it falls back to action `0` (stand still / go straight) rather than failing, so the pipeline can bootstrap from nothing.

## Observation Space

### Snake (`SNAKE_OBS_SIZE = 197` floats, `animals_engine/src/game.rs::get_relative_observation`)
- `[0..64)` — 8x8 grid in the snake's own rotated frame (4 cells ahead, 3 behind, 4 right, 3 left of the head):
  - **1.0**: Prey (either species)
  - **-1.0**: Wall / rock / own body
  - **-0.8**: Enemy snake head (the part that kills on collision or head-to-head)
  - **-0.5**: Enemy snake body
  - otherwise **`Species::Snake.speed_on(terrain) * 0.5`** (passable terrain, weighted by how fast a snake moves there)
- `[64]`/`[65]` — **unit** direction to the nearest prey the snake can *smell* (forward/right components in the snake's frame); zero if nothing is smelled. A snake only smells prey within `SMELL_RANGE = 60` torus-wrapped Manhattan cells of its head (`GameState::update_targets`) — it has no knowledge of prey farther away, however close it may appear on an absolute map view. A unit vector keeps the heading signal equally strong at any range within that radius, unlike the old `dx / max_dim` encoding which shrank to ~0.01 for a prey a few cells away.
- `[66]` — distance to that prey, normalized by `SMELL_RANGE` (`1.0` if nothing is smelled).
- `[67]` — hunger: `steps_since_last_eat / HUNGER_LIMIT` (see below).
- `[68]` — own length `/ 100`, capped at 1.
- `[69..133)` — 8x8 grass-health grid over the *same* rotated cells as `[0..64)`: `grass_health` in `[0, 1]` per cell (1.0 = full grass, 0.0 = grazed bare / non-grass). Lets the snake read where prey have recently fed and head toward likely prey.
- `[133..197)` — 8x8 **coarse visitation-recency grid**, same rotated frame/cell order as `[0..64)` but each cell spans an 8x8-tile block (a 2x2 group of the 4x4-tile coarse patches the engine tracks per snake in `SnakeState::visited`). Value is `1.0` for a patch entered this tick, decaying linearly to `0.0` over `VISIT_HORIZON = 1500` ticks (or if never visited) — the freshest (max) recency among the patches the cell covers. This exists so the exploration reward below (which depends on visitation history) is actually observable by a memoryless feedforward policy — previously the reward signal referenced state the network couldn't see, so the best it could learn was a biased random walk that circled under `deterministic=True` inference. See `GridCnnExtractor` below for how this is fed to the network.

### Prey / Amphibia (`PREY_OBS_SIZE = 131` floats, `animals_engine/src/game.rs::get_prey_observation`, shared by both species)
- `[0..64)` — 8x8 grid in the absolute frame (up is always north):
  - **-1.0**: Out of bounds / rock
  - **-0.8**: Snake head (the lethal part — snakes eat in a 3×3 radius around their head)
  - **-0.5**: Snake body
  - otherwise **`prey.species.speed_on(terrain) * 0.5`** — this is species-relative, so the *same* map cell reads differently to the two species (water ≈ 0.1 to Prey, ≈ 0.5 to Amphibia; grass ≈ 0.4 to Prey, ≈ 0.3 to Amphibia)
- `[64]`/`[65]` — unit direction (east/north) to the nearest **alive** snake head; zero if no snake is alive.
- `[66]` — distance to that head, normalized as `min(dist / 60, 1)` (full resolution over the 0-60 cell band where escape decisions actually happen) instead of dividing by the larger grid dimension, which made 10-30 cell threats nearly invisible at roughly 0.02-0.07 signal magnitude; still `1.0` if no snake is alive. This is a semantic change: older prey/amphibia checkpoints still shape-match (obs size unchanged) but will misread this scalar until retrained.
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
  - **Exploration bonus**: `+0.1` for entering a coarse (4×4-tile) grid cell not visited within the last `VISIT_HORIZON` ticks, `-0.05` otherwise, applied **only when nothing is currently smelled** (smell → pursue via shaping; no smell → explore). Driven by `SnakeState::entered_new_patch`, computed once per tick in `GameState::step` (`animals_engine/src/game.rs`) from a per-snake `visited: HashMap<(i32,i32), u64>` (coarse cell → last-visit tick) — moved into the engine (it used to live in the PyO3 `Simulation` struct as a `Vec<HashSet>`) specifically so the same visitation history can also be exposed to the network as the `[133..197)` observation channel above; a reward term the policy can't see is unlearnable. Recency naturally decays via `VISIT_HORIZON` rather than being hard-cleared on death/eat — death already resets it (a fresh `SnakeState`), and a decaying signal is more informative than an eat-triggered wipe.
  - **Turn cost**: `-0.002` per tick the action isn't "straight" (actions 1/2). Tiny relative to the explore bonus; exists only to break a memoryless policy's indifference between turning and going straight when neither is otherwise reinforced, without meaningfully deterring pursuit turns (whose shaping/kill rewards dwarf it).

### Prey / Amphibia
- **Death** (eaten this tick): `-10.0`.
- **Reproduction** (`grass_eaten >= 8` with no snake within 8 cells; the parent slot "dies" and 3 children spawn): `+25.0` terminal reward — previously this was wrongly punished as a death (`-10`), which trained prey to avoid grazing enough to trigger it.
- Long-dead pool slots (dead but not this tick, awaiting revival): reward exactly `0.0`, so they no longer pollute the PPO batch with fake `+0.1` survival signal.
- Alive: base `0.1`; grazing `+0.5 * grass_eaten` delta this tick (~`+0.25`/tick on full grass; dense "seek fresh grass" signal that also drives exploration since grazed cells deplete); stand penalty `-0.2` only when standing AND not currently grazing (grass delta `<= 0`); threat shaping `0.1 * clamp(delta-distance to nearest snake head, ±2)` with the distance now **torus-wrapped** (previously unwrapped, which was wrong near map edges); new danger-zone penalty `-0.15` per tick while within 10 cells of a snake head; existing crowding penalty/shaping unchanged.
- The per-sibling `+2.0` death bonus previously applied in the Python envs (`prey_environment.py` / `amphibia_environment.py`) has been removed (an uncontrollable event = pure variance, and it would have double-rewarded reproduction events).

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

The snake observation carries a **third grid** (`SNAKE_GRID3`, `[133..197)`): the coarse
visitation-recency grid described above. It's at a different spatial scale than grid1/grid2 (8
tiles/cell vs 1 tile/cell), so stacking it as a third channel of the same image would spatially
misalign it — instead `GridCnnExtractor` accepts an optional `grid3` kwarg and runs it through
its own small conv branch (1→8→16 channels), concatenating its flattened output with the
fine-grid branch's before the final linear projection to 128 features. Prey/amphibia models
don't pass `grid3` (their observation is unchanged) and get the original two-grid path.

On top of that extractor:
- **Snake**: MLP `pi=[256, 256]`, `vf=[256, 256]`.
- **Prey / Amphibia**: MLP `pi=[128, 128]`, `vf=[128, 128]` — simpler action space (5 discrete
  moves vs 3 turns), so a smaller head is sufficient and faster to train.
- Framework: PyTorch via Stable-Baselines3, Algorithm: PPO.

## PPO Hyperparameters & CPU Throughput

Training runs on `device="cpu"` (the policy MLPs are small enough that GPU host↔device transfer/launch overhead exceeds the compute it would save). On CPU, PPO's optimizer step count dominates wall-clock far more than environment rollout speed: with SB3's defaults (`batch_size=64`, `n_steps=2048`) and 16 parallel training envs, each policy update does `(2048*16/64) = 512` minibatches × 10 epochs = 5,120 tiny optimizer steps, versus rollout collection alone running at ~60,000 steps/s.

We instead use:
- Snake: `batch_size=4096`, `n_steps=512`, `ent_coef=0.01` (measured ~14,000 steps/s, a ~4.5x wall-clock speedup over SB3 defaults).
- Prey / Amphibia: `batch_size=2048`, `ent_coef=0.02` — lower than the snake's exploration needs less encouragement now that the reward includes dense threat-distance shaping rather than only sparse survive/death. Prey now uses `n_steps=128` paired with a small-pool env config (~160 envs total: 16 games x 10 max preys) for roughly 24 PPO updates per 500k training steps; amphibia is unchanged at `n_steps=512` until its own retrain.

Changing any observation size invalidates saved checkpoints in `learner/models/` (SB3 `.load()` fails on shape mismatch) — retrain or delete them. See `CLAUDE.md` for the full list of files that must stay in sync.

## Train/Play Parity

Training rolls out actions stochastically (PPO sampling; frozen opponents via
`model_utils.predict_actions`, which also samples by default). `play.py` (the Bevy inference
server) and `test.py` (headless eval) default to the same stochastic sampling now — pass
`--deterministic` to either to switch to argmax instead. This matters because a 3-action turn
policy with even a mild logit bias toward one turn looks fine under sampling (noise breaks the
bias) but locks into a perfect circle under argmax; watching/evaluating with the training-time
sampling distribution avoids exposing that artifact as if it were a real behavior bug.

Two more places the Bevy game previously diverged from the training simulation, both fixed:
- **Anti-suicide steering** (`GameState.auto_steer`, gates the force-turn-away-from-obstacles
  logic in `game.rs::step`): now off whenever a model drives (training and AI-mode play alike),
  on only for manual keyboard play. It used to be tied to `is_training` instead, so a policy
  would encounter a steering override in the game it never experienced in training — visually
  indistinguishable from the policy itself circling around an obstacle.
- **Corpse persistence**: the AI branch of `animals_game`'s `game_tick` now calls
  `engine.0.respawn_dead()` (matching `animals_simulation`'s per-tick respawn), instead of
  leaving dead snakes as permanent `-0.5` obstacles the policy never trained against.

`test.py` also reports circling-diagnostic stats per snake (`action_distribution`, `turn_bias`,
`longest_turn_run`, `unique_patches_per_life`, `unique_patches_per_100_ticks`,
`mean_displacement_per_100_ticks`) — run it with both `--deterministic` and without, and at low
vs. high prey counts, to check whether a given fix actually reduced circling rather than just
looking better anecdotally in the Bevy viewer.
