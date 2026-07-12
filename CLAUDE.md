# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Multi-agent reinforcement learning (MARL) project: a Rust snake game engine trained via Python Stable-Baselines3 (PPO), connected through PyO3 FFI. See `docs/architecture.md` and `docs/learning.md` for detailed specs — keep them updated when changing the architecture, observation space, or rewards.

## Commands

All workflows go through `task` (Taskfile runner). The env var `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` is required for all cargo builds (Python 3.14 is newer than pyo3 0.22 supports); the Taskfile sets it.

```bash
task check                     # cargo check the Rust workspace
task build-sim                 # Build PyO3 module + install into learner venv (uv pip install -e)
task train STEPS=500000        # Train the snake PPO model (saves to learner/models/snake_model.zip)
task train-prey STEPS=1000000  # Train the prey PPO model against a frozen snake model
task train-amphibia STEPS=1000000  # Train the amphibia PPO model against a frozen snake model
task play                      # Run the Bevy game, manual keyboard control
task play-ai -- --snakes 4 --model v1 --model v2   # Watch trained models play
task test-ai -- --snakes 4 --model v1 --output results.json  # Headless full-speed eval, JSON stats
```

Training options beyond STEPS require running directly (from `learner/`, with `PYTHONPATH=src`):

```bash
uv run python -m learner.main --num-games 16 --snakes-per-game 2 --num-procs 4 \
    --existing v1:4 --existing v2:2 \
    --preys-per-game 2 --amphibias-per-game 1 \
    --prey-model prey_model --amphibia-model amphibia_model
uv run python -m learner.train_prey --num-games 16 --preys-per-game 1 --snake-model snake_model
uv run python -m learner.train_amphibia --num-games 16 --amphibias-per-game 1 --snake-model snake_model
```

`--num-games` must be evenly divisible by `--num-procs`. `--existing path:count` fills snake slots with frozen past models (mixed-model self-play). Any frozen counterpart model (prey/amphibia for snake training, snake for prey/amphibia training) that is missing or has a stale observation shape falls back to a static "do nothing" action instead of erroring (`learner/src/learner/model_utils.py`) — this is what lets the pipeline bootstrap from zero trained models: train snake vs. static prey first, then prey/amphibia vs. that snake, then snake again vs. the newly trained prey/amphibia, and so on.

**After any Rust change to `animals_engine` or `animals_simulation`, run `task build-sim`** — otherwise Python keeps importing the stale compiled extension.

Toolchain: Rust 1.85+ (edition 2024), Python 3.14+, `uv` for Python deps (venv lives in `learner/.venv`). There is no meaningful test suite; verify Rust with `task check` and Python by running a short `task train STEPS=2048`.

## Architecture

Three Rust crates (cargo workspace) + one Python package. Three actor types: **Snake** (predator, self-play), **Prey**, **Amphibia** (the latter two share one observation layout and reward function but differ in terrain speed — `Species::speed_on`).

- **`animals_engine`** — headless game logic. `GameState` holds N `SnakeState`s and M `PreyState`s (each prey tagged with a `Species`). `get_relative_observation(snake_index)` produces the per-snake observation (`SNAKE_OBS_SIZE` = 197 floats); `get_prey_observation(prey_index)` produces the shared prey/amphibia observation (`PREY_OBS_SIZE` = 131 floats). Snakes run independent per-snake episodes: in training, on death a snake is respawned in place via `respawn_dead()` (fixed count, so SB3 sees continuous episodes); the visualizer instead calls `remove_dead_snakes()`, which reaps the dead snake's entity but leaves its body in `GameState.corpses` (a `HashSet` of obstacle cells) for a self-balancing ecosystem. Prey/amphibia respawn via a separate `respawn_dead_preys()` call — **not** automatic inside `step()`, so callers can read a dead prey's true terminal observation before it teleports to its next spawn. `GameState.auto_steer` gates an anti-suicide steering assist meant only for manual keyboard play (off for training and AI-driven play — see `docs/learning.md` "Train/Play Parity"). No I/O, no globals — instances are fully independent.
- **`animals_simulation`** — PyO3 `cdylib` wrapping `GameState` as a Python `Simulation` class (`new(num_snakes, num_preys, num_amphibias)`, `reset() -> obs per snake`, `get_all_prey_observations()`, `get_all_amphibia_observations()`, `step(actions, prey_actions, amphibia_actions) -> 12-tuple`, `get_stats()`). The 12-tuple is `(obs, rewards, dones, terminal_obs, prey_obs, prey_rewards, prey_dones, amphibia_obs, amphibia_rewards, amphibia_dones, prey_terminal_obs, amphibia_terminal_obs)`. All `*_dones[i]` are true on the tick that actor died; `*_obs` is always the post-respawn (next-episode) observation, `*_terminal_obs` is the true pre-respawn observation (meaningful only where `dones` is true). The reward function lives here (`step`), not in the engine.
- **`animals_game`** — Bevy 2D visualizer. In `--ai` mode it picks a free ephemeral TCP port, spawns `learner.play` as a child process (killed via `Drop` on `AiServerProcess`), and exchanges raw little-endian bytes per tick: `num_snakes * SNAKE_OBS_SIZE + total_preys * PREY_OBS_SIZE` f32 observations out, `num_snakes + total_preys` i32 actions back. It calls `engine.step()` then `engine.respawn_dead_preys()` every tick (it doesn't need terminal observations, so it respawns prey immediately). **Both game modes run a self-balancing ecosystem** (not fixed-count self-play): each tick calls `engine.remove_dead_snakes()`, which removes a dead snake's *entity* from `snakes` (uncounted, undriven, bounded memory → cycles indefinitely) but leaves its body cells in `GameState.corpses` as a static `-0.5` obstacle that living snakes see and die on. Snake population is thus governed by births (mitosis splits a snake at body length ≥ 12 into 3 in non-training mode) vs. deaths (hunger/collision/corpse). `num_snakes` varies per tick; the TCP protocol and the on-screen counter (`snakes.len()`) re-read it each tick, so the counter shows alive snakes only. `game_tick` sets `game_over` (freeze, press Space to restart) only when snakes hit 0 — total predator extinction in AI mode, or the player's death in manual mode.
- **`learner`** (Python, in `learner/src/learner/`) — `constants.py` mirrors the Rust size constants; `model_utils.py` has the shared `load_opponent`/`predict_actions` helpers (static-action fallback + batched inference) used by all three envs below. `environment.py` defines `RustMultiSnakeVecEnv` (trains snake, drives frozen prey+amphibia opponents) and `MultiProcRustVecEnv` (pipe-based multiprocess wrapper). `prey_environment.py`/`amphibia_environment.py` define `RustPreyVecEnv`/`RustAmphibiaVecEnv` (train prey/amphibia against a frozen snake). `main.py`/`train_prey.py`/`train_amphibia.py` are the three training entrypoints, `play.py` is the TCP inference server for Bevy, `test.py` runs headless evals.

### Cross-language invariants

The **snake observation size** (`SNAKE_OBS_SIZE = 197`: 8×8 relative grid (64) + smelled-prey unit-direction (2) + distance/SMELL_RANGE (1) + hunger (1) + own length (1) + grass-health grid (64) + 8×8 coarse visitation-recency grid (64)), the **prey/amphibia observation size** (`PREY_OBS_SIZE = 131`: 8×8 absolute grid (64) + nearest-snake-head unit-direction (2) + normalized distance (1) + grass-health grid (64)), the **snake action space** (3: straight/right/left), and the **prey/amphibia action space** (5: stand/up/right/down/left) are hardcoded in these places that must stay in sync:

1. `animals_engine/src/lib.rs` — `SNAKE_OBS_SIZE`, `PREY_OBS_SIZE`, `HUNGER_LIMIT`, `HUNGER_DEATH_LIMIT`, `SMELL_RANGE`, `VISIT_HORIZON` constants (used by `game.rs`'s two observation functions, `update_targets`, and the visitation-tracking pass in `step`). `HUNGER_LIMIT` normalizes the hunger observation/reward only; `HUNGER_DEATH_LIMIT` (lower, currently 500 vs. 1200) is the actual starvation threshold — kept separate so starvation speed can be tuned without invalidating a trained model's hunger calibration (see `docs/learning.md` "Hunger and Eating").
2. `learner/src/learner/constants.py` — Python mirror of the same constants, imported by every env/script below; also pins `SNAKE_GRID1/2/3` (the snake obs's three 8×8 grid slices — `SNAKE_GRID3` is the visitation channel, fed to `GridCnnExtractor`'s optional coarse conv branch)
3. `learner/src/learner/environment.py` — `spaces.Box(shape=(SNAKE_OBS_SIZE,))` (both VecEnv classes)
4. `learner/src/learner/prey_environment.py` / `amphibia_environment.py` — `spaces.Box(shape=(PREY_OBS_SIZE,))`
5. `learner/src/learner/play.py` — struct packing byte sizes for the Bevy TCP protocol
6. `animals_game/src/main.rs` — byte payload sizing in `gather_observations`/`spawn_ai_worker` (imports the same Rust constants, so this one syncs automatically)

Changing either observation size also invalidates saved checkpoints in `learner/models/` (SB3 load will fail on shape mismatch) — retrain or delete them. A frozen counterpart model with a stale shape is treated as "missing" and falls back to a static action rather than crashing (`model_utils.load_opponent`).

Grid size (400×400) is duplicated in `animals_simulation/src/lib.rs` (`GameState::new(400, 400, ...)`) and `animals_game/src/main.rs` (`GRID_WIDTH`/`GRID_HEIGHT` constants).
