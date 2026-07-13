# GEMINI.md

This file provides guidance to Gemini CLI when working with code in this repository.

## Project

Multi-agent reinforcement learning (MARL) project: a Rust snake game engine trained via Python Stable-Baselines3 (PPO), connected through PyO3 FFI. See `docs/architecture.md` and `docs/learning.md` for detailed specs — keep them updated when changing the architecture, observation space, or rewards.

## Commands

All workflows go through `task` (Taskfile runner). The env var `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` is required for all cargo builds (Python 3.14 is newer than pyo3 0.22 supports); the Taskfile sets it.

```bash
task check                # cargo check the Rust workspace
task build-sim            # Build PyO3 module + install into learner venv (uv pip install -e)
task train STEPS=500000   # Train the PPO model (saves to learner/models/snake_model.zip)
task play                 # Run the Bevy game, manual keyboard control
task play-ai -- --snakes 4 --model v1 --model v2   # Watch trained models play
task test-ai -- --snakes 4 --model v1 --output results.json  # Headless full-speed eval, JSON stats
```

Training options beyond STEPS require running directly (from `learner/`, with `PYTHONPATH=src`):

```bash
uv run python -m learner.main --num-games 16 --snakes-per-game 2 --num-procs 4 \
    --existing v1:4 --existing v2:2
```

`--num-games` must be evenly divisible by `--num-procs`. `--existing path:count` fills snake slots with frozen past models (mixed-model self-play).

**After any Rust change to `animals_engine` or `animals_simulation`, run `task build-sim`** — otherwise Python keeps importing the stale compiled extension.

Toolchain: Rust 1.85+ (edition 2024), Python 3.14+, `uv` for Python deps (venv lives in `learner/.venv`). There is no meaningful test suite; verify Rust with `task check` and Python by running a short `task train STEPS=2048`.

## Architecture

Three Rust crates (cargo workspace) + one Python package:

- **`animals_engine`** — headless game logic. `GameState` holds N `SnakeState`s; `get_relative_observation(snake_index)` produces the per-snake observation. Snakes run independent per-snake episodes: on death a snake is respawned in place via `respawn_dead()` rather than ending the whole game, so one snake's mistake never truncates the others. No I/O, no globals — instances are fully independent.
- **`animals_simulation`** — PyO3 `cdylib` wrapping `GameState` as a Python `Simulation` class (`new(num_snakes)`, `reset() -> obs per snake`, `step(actions: list) -> (obs, rewards, dones, terminal_obs)`, `get_stats()`). `dones[i]` is true on the tick snake i died; `obs[i]` is that snake's post-respawn observation and `terminal_obs[i]` its pre-respawn (true terminal) observation on death ticks. The reward function lives here (`step`), not in the engine.
- **`animals_game`** — Bevy 2D visualizer. In `--ai` mode it picks a free ephemeral TCP port, spawns `learner.play` as a child process (killed via `Drop` on `AiServerProcess`), and exchanges raw little-endian bytes per tick containing the observations for snakes, preys, amphibias, and corpsefags out, and actions back. The engine no longer sets `game_over` itself, so `game_tick` sets it when it sees a dead snake to preserve the manual "freeze, press Space to restart" UX.
- **`learner`** (Python, in `learner/src/learner/`) — `environment.py` defines `RustMultiSnakeVecEnv`, a custom SB3 `VecEnv` that presents only the training snakes to PPO while internally computing actions for frozen "existing" opponent models ("shared brain" self-play trick — see `docs/learning.md`), and `MultiProcRustVecEnv`, a pipe-based multiprocess wrapper around it. `main.py` trains, `play.py` is the TCP inference server for Bevy, `test.py` runs headless evals.

### Cross-language invariants

The observation sizes (Snake: 197, Prey/Amphibia: 132, Corpsefag: 18) and the discrete spaces are hardcoded in four places that must stay in sync:

1. `animals_engine/src/game.rs` — `get_relative_observation`, `get_prey_observation`, etc.
2. `learner/src/learner/constants.py` — `SNAKE_OBS_SIZE`, `PREY_OBS_SIZE`, `CORPSEFAG_OBS_SIZE`
3. `learner/src/learner/play.py` — struct unpacking based on these sizes
4. `animals_game/src/ai.rs` — byte payload sizing in `game_tick`

Changing the observation also invalidates saved checkpoints in `learner/models/` (SB3 load will fail on shape mismatch) — retrain or delete them.

Grid size (400×400) is duplicated in `animals_simulation/src/lib.rs` (`GameState::new(400, 400, ...)`) and `animals_game/src/main.rs` (`GRID_WIDTH`/`GRID_HEIGHT` constants).
