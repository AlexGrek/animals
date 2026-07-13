# Architecture Overview

This workspace is a multi-language, multi-agent reinforcement learning (MARL) project that combines a Rust-based game engine with a Python-based ML training pipeline. It utilizes an elegant **Shared Brain** architecture to enable seamless self-play using Stable-Baselines3.

## Core Components

### 1. Game Engine (`animals_engine`)
A lightweight, headless Rust library containing the core logic, physics, and state representations for the Snake game. It handles all collision math, movement, and reward tracking for multiple snakes and preys. There is no global "game over": snakes die independently (wall, self, opponent, head-to-head collision — the latter always kills both snakes involved — or running into a corpse). Training respawns each dead snake in place via `GameState::respawn_dead()` (fixed count, continuous SB3 episodes), while the visualizer calls `GameState::remove_dead_snakes()`, which reaps the dead snake's entity but moves its body cells into `GameState.corpses` — a `HashSet` of static obstacle cells that living snakes read as `-0.5` and die on — giving a self-balancing ecosystem (births via mitosis vs. deaths). Prey/amphibia respawn similarly via `GameState::respawn_dead_preys()`, called explicitly by callers (not automatically inside `step()`) so the training simulation can read a dead prey's true terminal observation first. See `docs/learning.md` for how this maps to per-actor `dones` in the training environment.

`GameState.auto_steer` gates an anti-suicide steering assist (force-turn away from a heading that would immediately kill the snake) that only makes sense for a human player — it's off for both training and AI-driven play, since a policy that never experienced the override during training would otherwise get fought by it at play time. `GameState.steps` and each `SnakeState.visited`/`entered_new_patch` track per-snake coarse-cell visitation recency, feeding both the exploration reward (`animals_simulation`) and an observation channel (`get_relative_observation`'s `[133..197)`) — see `docs/learning.md`.

Preys come in two species (`animals_engine/src/species.rs::Species`): `Prey` (fast on land, slow in water) and `Amphibia` (fast in water, slower on land) — `Species::speed_on(terrain)` is the only place their movement differs; they share one `PreyState` struct and one observation function.
Additionally, the ecosystem includes the `Corpsefag`, a scavenger species that primarily feeds on static snake corpses rather than live actors.

### 2. PyO3 Simulation Binding (`animals_simulation`)
A Rust library compiled to a Python C-extension via Maturin. This exposes the `animals_engine` logic directly into the Python memory space, circumventing the overhead of Inter-Process Communication (IPC). The Python `step()` calls directly execute the highly optimized compiled Rust code, and also carries the reward function for all four actor types (snake, prey, amphibia, corpsefag). `step()` returns a nested tuple of four `(obs, rewards, dones, terminal_obs)` groups, one per actor type, in that order.

### 3. ML Training Client (`learner`)
A Python package managed by `uv`. It leverages **Stable-Baselines3** to train four independent policies via Proximal Policy Optimization (PPO): snake, prey, amphibia, and corpsefag. The corpsefag policy trains against a frozen snake model as a separate track, not part of the snake↔prey/amphibia co-evolution loop (it isn't chained into the `train-all-stages` pipeline and isn't consumed by `test.py`).
- **`environment.py`**: Contains `RustMultiSnakeVecEnv` and `MultiProcRustVecEnv` — trains the snake policy. Tricks SB3 into thinking it's interacting with $K$ single-player games, while actually interacting with multiple PyO3 instances containing arbitrary numbers of snakes plus frozen prey/amphibia opponents. It bypasses the Python GIL using multiprocessing Pipes, and internally manages actions for any configured existing snake models to allow mixed-model self-play.
- **`prey_environment.py`** / **`amphibia_environment.py`** / **`corpsefag_environment.py`**: Mirror structure, training the herbivores or scavengers against frozen opponent models.
- **`model_utils.py`**: Shared `load_opponent`/`predict_actions` helpers used by all envs — loads a frozen counterpart model with a graceful fallback (static action `0`) when the checkpoint is missing or its observation shape doesn't match, so the pipeline can bootstrap without any pre-existing models; also batches a counterpart's action prediction across every game in one `predict()` call.
- **`main.py`** / **`train_prey.py`** / **`train_amphibia.py`** / **`train_corpsefag.py`**: Training orchestrators, one per policy. `main.py` distributes simulation instances across background CPU cores (`--num-procs`) and supports configuring multiple existing snake models via `--existing`.
- **`play.py`**: A lightweight TCP server acting as an inference engine for the Bevy visualizer. Supports dynamically sized buffers to infer actions for arbitrary numbers of snakes/prey/amphibia/corpsefags from multiple loaded models.
- **`test.py`**: A headless, fast-forward testing script. It runs the PyO3 simulation at uncapped speeds natively in Python, extracting metrics like kills, length, causes of death, and prey/amphibia survival into a JSON dump.

### 4. Bevy Visualizer (`animals_game`)
A 2D graphical representation of the game using the Bevy framework in Rust. It can spawn any number of snakes (`--snakes N`), land prey (`--preys N`), amphibia (`--amphibias N`), and corpsefags (`--corpsefags N`).
- **AI Mode Integration**: When launched with `--ai`, the Rust game dynamically acquires a free ephemeral TCP port, spawns `learner.play` as a background child process, and feeds it raw observation data via TCP.
- **Lifecycle Management**: The Bevy app retains strict ownership over the spawned Python process, killing it cleanly via custom `Drop` trait logic when the user closes the window.

## Data Flow Diagram

```mermaid
graph TD
    subgraph Rust Workspace
        engine[animals_engine - Core Logic]
        sim[animals_simulation - PyO3 Binding]
        game[animals_game - Bevy App]
        
        sim -->|Wraps| engine
        game -->|Uses| engine
    end

    subgraph Python Environment (ML Client)
        env[environment.py - Custom VecEnv]
        main[main.py - SB3 Training]
        play[play.py - Inference TCP Server]
        
        main --> env
        env -->|In-memory FFI| sim
    end

    game <-->|TCP (Dynamic Port)| play
```
