# Architecture Overview

This workspace is a multi-language, multi-agent reinforcement learning (MARL) project that combines a Rust-based game engine with a Python-based ML training pipeline. It utilizes an elegant **Shared Brain** architecture to enable seamless self-play using Stable-Baselines3.

## Core Components

### 1. Game Engine (`animals_engine`)
A lightweight, headless Rust library containing the core logic, physics, and state representations for the Snake game. It handles all collision math, movement, and reward tracking for multiple snakes. There is no global "game over": snakes die independently (wall, self, opponent, or head-to-head collision — the latter always kills both snakes involved) and are respawned in place via `GameState::respawn_dead()`, so one snake's death doesn't truncate the others' episodes. See `docs/learning.md` for how this maps to per-snake `dones` in the training environment.

### 2. PyO3 Simulation Binding (`animals_simulation`)
A Rust library compiled to a Python C-extension via Maturin. This exposes the `animals_engine` logic directly into the Python memory space, circumventing the overhead of Inter-Process Communication (IPC). The Python `step()` calls directly execute the highly optimized compiled Rust code.

### 3. ML Training Client (`learner`)
A Python package managed by `uv`. It leverages **Stable-Baselines3** to train the agents via Proximal Policy Optimization (PPO).
- **`environment.py`**: Contains `RustMultiSnakeVecEnv` and `MultiProcRustVecEnv`. It tricks SB3 into thinking it's interacting with $K$ single-player games, while actually interacting with multiple PyO3 instances containing arbitrary numbers of snakes. It bypasses the Python GIL using multiprocessing Pipes, and internally manages actions for any configured existing AI models to allow mixed-model training.
- **`main.py`**: The training orchestrator. Distributes simulation instances across background CPU cores (`--num-procs`) and supports configuring multiple existing models alongside the training model via the `--existing` flag.
- **`play.py`**: A lightweight TCP server acting as an inference engine for the Bevy visualizer. Supports dynamically sized buffers to infer actions for arbitrary numbers of snakes from multiple loaded models.
- **`test.py`**: A headless, fast-forward testing script. It runs the PyO3 simulation at uncapped speeds natively in Python, extracting metrics like kills, length, and causes of death into a JSON dump.

### 4. Bevy Visualizer (`animals_game`)
A 2D graphical representation of the game using the Bevy framework in Rust. It can spawn any number of snakes (`--snakes N`).
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
