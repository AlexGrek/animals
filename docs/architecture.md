# Architecture Overview

This workspace is a multi-language, multi-agent reinforcement learning (MARL) project that combines a Rust-based game engine with a Python-based ML training pipeline. It utilizes an elegant **Shared Brain** architecture to enable seamless self-play using Stable-Baselines3.

## Core Components

### 1. Game Engine (`animals_engine`)
A lightweight, headless Rust library containing the core logic, physics, and state representations for the Snake game. It handles all collision math, movement, and reward tracking for multiple snakes.

### 2. PyO3 Simulation Binding (`animals_simulation`)
A Rust library compiled to a Python C-extension via Maturin. This exposes the `animals_engine` logic directly into the Python memory space, circumventing the overhead of Inter-Process Communication (IPC). The Python `step()` calls directly execute the highly optimized compiled Rust code.

### 3. ML Training Client (`learner`)
A Python package managed by `uv`. It leverages **Stable-Baselines3** to train the agents via Proximal Policy Optimization (PPO).
- **`environment.py`**: Contains `RustMultiSnakeVecEnv`, a custom `VecEnv` that tricks SB3 into thinking it's interacting with $K$ single-player games, while actually interacting with $N$ PyO3 games containing 2 snakes each. It internally manages and generates actions for any configured existing AI models, allowing the training agent to play against a diverse set of opponents (mixed-model training).
- **`main.py`**: The training orchestrator. Supports configuring multiple existing models alongside the actively training model via the `--existing` flag.
- **`play.py`**: A lightweight TCP server designed to act as an inference engine for the Bevy visualizer.

### 4. Bevy Visualizer (`animals_game`)
A 2D graphical representation of the game using the Bevy framework in Rust. 
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
