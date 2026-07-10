# Animals Multi-Agent Reinforcement Learning (MARL)

A hybrid Rust and Python workspace for training reinforcement learning agents to play Snake in a competitive, multi-agent environment using Stable-Baselines3.

## Overview

This project implements an elegant "Shared Brain" architecture to solve multi-agent self-play using standard single-agent libraries. It leverages the performance of a Rust-based game engine directly embedded into Python memory via PyO3, completely bypassing Inter-Process Communication (IPC) overhead during training. 

The environment can be distributed across multiple CPU cores, allowing you to train hundreds of agents simultaneously at tens of thousands of frames per second.

## Features

- **Blazing Fast Headless Simulation**: The core game engine (`animals_engine`) is written in Rust.
- **Zero-Overhead Bindings**: `animals_simulation` uses PyO3 to expose the engine directly to Python as a C-extension.
- **Multi-Agent / Mixed-Model Training**: A custom SB3 Vector Environment tricks the trainer into thinking it is playing single-player games. In reality, the environment manages multiple existing opponent models, predicting actions dynamically so the training agent plays against a diverse roster of past iterations.
- **Multiprocessing**: Bypasses the Python Global Interpreter Lock (GIL) by spawning background workers, scaling linearly with CPU cores.
- **Bevy Visualizer**: A 2D frontend (`animals_game`) that spawns a lightweight TCP inference server to let you watch the agents play in real-time.
- **Headless Fast-Forward Testing**: Evaluate an arbitrary number of snakes and models at uncapped speeds and dump the analytics into JSON.

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Reinforcement Learning Specs](docs/learning.md)

## Prerequisites

- [Rust](https://rustup.rs/) (1.85+, required for edition 2024)
- [uv](https://github.com/astral-sh/uv) (for ultra-fast Python package management)
- Python 3.14+
- task (Taskfile runner)

## Getting Started

1. **Build the PyO3 Simulation Binding**
   The Python environment requires the Rust simulation to be compiled first:
   ```bash
   task build-sim
   ```

2. **Train an Agent**
   Train a model using Stable-Baselines3. You can configure the number of parallel games, snakes per game, and the number of CPU processes.
   ```bash
   cd learner && uv run python src/learner/main.py --num-games 16 --snakes-per-game 2 --num-procs 4 --steps 1000000
   ```
   *Note: This will save the model to `models/snake_model.zip` by default.*

3. **Train against Past Iterations (Mixed-Model Training)**
   Continue training a new model while pitting it against 4 instances of `v1` and 2 instances of `v2`:
   ```bash
   cd learner && uv run python src/learner/main.py --num-games 16 --snakes-per-game 2 --existing v1:4 --existing v2:2
   ```

## Evaluation & Visualization

**Watch the Agents Play**
Use the Bevy visualizer to watch your trained models in action. The Rust application automatically spawns the Python inference server.
```bash
task play-ai -- --snakes 4 --model snake_model --model v1
```
*(If you supply M models and N snakes, models map 1:1. If you supply 1 model, it is duplicated for all N snakes).*

**Headless Fast-Forward Test**
Run a full-speed simulation without rendering and dump the final game statistics (score, length, kills, causes of death) to a JSON file.
```bash
task test-ai -- --snakes 4 --model snake_model --output metrics.json
```
