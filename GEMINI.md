# Animals Simulation Workspace Architecture

This project is a multi-language workspace combining a Rust-based game engine/simulation with a Python-based machine learning (Reinforcement Learning) client.

For detailed architecture logic, refer to:
- [Architecture Details](file:///Users/vedmedik/dev/animals/docs/architecture.md)
- [Reinforcement Learning Specs](file:///Users/vedmedik/dev/animals/docs/learning.md)

```mermaid
graph TD
    subgraph Rust Workspace
        game[animals_game - Bevy App]
        engine[animals_engine - Rust Lib]
        sim[animals_simulation - PyO3 API Layer]
        
        game -->|Uses| engine
        sim -->|Uses| engine
    end

    subgraph Python Environment (Client)
        learner[learner - ML Project]
        agent[agent.py - PyTorch DQN]
        env[environment.py - Gymnasium Env]
        
        learner --> agent
        learner --> env
        
        env -->|FFI calls| sim
    end
```

## System Components

### 1. Game Engine & Simulation (`animals_game` & `animals_engine`)
* **`animals_game`**: A Bevy-based 2D application that manages the game loop, graphics/rendering, physics, and agent simulation environment. This is the standalone graphical game.
* **`animals_engine`**: A library crate shared by engine tools containing shared logic, math, and helper functions (e.g., coordinates, calculations).

### 2. PyO3 API Layer (`animals_simulation`)
* **`animals_simulation`**: A Rust library crate (`cdylib`) that acts as an API layer for the Python ML stage. It is built using **PyO3 and Maturin** and provides a C-extension module that the Python code can import directly. This eliminates the need for IPC/sockets, allowing the ML loop to step the simulation synchronously and efficiently in-memory.

### 3. Reinforcement Learning Client (`learner`)
A Python package managed by `uv` containing the ML training loop:
* **`environment.py`**: A custom `gymnasium.Env` wrapper that directly calls the compiled `animals_simulation` PyO3 module to step the state and gather observations/rewards.
* **`agent.py`**: A PyTorch deep reinforcement learning agent (DQN skeleton) deciding actions based on environment states.
* **`main.py`**: Orchestrator executing the training loop.

---

## Communication Protocol (PyO3 FFI)

The Python client and the Rust simulation engine communicate directly via FFI (Foreign Function Interface) provided by PyO3.

### Handshake / Reset
The Python environment creates an instance of the Rust `Simulation` object and calls its `reset()` method:
```python
import animals_simulation

sim = animals_simulation.Simulation()
obs0, obs1 = sim.reset()
```

### Action / Step
For each simulation step, the Python environment calls the `step(action0, action1)` method on the Rust object:
```python
(o0, o1), (r0, r1), (d0, d1) = sim.step(action0, action1)
```
This synchronously executes the physics/logic in Rust and returns the next frame's data back to Python.
