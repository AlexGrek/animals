# Reinforcement Learning Details

The RL system utilizes a competitive self-play setup where two snakes share the same neural network to learn both cooperative and competitive behaviors simultaneously.

## Observation Space (Sensory Grid)
Each snake "sees" an 8x8 patch of the map centered slightly ahead of its head. The grid uses semantic encoding to represent game objects:

- **0.0**: Empty Space
- **1.0**: Food (Apple)
- **-0.5**: Opponent's Body (Danger, but also a target)
- **-1.0**: Solid Obstacle (Wall or own body)

In addition to the 64-float grid, the network receives a 2D normalized vector representing the global direction to the apple, yielding a final observation array of **66 floats**.

## Reward Function
The reward function is tuned to encourage aggression and survival:

- **Eat Apple**: `+10.0`
- **Kill Opponent**: `+50.0` (Triggered when the opponent collides with your body)
- **Death**: `-10.0` (Wall collision, self-collision, head-to-head collision, or getting trapped by the opponent)
- **Time Penalty**: `-0.01` (Per step, to encourage efficient pathing)

## Episode Termination: Per-Snake Respawn

Snakes do **not** share a single game-over condition. Previously, the whole `GameState` set a global `game_over` flag the instant any one snake died, which truncated every other (still-alive) snake's episode through no fault of its own, wasted roughly half the collected samples, and biased the value function (SB3 was never told those were truncations rather than true terminal states, since `TimeLimit.truncated` was never set).

Instead, `GameState::step()` never sets `game_over` on death. When a snake dies (wall, self-collision, opponent-collision, or head-to-head), it is immediately respawned by `GameState::respawn_dead()`: fresh body of length 1 at a spawn position (the same evenly-spaced-columns/mid-height layout used at game start, falling back to a random free cell if that column is occupied), with score, kills, and death-cause flags reset for the new life.

The PyO3 `Simulation.step()` API reflects this with **per-snake** `dones`: `dones[i]` is `True` exactly on the tick snake `i` died, independent of the other snakes. Because the snake is respawned within the same `step()` call, the `obs` returned for a snake that just died is already its post-respawn ("reset") observation; the pre-respawn terminal observation is returned separately as `terminal_obs[i]` (meaningful only where `dones[i]` is `True`), so `RustMultiSnakeVecEnv.step_wait()` can populate SB3's `infos[i]["terminal_observation"]` correctly without needing to reset the whole game.

Head-to-head collisions (two snakes' heads landing on the same cell in the same tick) kill **both** snakes — this is computed from a pre-step snapshot of alive snakes and their next head positions, so which snake happens to be checked first in the collision loop no longer matters.

The Bevy visualizer (`animals_game`) still wants a classic "game over, press Space to restart" experience for manual/AI-watch play: it detects any snake death itself after calling `engine.step()` and sets `GameState.game_over` (a field the engine keeps around for exactly this purpose, but no longer mutates itself).

## The Vector Environment Trick & Mixed-Model Training
Stable-Baselines3 natively only supports single-agent environments. To enable MARL (Multi-Agent RL) without migrating to heavy libraries like PettingZoo, we built a custom `RustMultiSnakeVecEnv`. 

This environment:
1. Spawns multiprocessing workers, each managing multiple PyO3 Rust instances (representing $S$ total snakes across all instances).
2. Randomly assigns the $S$ snake slots across all parallel instances to either the model actively being trained or to one or more statically loaded, existing AI models.
3. Exposes only the $K$ environments assigned to the active training model to SB3.
4. During a step, it intercepts the $K$ actions from SB3, internally computes actions for the existing AI models using their past checkpoints, steps the PyO3 games in background processes to bypass the GIL, and returns only the $K$ outcomes to SB3.

This effectively runs self-play seamlessly inside the PPO agent's vectorized collection buffer, while enabling the agent to learn robust policies by playing against a diverse roster of past iterations instead of only itself.
## Neural Network Architecture
The agent uses a Deep Multi-Layer Perceptron (MLP) architecture:
- 3 hidden layers: `[256, 256, 256]`
- Framework: PyTorch via Stable-Baselines3
- Algorithm: Proximal Policy Optimization (PPO)

## PPO Hyperparameters & CPU Throughput

Training runs on `device="cpu"` (the policy MLP is small enough that GPU host↔device transfer/launch overhead exceeds the compute it would save). On CPU, PPO's optimizer step count dominates wall-clock far more than environment rollout speed: with SB3's defaults (`batch_size=64`, `n_steps=2048`) and 16 parallel training envs, each policy update does `(2048*16/64) = 512` minibatches × 10 epochs = 5,120 tiny optimizer steps, versus rollout collection alone running at ~60,000 steps/s. Measured throughput was ~3,100 steps/s overall.

We instead use:
- `batch_size=4096` — far fewer, larger minibatches per update, cutting Python/optimizer overhead drastically (measured ~14,000 steps/s, a ~4.5x wall-clock speedup).
- `n_steps=512` — smaller rollout buffer per env, so policy updates happen more frequently for the same total sample count.
- `ent_coef=0.01` — encourages exploration, which matters more with the sparse (`+50`/`+10`/`-10`) reward structure.
