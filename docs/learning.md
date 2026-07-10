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

## The Vector Environment Trick & Mixed-Model Training
Stable-Baselines3 natively only supports single-agent environments. To enable MARL (Multi-Agent RL) without migrating to heavy libraries like PettingZoo, we built a custom `RustMultiSnakeVecEnv`. 

This environment:
1. Spawns $N$ Rust PyO3 instances (representing $2N$ total snakes).
2. Randomly assigns the $2N$ snake slots to either the model actively being trained or to one or more statically loaded, existing AI models.
3. Exposes only the $K$ environments assigned to the active training model to SB3.
4. During a step, it intercepts the $K$ actions from SB3, internally computes actions for the existing AI models using their past checkpoints, steps the $N$ Rust games, and returns only the $K$ outcomes to SB3.

This effectively runs self-play seamlessly inside the PPO agent's vectorized collection buffer, while enabling the agent to learn robust policies by playing against a diverse roster of past iterations instead of only itself.
## Neural Network Architecture
The agent uses a Deep Multi-Layer Perceptron (MLP) architecture:
- 3 hidden layers: `[256, 256, 256]`
- Framework: PyTorch via Stable-Baselines3
- Algorithm: Proximal Policy Optimization (PPO)
