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

## The Vector Environment Trick
Stable-Baselines3 natively only supports single-agent environments. To enable MARL (Multi-Agent RL) without migrating to heavy libraries like PettingZoo, we built a custom `RustMultiSnakeVecEnv`. 

This environment:
1. Spawns $N$ Rust PyO3 instances.
2. Unpacks the 2 snakes in each instance, presenting $2N$ independent environments to SB3.
3. Repacks the $2N$ predicted actions into pairs and feeds them back into the $N$ Rust games.

This effectively runs self-play seamlessly inside the PPO agent's vectorized collection buffer.

## Neural Network Architecture
The agent uses a Deep Multi-Layer Perceptron (MLP) architecture:
- 3 hidden layers: `[256, 256, 256]`
- Framework: PyTorch via Stable-Baselines3
- Algorithm: Proximal Policy Optimization (PPO)
