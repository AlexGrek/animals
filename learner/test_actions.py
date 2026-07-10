import animals_simulation
from stable_baselines3 import PPO
import numpy as np

sim = animals_simulation.Simulation(1)
model = PPO.load("models/snake_model")
obs = sim.reset()
obs = np.array(obs, dtype=np.float32)

for _ in range(20):
    a, _ = model.predict(obs, deterministic=True)
    action = int(a[0])
    obs, rewards, dones, _ = sim.step([action])
    obs = np.array(obs, dtype=np.float32)
    print(f"Action: {action}, Reward: {rewards[0]}, Done: {dones[0]}")
