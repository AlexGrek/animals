import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

try:
    import animals_simulation
except ImportError:
    # Allow fallback during development/mock modes
    animals_simulation = None

class SimulationEnv(gym.Env):
    """Custom Gymnasium Environment wrapper around the compiled PyO3 animals_simulation library."""
    
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        
        # Define action space: 3 discrete actions (0: Straight, 1: Turn Right, 2: Turn Left)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space: 8-dimensional float vector:
        # [0] Danger Straight, [1] Danger Left, [2] Danger Right
        # [3] Food Ahead, [4] Food Behind, [5] Food Left, [6] Food Right
        # [7] Normalized Distance to Food
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(8,), 
            dtype=np.float32
        )
        
        if animals_simulation is None:
            raise ImportError(
                "Could not import 'animals_simulation'. "
                "Please build the Rust subproject using maturin (e.g. 'task build-sim')."
            )
            
        self.sim = animals_simulation.Simulation()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        obs_list = self.sim.reset()
        observation = np.array(obs_list, dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs_list, reward, terminated, truncated = self.sim.step(int(action))
        
        observation = np.array(obs_list, dtype=np.float32)
        info = {}
        
        return observation, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        pass
