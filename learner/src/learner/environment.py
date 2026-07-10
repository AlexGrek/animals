import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from typing import Tuple, Dict, Any, Optional

try:
    import animals_simulation
except ImportError:
    # Allow fallback during development/mock modes
    animals_simulation = None

class RustMultiSnakeVecEnv(VecEnv):
    """
    A Custom VecEnv that takes N Rust PyO3 simulation instances
    and presents 2N independent environments to Stable-Baselines3.
    This enables shared-brain self-play MARL with native SB3!
    """

    def __init__(self, num_games: int = 4):
        # We have 2 snakes per game, so total envs = num_games * 2
        self.num_games = num_games
        num_envs = num_games * 2
        
        # Action space: 3 discrete actions (0: Straight, 1: Turn Right, 2: Turn Left)
        action_space = spaces.Discrete(3)
        
        # Observation space: 66 floats (8x8 grid + 2D global direction to apple)
        observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(66,), 
            dtype=np.float32
        )
        
        super().__init__(num_envs, observation_space, action_space)

        if animals_simulation is None:
            raise ImportError(
                "Could not import 'animals_simulation'. "
                "Please build the Rust subproject using maturin (e.g. 'task build-sim')."
            )

        self.games = [animals_simulation.Simulation() for _ in range(num_games)]
        
        # Buffers for VecEnv Step Returns
        self.buf_obs = np.zeros((num_envs, 66), dtype=np.float32)
        self.buf_rews = np.zeros((num_envs,), dtype=np.float32)
        self.buf_dones = np.zeros((num_envs,), dtype=bool)
        self.buf_infos = [{} for _ in range(num_envs)]
        
        self.actions = np.zeros((num_envs,), dtype=int)

    def reset(self) -> np.ndarray:
        for i, game in enumerate(self.games):
            obs0, obs1 = game.reset()
            self.buf_obs[i * 2] = obs0
            self.buf_obs[i * 2 + 1] = obs1
        return np.copy(self.buf_obs)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[Dict[str, Any]]]:
        for i, game in enumerate(self.games):
            # Get actions for both snakes in this game
            a0 = int(self.actions[i * 2])
            a1 = int(self.actions[i * 2 + 1])
            
            # Step the rust game
            (o0, o1), (r0, r1), (d0, d1) = game.step(a0, a1)
            
            # Game terminates if either snake is dead.
            # In our rust code, we set terminated=True for both if game_over.
            game_done = bool(d0 or d1)
            
            self.buf_rews[i * 2] = r0
            self.buf_rews[i * 2 + 1] = r1
            self.buf_dones[i * 2] = game_done
            self.buf_dones[i * 2 + 1] = game_done
            
            self.buf_infos[i * 2] = {}
            self.buf_infos[i * 2 + 1] = {}
            
            if game_done:
                # Save terminal observation for SB3 auto-reset logic
                self.buf_infos[i * 2]["terminal_observation"] = np.array(o0, dtype=np.float32)
                self.buf_infos[i * 2 + 1]["terminal_observation"] = np.array(o1, dtype=np.float32)
                
                # Auto-reset
                o0, o1 = game.reset()
                
            self.buf_obs[i * 2] = o0
            self.buf_obs[i * 2 + 1] = o1

        return np.copy(self.buf_obs), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy()

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        """SB3 required method, but we don't have python-level attributes."""
        if attr_name == "render_mode":
            return [None] * self.num_envs
        return [None] * self.num_envs

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        raise NotImplementedError("RustMultiSnakeVecEnv does not support set_attr")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        raise NotImplementedError("RustMultiSnakeVecEnv does not support env_method")

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        # We are not wrapping any standard gym envs!
        return [False] * self.num_envs
