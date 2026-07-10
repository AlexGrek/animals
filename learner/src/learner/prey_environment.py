import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from typing import Tuple, Dict, Any, Optional, List
import os
import animals_simulation
from stable_baselines3 import PPO

class RustPreyVecEnv(VecEnv):
    def __init__(self, num_games: int = 4, snakes_per_game: int = 2, preys_per_game: int = 1, snake_model_path: str = "models/snake_model.zip"):
        self.num_games = num_games
        self.snakes_per_game = snakes_per_game
        self.preys_per_game = preys_per_game
        self.total_snakes = num_games * snakes_per_game
        self.total_preys = num_games * preys_per_game
        
        # Load the snake model
        if not os.path.exists(snake_model_path):
            alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "learner", snake_model_path)
            if os.path.exists(alt_path):
                snake_model_path = alt_path
            else:
                alt_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models/snake_model.zip")
                if os.path.exists(alt_path2):
                    snake_model_path = alt_path2
                else:
                    raise FileNotFoundError(f"Snake model not found at {snake_model_path}")
                    
        self.snake_model = PPO.load(snake_model_path)
        
        action_space = spaces.Discrete(5) # 0: Stand, 1: Up, 2: Right, 3: Down, 4: Left
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(64,), dtype=np.float32)
        
        super().__init__(self.total_preys, observation_space, action_space)
        
        # Instantiate simulation with multiple preys
        self.games = [animals_simulation.Simulation(snakes_per_game, preys_per_game) for _ in range(num_games)]
        
        # Buffers
        self.all_prey_obs = np.zeros((self.total_preys, 64), dtype=np.float32)
        self.all_prey_rews = np.zeros((self.total_preys,), dtype=np.float32)
        self.all_prey_dones = np.zeros((self.total_preys,), dtype=bool)
        self.all_prey_infos = [{} for _ in range(self.total_preys)]
        
        # Last snake observations (needed to predict their actions)
        self.last_snake_obs = np.zeros((self.total_snakes, 66), dtype=np.float32)

    def reset(self) -> np.ndarray:
        for i, game in enumerate(self.games):
            snake_obs = game.reset()
            for s in range(self.snakes_per_game):
                self.last_snake_obs[i * self.snakes_per_game + s] = snake_obs[s]
            prey_obs_list = game.get_all_prey_observations()
            for p in range(self.preys_per_game):
                self.all_prey_obs[i * self.preys_per_game + p] = prey_obs_list[p]
        return np.copy(self.all_prey_obs)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[Dict[str, Any]]]:
        # 1. Compute snake actions
        snake_actions, _ = self.snake_model.predict(self.last_snake_obs, deterministic=False)
        
        # 2. Step all games
        for i, game in enumerate(self.games):
            start_s_idx = i * self.snakes_per_game
            end_s_idx = start_s_idx + self.snakes_per_game
            s_actions = snake_actions[start_s_idx:end_s_idx].tolist()
            
            start_p_idx = i * self.preys_per_game
            end_p_idx = start_p_idx + self.preys_per_game
            p_actions = self.actions[start_p_idx:end_p_idx].tolist()
            
            snake_obs, _, _, _, prey_obs_list, prey_rew_list, prey_done_list = game.step(s_actions, p_actions)
            
            # Save next snake obs
            for s in range(self.snakes_per_game):
                self.last_snake_obs[start_s_idx + s] = snake_obs[s]
                
            for p in range(self.preys_per_game):
                p_global_idx = start_p_idx + p
                base_reward = prey_rew_list[p]
                
                # Check if another prey died this tick
                other_died_reward = 0.0
                if self.preys_per_game > 1:
                    for other_p in range(self.preys_per_game):
                        if other_p != p and prey_done_list[other_p]:
                            other_died_reward += 2.0 # Give small reward each time another prey is eaten
                
                # Surviving preys get the bonus
                if not prey_done_list[p]:
                    self.all_prey_rews[p_global_idx] = base_reward + other_died_reward
                else:
                    self.all_prey_rews[p_global_idx] = base_reward
                
                self.all_prey_obs[p_global_idx] = prey_obs_list[p]
                self.all_prey_dones[p_global_idx] = prey_done_list[p]
                
                if prey_done_list[p]:
                    self.all_prey_infos[p_global_idx] = {
                        "terminal_observation": np.array(prey_obs_list[p], dtype=np.float32)
                    }
                else:
                    self.all_prey_infos[p_global_idx] = {}
                
        return np.copy(self.all_prey_obs), np.copy(self.all_prey_rews), np.copy(self.all_prey_dones), list(self.all_prey_infos)

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return [None] * self.num_envs

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        pass

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        return [False] * self.num_envs
