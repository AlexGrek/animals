import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from typing import Tuple, Dict, Any, Optional, List
import os
import animals_simulation
from stable_baselines3 import PPO

from learner.constants import SNAKE_OBS_SIZE, CORPSEFAG_OBS_SIZE, CORPSEFAG_NUM_ACTIONS, SNAKE_NUM_ACTIONS
from learner.model_utils import load_opponent, predict_actions


class RustCorpsefagVecEnv(VecEnv):
    def __init__(self, num_games: int = 4, snakes_per_game: int = 2, num_corpsefags: int = 10, snake_model_path: str = "models/snake_model.zip"):
        self.num_games = num_games
        self.snakes_per_game = snakes_per_game
        self.num_corpsefags = num_corpsefags
        self.total_snakes = num_games * snakes_per_game
        self.total_corpsefags = num_games * num_corpsefags

        # Frozen predator snake model
        self.snake_model = load_opponent(snake_model_path, SNAKE_OBS_SIZE)

        action_space = spaces.Discrete(CORPSEFAG_NUM_ACTIONS)  # 0 Stand, 1 Up, 2 Right, 3 Down, 4 Left
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(CORPSEFAG_OBS_SIZE,), dtype=np.float32)

        super().__init__(self.total_corpsefags, observation_space, action_space)

        self.games = [animals_simulation.Simulation(snakes_per_game, 0, 0, 0, 0, num_corpsefags) for _ in range(num_games)]

        # Buffers
        self.all_cf_obs = np.zeros((self.total_corpsefags, CORPSEFAG_OBS_SIZE), dtype=np.float32)
        self.all_cf_rews = np.zeros((self.total_corpsefags,), dtype=np.float32)
        self.all_cf_dones = np.zeros((self.total_corpsefags,), dtype=bool)
        self.all_cf_infos = [{} for _ in range(self.total_corpsefags)]

        # Last snake observations (needed to predict their actions)
        self.last_snake_obs = np.zeros((self.total_snakes, SNAKE_OBS_SIZE), dtype=np.float32)

        self.steps = 0

    def reset(self) -> np.ndarray:
        for i, game in enumerate(self.games):
            snake_data, _, _, cf_data = game.reset()
            game.spawn_corpses(50)
            for s in range(self.snakes_per_game):
                self.last_snake_obs[i * self.snakes_per_game + s] = snake_data[s]
            for p in range(self.num_corpsefags):
                self.all_cf_obs[i * self.num_corpsefags + p] = cf_data[p]
        return np.copy(self.all_cf_obs)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[Dict[str, Any]]]:
        # 1. Compute snake actions
        snake_actions = predict_actions(self.snake_model, self.last_snake_obs, SNAKE_NUM_ACTIONS)

        self.steps += 1
        if self.steps % 500 == 0:
            for game in self.games:
                game.spawn_corpses(25)

        # 2. Step all games
        for i, game in enumerate(self.games):
            start_s_idx = i * self.snakes_per_game
            s_actions = snake_actions[start_s_idx:start_s_idx + self.snakes_per_game]

            start_p_idx = i * self.num_corpsefags
            c_actions = self.actions[start_p_idx:start_p_idx + self.num_corpsefags].tolist()

            snake_data, _, _, cf_data = game.step(s_actions, [], [], c_actions)
            snake_obs, _, _, _ = snake_data
            cf_obs_list, cf_rew_list, cf_done_list, cf_terminal_list = cf_data

            # Save next snake obs
            for s in range(self.snakes_per_game):
                self.last_snake_obs[start_s_idx + s] = snake_obs[s]

            for p in range(self.num_corpsefags):
                p_global_idx = start_p_idx + p

                self.all_cf_rews[p_global_idx] = cf_rew_list[p]
                self.all_cf_obs[p_global_idx] = cf_obs_list[p]
                self.all_cf_dones[p_global_idx] = cf_done_list[p]

                if cf_done_list[p]:
                    self.all_cf_infos[p_global_idx] = {
                        "terminal_observation": np.array(cf_terminal_list[p], dtype=np.float32)
                    }
                else:
                    self.all_cf_infos[p_global_idx] = {}

        return np.copy(self.all_cf_obs), np.copy(self.all_cf_rews), np.copy(self.all_cf_dones), list(self.all_cf_infos)

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
