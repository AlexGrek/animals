import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from typing import Tuple, Dict, Any, Optional, List
import os
import animals_simulation
from stable_baselines3 import PPO

from learner.constants import SNAKE_OBS_SIZE, PREY_OBS_SIZE, PREY_NUM_ACTIONS, SNAKE_NUM_ACTIONS
from learner.model_utils import load_opponent, predict_actions


class RustPreyVecEnv(VecEnv):
    def __init__(self, num_games: int = 4, snakes_per_game: int = 2, preys_per_game: int = 32, max_preys: int = 100, snake_model_path: str = "models/snake_model.zip"):
        self.num_games = num_games
        self.snakes_per_game = snakes_per_game
        self.preys_per_game = preys_per_game
        self.max_preys = max_preys
        self.total_snakes = num_games * snakes_per_game
        self.total_preys = num_games * max_preys

        # Frozen predator snake model; random-action fallback if missing or the
        # checkpoint predates the current observation size (see model_utils).
        self.snake_model = load_opponent(snake_model_path, SNAKE_OBS_SIZE)

        action_space = spaces.Discrete(PREY_NUM_ACTIONS)  # 0 Stand, 1 Up, 2 Right, 3 Down, 4 Left
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(PREY_OBS_SIZE,), dtype=np.float32)

        super().__init__(self.total_preys, observation_space, action_space)

        # Instantiate simulation with multiple preys, no amphibias.
        self.games = [animals_simulation.Simulation(snakes_per_game, preys_per_game, max_preys, 0, 0) for _ in range(num_games)]

        # Buffers
        self.all_prey_obs = np.zeros((self.total_preys, PREY_OBS_SIZE), dtype=np.float32)
        self.all_prey_rews = np.zeros((self.total_preys,), dtype=np.float32)
        self.all_prey_dones = np.zeros((self.total_preys,), dtype=bool)
        self.all_prey_infos = [{} for _ in range(self.total_preys)]

        # Last snake observations (needed to predict their actions)
        self.last_snake_obs = np.zeros((self.total_snakes, SNAKE_OBS_SIZE), dtype=np.float32)

    def reset(self) -> np.ndarray:
        for i, game in enumerate(self.games):
            snake_obs = game.reset()
            for s in range(self.snakes_per_game):
                self.last_snake_obs[i * self.snakes_per_game + s] = snake_obs[s]
            prey_obs_list = game.get_all_prey_observations()
            for p in range(self.max_preys):
                self.all_prey_obs[i * self.max_preys + p] = prey_obs_list[p]
        return np.copy(self.all_prey_obs)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[Dict[str, Any]]]:
        # 1. Compute snake actions (batched; random fallback for a None model)
        snake_actions = predict_actions(self.snake_model, self.last_snake_obs, SNAKE_NUM_ACTIONS)

        # 2. Step all games
        for i, game in enumerate(self.games):
            start_s_idx = i * self.snakes_per_game
            s_actions = snake_actions[start_s_idx:start_s_idx + self.snakes_per_game]

            start_p_idx = i * self.max_preys
            a_actions = self.actions[start_p_idx:start_p_idx + self.max_preys].tolist()

            (snake_obs, _, _, _,
             prey_obs_list, prey_rew_list, prey_done_list,
             _, _, _,
             prey_terminal_list, _) = game.step(s_actions, a_actions, [])

            # Save next snake obs
            for s in range(self.snakes_per_game):
                self.last_snake_obs[start_s_idx + s] = snake_obs[s]

            for p in range(self.max_preys):
                p_global_idx = start_p_idx + p

                # No more per-sibling death bonus: a sibling's death is outside this
                # agent's control, so rewarding it only adds variance, and reproduction
                # now also sets done, which would have showered +2 on everyone per
                # birth. The Rust reward is the single source of truth.
                self.all_prey_rews[p_global_idx] = prey_rew_list[p]

                self.all_prey_obs[p_global_idx] = prey_obs_list[p]
                self.all_prey_dones[p_global_idx] = prey_done_list[p]

                if prey_done_list[p]:
                    # True pre-respawn terminal observation (prey_obs_list[p] is
                    # already the fresh post-respawn state).
                    self.all_prey_infos[p_global_idx] = {
                        "terminal_observation": np.array(prey_terminal_list[p], dtype=np.float32)
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
