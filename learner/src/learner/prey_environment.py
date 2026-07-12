import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from typing import Tuple, Dict, Any, Optional, List
import os
import random
import animals_simulation
from stable_baselines3 import PPO

from learner.constants import SNAKE_OBS_SIZE, PREY_OBS_SIZE, PREY_NUM_ACTIONS, SNAKE_NUM_ACTIONS
from learner.model_utils import load_opponent, predict_actions


class RustPreyVecEnv(VecEnv):
    def __init__(self, num_games: int = 4, snakes_per_game: int = 2, 
                 preys_per_game: int = 32, max_preys: int = 100, 
                 training_count: Optional[int] = None,
                 existing_models: Optional[Dict[str, int]] = None,
                 snake_model_path: str = "models/snake_model.zip"):
        self.num_games = num_games
        self.snakes_per_game = snakes_per_game
        self.preys_per_game = preys_per_game
        self.max_preys = max_preys
        self.total_snakes = num_games * snakes_per_game
        self.total_preys = num_games * max_preys

        if existing_models is None:
            existing_models = {}
        
        self.existing_models = {}
        for path, count in existing_models.items():
            if count > 0:
                self.existing_models[path] = load_opponent(path, PREY_OBS_SIZE)

        # Distribute assignments per game so exactly 'count' existing models are in the first 'preys_per_game' slots
        # and the rest are distributed proportionally in the remaining capacity.
        assignments = []
        
        for g in range(num_games):
            # For the first `preys_per_game` (alive at start)
            active_slots = []
            ex_counts_in_active = {}
            for name, count in existing_models.items():
                # count is per game
                active_slots.extend([name] * count)
                ex_counts_in_active[name] = count
                
            train_in_active = self.preys_per_game - len(active_slots)
            if train_in_active < 0:
                raise ValueError(f"preys_per_game ({self.preys_per_game}) cannot be less than total existing models per game ({len(active_slots)})")
            active_slots.extend(['train'] * train_in_active)
            random.shuffle(active_slots)
            
            # For the remaining dead slots, keep roughly the same proportion
            dead_slots = []
            if self.max_preys > self.preys_per_game:
                remaining_capacity = self.max_preys - self.preys_per_game
                for name, count in existing_models.items():
                    prop_count = int(remaining_capacity * (count / self.preys_per_game))
                    dead_slots.extend([name] * prop_count)
                train_in_dead = remaining_capacity - len(dead_slots)
                dead_slots.extend(['train'] * train_in_dead)
                random.shuffle(dead_slots)
            
            assignments.extend(active_slots + dead_slots)
            
        self.training_indices = [i for i, a in enumerate(assignments) if a == 'train']
        self.model_indices = {name: [i for i, a in enumerate(assignments) if a == name] for name in self.existing_models.keys()}
        
        self.training_count = len(self.training_indices)

        # Frozen predator snake model
        self.snake_model = load_opponent(snake_model_path, SNAKE_OBS_SIZE)

        action_space = spaces.Discrete(PREY_NUM_ACTIONS)  # 0 Stand, 1 Up, 2 Right, 3 Down, 4 Left
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(PREY_OBS_SIZE,), dtype=np.float32)

        super().__init__(self.training_count, observation_space, action_space)

        # Instantiate simulation with multiple preys, no amphibias.
        self.games = [animals_simulation.Simulation(snakes_per_game, preys_per_game, max_preys, 0, 0, 0) for _ in range(num_games)]

        # Buffers
        self.all_prey_obs = np.zeros((self.total_preys, PREY_OBS_SIZE), dtype=np.float32)
        self.all_prey_rews = np.zeros((self.total_preys,), dtype=np.float32)
        self.all_prey_dones = np.zeros((self.total_preys,), dtype=bool)
        self.all_prey_infos = [{} for _ in range(self.total_preys)]
        self.actions = np.zeros((self.training_count,), dtype=int)

        # Last snake observations (needed to predict their actions)
        self.last_snake_obs = np.zeros((self.total_snakes, SNAKE_OBS_SIZE), dtype=np.float32)

    def reset(self) -> np.ndarray:
        for i, game in enumerate(self.games):
            snake_obs, _, _, _ = game.reset()
            for s in range(self.snakes_per_game):
                self.last_snake_obs[i * self.snakes_per_game + s] = snake_obs[s]
            prey_obs_list = game.get_all_prey_observations()
            for p in range(self.max_preys):
                self.all_prey_obs[i * self.max_preys + p] = prey_obs_list[p]
        if self.training_count == 0:
            return np.zeros((0, PREY_OBS_SIZE), dtype=np.float32)
        return np.copy(self.all_prey_obs[self.training_indices])

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[Dict[str, Any]]]:
        # 1. Compute snake actions
        snake_actions = predict_actions(self.snake_model, self.last_snake_obs, SNAKE_NUM_ACTIONS)

        # 2. Compute prey actions (mixing training and existing models)
        all_prey_actions = np.zeros(self.total_preys, dtype=int)
        
        for idx, train_idx in enumerate(self.training_indices):
            all_prey_actions[train_idx] = self.actions[idx]
            
        for name, model in self.existing_models.items():
            indices = self.model_indices[name]
            if len(indices) > 0:
                acts = predict_actions(model, self.all_prey_obs[indices], PREY_NUM_ACTIONS)
                for i, idx in enumerate(indices):
                    all_prey_actions[idx] = acts[i]

        # 3. Step all games
        for i, game in enumerate(self.games):
            start_s_idx = i * self.snakes_per_game
            s_actions = snake_actions[start_s_idx:start_s_idx + self.snakes_per_game]

            start_p_idx = i * self.max_preys
            a_actions = all_prey_actions[start_p_idx:start_p_idx + self.max_preys].tolist()

            snakes_data, preys_data, _, _ = game.step(s_actions, a_actions, [], [])
            snake_obs = snakes_data[0]
            prey_obs_list = preys_data[0]
            prey_rew_list = preys_data[1]
            prey_done_list = preys_data[2]
            prey_terminal_list = preys_data[3]

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

        return (
            np.copy(self.all_prey_obs[self.training_indices]), 
            np.copy(self.all_prey_rews[self.training_indices]), 
            np.copy(self.all_prey_dones[self.training_indices]), 
            [self.all_prey_infos[idx] for idx in self.training_indices]
        )

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
