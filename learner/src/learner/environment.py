import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from typing import Tuple, Dict, Any, Optional, List, Callable
import random
import multiprocessing as mp
import os

try:
    import animals_simulation
except ImportError:
    # Allow fallback during development/mock modes
    animals_simulation = None

from stable_baselines3 import PPO

from learner.constants import (
    SNAKE_OBS_SIZE,
    PREY_OBS_SIZE,
    SNAKE_NUM_ACTIONS,
    PREY_NUM_ACTIONS,
)
from learner.model_utils import load_opponent, predict_actions

class RustMultiSnakeVecEnv(VecEnv):
    """
    A Custom VecEnv that takes N Rust PyO3 simulation instances
    and presents training_count independent environments to Stable-Baselines3.
    It internally manages the remaining snakes using existing_models, and drives
    the prey/amphibia opponents with their own frozen models so snakes learn to
    hunt both.
    """

    def __init__(self, num_games: int = 4, snakes_per_game: int = 2, training_count: Optional[int] = None,
                 existing_models: Optional[Dict[str, int]] = None,
                 preys_per_game: int = 2, amphibias_per_game: int = 1,
                 prey_model_path: str = "models/prey_model.zip",
                 amphibia_model_path: str = "models/amphibia_model.zip"):
        self.num_games = num_games
        self.snakes_per_game = snakes_per_game
        self.preys_per_game = preys_per_game
        self.amphibias_per_game = amphibias_per_game
        self.total_snakes = num_games * snakes_per_game

        if training_count is None:
            training_count = self.total_snakes
        if existing_models is None:
            existing_models = {}

        self.training_count = training_count
        self.existing_models = {}

        # Load opponent snake checkpoints, degrading to random actions on a
        # missing file or obs-shape mismatch (see model_utils.load_opponent).
        for path, count in existing_models.items():
            if count > 0:
                self.existing_models[path] = load_opponent(path, SNAKE_OBS_SIZE)

        # Create snake assignment list
        assignments = ['train'] * training_count
        for name, count in existing_models.items():
            assignments.extend([name] * count)

        if len(assignments) != self.total_snakes:
            raise ValueError(f"Total snakes ({self.total_snakes}) does not match sum of training ({training_count}) and existing models.")

        # Shuffle assignments so training agent plays diverse opponents
        random.shuffle(assignments)

        # Pre-compute indices for each model
        self.training_indices = [i for i, a in enumerate(assignments) if a == 'train']
        self.model_indices = {name: [i for i, a in enumerate(assignments) if a == name] for name in self.existing_models.keys()}

        # Action space: 3 discrete actions (0: Straight, 1: Turn Right, 2: Turn Left)
        action_space = spaces.Discrete(SNAKE_NUM_ACTIONS)

        observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(SNAKE_OBS_SIZE,),
            dtype=np.float32
        )

        # Super init with training_count (the number of envs SB3 sees)
        super().__init__(self.training_count, observation_space, action_space)

        if animals_simulation is None:
            raise ImportError(
                "Could not import 'animals_simulation'. "
                "Please build the Rust subproject using maturin (e.g. 'task build-sim')."
            )

        # Frozen prey/amphibia opponents (random-action fallback if unavailable).
        self.prey_model = load_opponent(prey_model_path, PREY_OBS_SIZE)
        self.amphibia_model = load_opponent(amphibia_model_path, PREY_OBS_SIZE)

        self.games = [
            animals_simulation.Simulation(self.snakes_per_game, self.preys_per_game, self.amphibias_per_game)
            for _ in range(num_games)
        ]

        # Global Buffers for ALL snakes
        self.all_obs = np.zeros((self.total_snakes, SNAKE_OBS_SIZE), dtype=np.float32)
        self.all_rews = np.zeros((self.total_snakes,), dtype=np.float32)
        self.all_dones = np.zeros((self.total_snakes,), dtype=bool)
        self.all_infos = [{} for _ in range(self.total_snakes)]

        # SB3 action buffer (only size of training_count)
        self.actions = np.zeros((self.training_count,), dtype=int)

    def reset(self) -> np.ndarray:
        for i, game in enumerate(self.games):
            obs_list = game.reset()
            for s in range(self.snakes_per_game):
                self.all_obs[i * self.snakes_per_game + s] = obs_list[s]
            
        # Return only training observations
        if self.training_count == 0:
            return np.zeros((0, SNAKE_OBS_SIZE), dtype=np.float32)
        return np.copy(self.all_obs[self.training_indices])

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def _gather_prey_actions(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Batch-predict prey and amphibia actions for every game in one shot.

        Returns (prey_actions_per_game, amphibia_actions_per_game). Gathering all
        observations first and running a single vectorized `predict` per model
        replaces the old per-agent single-row forward passes (~32 calls/step).
        """
        prey_obs_all, amphibia_obs_all = [], []
        for game in self.games:
            prey_obs_all.extend(game.get_all_prey_observations())
            amphibia_obs_all.extend(game.get_all_amphibia_observations())

        prey_arr = np.asarray(prey_obs_all, dtype=np.float32).reshape(-1, PREY_OBS_SIZE)
        amphibia_arr = np.asarray(amphibia_obs_all, dtype=np.float32).reshape(-1, PREY_OBS_SIZE)
        prey_flat = predict_actions(self.prey_model, prey_arr, PREY_NUM_ACTIONS)
        amphibia_flat = predict_actions(self.amphibia_model, amphibia_arr, PREY_NUM_ACTIONS)

        prey_per_game, amphibia_per_game = [], []
        for i in range(self.num_games):
            p0 = i * self.preys_per_game
            a0 = i * self.amphibias_per_game
            prey_per_game.append(prey_flat[p0:p0 + self.preys_per_game])
            amphibia_per_game.append(amphibia_flat[a0:a0 + self.amphibias_per_game])
        return prey_per_game, amphibia_per_game

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[Dict[str, Any]]]:
        # 1. Compute actions for all snakes
        all_actions = np.zeros(self.total_snakes, dtype=int)

        # Place training actions
        for idx, train_idx in enumerate(self.training_indices):
            all_actions[train_idx] = self.actions[idx]

        # Compute actions for existing opponent snake models (batched; a None
        # model — missing/incompatible checkpoint — acts uniformly at random).
        for name, model in self.existing_models.items():
            indices = self.model_indices[name]
            if len(indices) > 0:
                acts = predict_actions(model, self.all_obs[indices], SNAKE_NUM_ACTIONS)
                for i, idx in enumerate(indices):
                    all_actions[idx] = acts[i]

        prey_per_game, amphibia_per_game = self._gather_prey_actions()

        for i, game in enumerate(self.games):
            start_idx = i * self.snakes_per_game
            end_idx = start_idx + self.snakes_per_game
            actions_list = all_actions[start_idx:end_idx].tolist()

            obs_list, rews_list, dones_list, terminal_obs_list, *_ = game.step(
                actions_list, prey_per_game[i], amphibia_per_game[i]
            )

            for s in range(self.snakes_per_game):
                idx = start_idx + s
                snake_done = bool(dones_list[s])

                self.all_rews[idx] = rews_list[s]
                self.all_dones[idx] = snake_done
                self.all_obs[idx] = obs_list[s]

                if snake_done:
                    self.all_infos[idx] = {
                        "terminal_observation": np.array(terminal_obs_list[s], dtype=np.float32)
                    }
                else:
                    self.all_infos[idx] = {}

        # 3. Extract and return training snake data
        if self.training_count == 0:
             return np.zeros((0, SNAKE_OBS_SIZE), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool), []

        buf_obs = self.all_obs[self.training_indices]
        buf_rews = self.all_rews[self.training_indices]
        buf_dones = self.all_dones[self.training_indices]
        buf_infos = [self.all_infos[idx] for idx in self.training_indices]

        return np.copy(buf_obs), np.copy(buf_rews), np.copy(buf_dones), buf_infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        if attr_name == "render_mode":
            return [None] * self.num_envs
        return [None] * self.num_envs

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        pass

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        return [False] * self.num_envs

def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                env.step_async(data)
                obs, rews, dones, infos = env.step_wait()
                remote.send((obs, rews, dones, infos))
            elif cmd == 'reset':
                obs = env.reset()
                remote.send(obs)
            elif cmd == 'get_training_count':
                remote.send(env.training_count)
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"Command {cmd} is not implemented in worker.")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Worker failed: {e}")
    finally:
        env.close()

class CloudpickleWrapper:
    def __init__(self, var):
        self.var = var
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.var)
    def __setstate__(self, obs):
        import pickle
        self.var = pickle.loads(obs)

class MultiProcRustVecEnv(VecEnv):
    """
    Multiprocessing wrapper that distributes RustMultiSnakeVecEnv across multiple Python processes.
    This effectively uses multiple CPU cores to step the environments and generate actions for the static models.
    """
    def __init__(self, env_fns: List[Callable[[], RustMultiSnakeVecEnv]]):
        self.waiting = False
        self.closed = False
        self.num_procs = len(env_fns)
        
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_procs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = mp.Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            process.daemon = True
            process.start()
            self.processes.append(process)
            work_remote.close()
            
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(SNAKE_OBS_SIZE,), dtype=np.float32)
        action_space = spaces.Discrete(SNAKE_NUM_ACTIONS)
        
        self.training_counts = []
        for remote in self.remotes:
            remote.send(('get_training_count', None))
            self.training_counts.append(remote.recv())
            
        total_training_envs = sum(self.training_counts)
        super().__init__(total_training_envs, observation_space, action_space)

    def reset(self) -> np.ndarray:
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        
        if self.num_envs == 0:
            return np.zeros((0, SNAKE_OBS_SIZE), dtype=np.float32)

        # We need to filter out empty arrays (e.g. if a proc has 0 training count)
        valid_results = [r for r in results if len(r) > 0]
        if not valid_results:
             return np.zeros((0, SNAKE_OBS_SIZE), dtype=np.float32)
        return np.concatenate(valid_results)

    def step_async(self, actions: np.ndarray) -> None:
        start_idx = 0
        for remote, count in zip(self.remotes, self.training_counts):
            end_idx = start_idx + count
            remote.send(('step', actions[start_idx:end_idx]))
            start_idx = end_idx
        self.waiting = True

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[Dict[str, Any]]]:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        obs_list, rews_list, dones_list, infos_list = zip(*results)
        
        # Merge lists of dicts
        merged_infos = []
        for info_list in infos_list:
            merged_infos.extend(info_list)
            
        if self.num_envs == 0:
             return np.zeros((0, SNAKE_OBS_SIZE), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool), []

        return np.concatenate([o for o in obs_list if len(o) > 0]), \
               np.concatenate([r for r in rews_list if len(r) > 0]), \
               np.concatenate([d for d in dones_list if len(d) > 0]), \
               merged_infos

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return [None] * self.num_envs

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        pass

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        return [False] * self.num_envs
