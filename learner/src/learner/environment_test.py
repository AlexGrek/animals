import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from typing import Tuple, Dict, Any, Optional, List, Callable
import random
import multiprocessing as mp

try:
    import animals_simulation
except ImportError:
    animals_simulation = None

from stable_baselines3 import PPO

class RustMultiSnakeVecEnv(VecEnv):
    def __init__(self, num_games: int = 4, training_count: Optional[int] = None, existing_models: Optional[Dict[str, int]] = None):
        self.num_games = num_games
        self.total_snakes = num_games * 2
        
        if training_count is None:
            training_count = self.total_snakes
        if existing_models is None:
            existing_models = {}
            
        self.training_count = training_count
        self.existing_models = {}
        
        # Load the models!
        for path, count in existing_models.items():
            if count > 0:
                self.existing_models[path] = PPO.load(path)
            
        assignments = ['train'] * training_count
        for path, count in existing_models.items():
            assignments.extend([path] * count)
            
        if len(assignments) != self.total_snakes:
            raise ValueError(f"Total snakes ({self.total_snakes}) does not match sum of training ({training_count}) and existing models.")
            
        random.shuffle(assignments)
        
        self.training_indices = [i for i, a in enumerate(assignments) if a == 'train']
        self.model_indices = {path: [i for i, a in enumerate(assignments) if a == path] for path in self.existing_models.keys()}
        
        action_space = spaces.Discrete(3)
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(66,), dtype=np.float32)
        
        super().__init__(self.training_count, observation_space, action_space)

        if animals_simulation is None:
            raise ImportError("Could not import 'animals_simulation'.")

        self.games = [animals_simulation.Simulation() for _ in range(num_games)]
        
        self.all_obs = np.zeros((self.total_snakes, 66), dtype=np.float32)
        self.all_rews = np.zeros((self.total_snakes,), dtype=np.float32)
        self.all_dones = np.zeros((self.total_snakes,), dtype=bool)
        self.all_infos = [{} for _ in range(self.total_snakes)]
        
        self.actions = np.zeros((self.training_count,), dtype=int)

    def reset(self) -> np.ndarray:
        for i, game in enumerate(self.games):
            obs0, obs1 = game.reset()
            self.all_obs[i * 2] = obs0
            self.all_obs[i * 2 + 1] = obs1
        return np.copy(self.all_obs[self.training_indices])

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[Dict[str, Any]]]:
        all_actions = np.zeros(self.total_snakes, dtype=int)
        
        for idx, train_idx in enumerate(self.training_indices):
            all_actions[train_idx] = self.actions[idx]
            
        for path, model in self.existing_models.items():
            indices = self.model_indices[path]
            if len(indices) > 0:
                obs_for_model = self.all_obs[indices]
                actions, _ = model.predict(obs_for_model, deterministic=False)
                for i, idx in enumerate(indices):
                    all_actions[idx] = actions[i]

        for i, game in enumerate(self.games):
            a0 = int(all_actions[i * 2])
            a1 = int(all_actions[i * 2 + 1])
            
            (o0, o1), (r0, r1), (d0, d1) = game.step(a0, a1)
            game_done = bool(d0 or d1)
            
            self.all_rews[i * 2] = r0
            self.all_rews[i * 2 + 1] = r1
            self.all_dones[i * 2] = game_done
            self.all_dones[i * 2 + 1] = game_done
            self.all_infos[i * 2] = {}
            self.all_infos[i * 2 + 1] = {}
            
            if game_done:
                self.all_infos[i * 2]["terminal_observation"] = np.array(o0, dtype=np.float32)
                self.all_infos[i * 2 + 1]["terminal_observation"] = np.array(o1, dtype=np.float32)
                o0, o1 = game.reset()
                
            self.all_obs[i * 2] = o0
            self.all_obs[i * 2 + 1] = o1

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
            
        # We need to compute total num_envs across all procs.
        # But we can't easily query the workers. Let's just pass it or create a dummy to inspect space.
        dummy = env_fns[0]()
        observation_space = dummy.observation_space
        action_space = dummy.action_space
        # Actually we need total training_count. Let's get it from the fns.
        # But fns are callables. Let's assume the caller passes the training counts or we do a quick check.
        # We can just sum them.
        dummy.close()
        
        self.training_counts = [] # need to know to split actions
        # Let's send a custom command or just rely on the user passing it.
        # Better: let's pass a `training_counts` list to __init__.
