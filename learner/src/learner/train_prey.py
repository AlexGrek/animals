import argparse
import logging
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("learner.train_prey")

# Ensure src directory is in sys.path if running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learner.prey_environment import RustPreyVecEnv
from learner.policy import GridCnnExtractor
from learner.constants import PREY_GRID1, PREY_GRID2, PREY_OBS_SIZE, PREY_NUM_ACTIONS
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser(description="Train Prey agent to survive snakes.")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total timesteps to train.")
    parser.add_argument("--num-games", type=int, default=16, help="Number of parallel games.")
    parser.add_argument("--snakes-per-game", type=int, default=2, help="Number of snakes per game instance.")
    parser.add_argument("--preys-per-game", type=int, default=32, help="Number of preys per game instance.")
    parser.add_argument("--max-preys", type=int, default=100, help="Max preys per game instance.")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of background processes to spawn for environment stepping.")
    parser.add_argument("--snake-model", type=str, default="models/snake_model.zip", help="Path to snake model to use as predator.")
    parser.add_argument("--model-path", type=str, default="models/prey_model.zip", help="Path to save the prey model.")
    parser.add_argument("--resume", action="store_true", help="Resume training from model-path if it exists.")
    parser.add_argument("--existing", action="append", type=str, help="Existing model config in format path:count (per game). e.g. models/prey_v1.zip:10")

    args = parser.parse_args()

    from learner.model_utils import resolve_model_path, normalize_model_path
    args.model_path = normalize_model_path(args.model_path)
    if args.snake_model:
        args.snake_model = resolve_model_path(args.snake_model) or normalize_model_path(args.snake_model)

    try:
        if args.num_games % args.num_procs != 0:
            raise ValueError(f"--num-games ({args.num_games}) must be evenly divisible by --num-procs ({args.num_procs}).")
        games_per_proc = args.num_games // args.num_procs

        # Cap PyTorch's per-process thread pool so the main process and every
        # worker (each of which loads its own copy of the frozen snake model)
        # don't all independently claim every physical core and thrash.
        import torch
        threads = max(1, (os.cpu_count() or 4) // (args.num_procs + 1)) if args.num_procs > 1 else (os.cpu_count() or 4)
        torch.set_num_threads(threads)

        # Parse existing models (counts are per game for prey)
        existing_models = {}
        if args.existing:
            for ex in args.existing:
                parts = ex.split(":")
                if len(parts) != 2:
                    raise ValueError(f"Invalid --existing format: {ex}. Expected path:count")
                path = parts[0]
                count = int(parts[1])
                
                resolved = resolve_model_path(path)
                if resolved is None:
                    raise FileNotFoundError(f"Existing model not found at {path}")
                
                existing_models[resolved] = existing_models.get(resolved, 0) + count

        logger.info(f"Creating Prey Vector Env with {args.num_games} games across {args.num_procs} process(es) ({torch.get_num_threads()} torch threads/process)...")
        
        def make_env_fn(proc_idx):
            def _init():
                return RustPreyVecEnv(
                    num_games=games_per_proc,
                    snakes_per_game=args.snakes_per_game,
                    preys_per_game=args.preys_per_game,
                    max_preys=args.max_preys,
                    existing_models=existing_models,
                    snake_model_path=args.snake_model,
                )
            return _init

        if args.num_procs == 1:
            env = make_env_fn(0)()
        else:
            from learner.environment import MultiProcRustVecEnv
            env_fns = [make_env_fn(i) for i in range(args.num_procs)]
            env = MultiProcRustVecEnv(env_fns, obs_size=PREY_OBS_SIZE, num_actions=PREY_NUM_ACTIONS, threads_per_proc=threads)

        logger.info("Initializing PPO agent for PREY with device='cpu'...")
        if args.resume and os.path.exists(args.model_path):
            logger.info(f"Resuming Prey training from {args.model_path}...")
            custom_objects = {
                "n_steps": 16,
                "batch_size": 2560,
            }
            model = PPO.load(args.model_path, env=env, custom_objects=custom_objects)
        else:
            # The observation holds two 8x8 grids; a small CNN encoder preserves their
            # spatial structure, then a compact MLP head maps to the 5 discrete moves.
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                policy_kwargs=dict(
                    features_extractor_class=GridCnnExtractor,
                    features_extractor_kwargs=dict(grid1=PREY_GRID1, grid2=PREY_GRID2),
                    net_arch=dict(pi=[128, 128], vf=[128, 128]),
                ),
                verbose=1,
                device="cpu",
                batch_size=2560,
                # 16 games * 100 max preys = 1600 envs
                # 1600 * 16 = 25,600 transitions per PPO update
                n_steps=16,
                ent_coef=0.02, # Reward now includes threat-distance shaping (dense signal), so less entropy is needed to find escape routes than with pure sparse survival reward.
                gamma=0.995,   # Match the snake's horizon; prey lifespans run 200-500 steps, so survival is a long-horizon objective.
            )
        
        logger.info(f"Starting Prey training for {args.steps} steps...")
        model.learn(total_timesteps=args.steps, progress_bar=True)
        
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        model.save(args.model_path)
        logger.info(f"Prey training complete. Saving model to {args.model_path}")
        
    except Exception as e:
        logger.error(f"Prey training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
