import argparse
import logging
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("learner.main")

# Ensure src directory is in sys.path if running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learner.environment import RustMultiSnakeVecEnv, MultiProcRustVecEnv
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser(description="Reinforcement learning agent for animal behavior simulation.")
    parser.add_argument("--steps", type=int, default=100_000, help="Total timesteps to train.")
    parser.add_argument("--num-games", type=int, default=8, help="Number of parallel games.")
    parser.add_argument("--snakes-per-game", type=int, default=2, help="Number of snakes per game instance.")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of background processes to spawn for environment stepping.")
    parser.add_argument("--model-path", type=str, default="models/snake_model.zip", help="Path to save the model.")
    parser.add_argument("--existing", action="append", type=str, help="Existing model config in format path:count, e.g. models/v1.zip:4")

    args = parser.parse_args()

    try:
        if args.num_games % args.num_procs != 0:
            raise ValueError(f"--num-games ({args.num_games}) must be evenly divisible by --num-procs ({args.num_procs}).")
            
        games_per_proc = args.num_games // args.num_procs
        
        # Parse existing models
        total_existing_counts = {}
        total_existing_snakes = 0
        
        if args.existing:
            for ex in args.existing:
                parts = ex.split(":")
                if len(parts) != 2:
                    raise ValueError(f"Invalid --existing format: {ex}. Expected path:count")
                path = parts[0]
                count = int(parts[1])
                
                if not os.path.exists(path) and not os.path.exists(path + ".zip"):
                    raise FileNotFoundError(f"Existing model not found at {path}")
                
                total_existing_counts[path] = total_existing_counts.get(path, 0) + count
                total_existing_snakes += count
                
        total_snakes = args.num_games * args.snakes_per_game
        training_count = total_snakes - total_existing_snakes
        
        if training_count < 1:
            raise ValueError(f"Total snakes ({total_snakes}) must be greater than total existing snakes ({total_existing_snakes}) to leave room for training.")

        logger.info(f"Creating MARL Vector Env with {args.num_games} games ({total_snakes} total snakes) across {args.num_procs} processes...")
        logger.info(f"  - Training snakes: {training_count}")
        for path, count in total_existing_counts.items():
            logger.info(f"  - Existing model '{path}' snakes: {count}")
            
        # Distribute existing snakes evenly
        existing_models_per_proc = [{} for _ in range(args.num_procs)]
        for path, count in total_existing_counts.items():
            for i in range(count):
                proc_idx = i % args.num_procs
                existing_models_per_proc[proc_idx][path] = existing_models_per_proc[proc_idx].get(path, 0) + 1
                
        def make_env_fn(proc_idx):
            def _init():
                ex_models = existing_models_per_proc[proc_idx]
                snakes_in_proc = games_per_proc * args.snakes_per_game
                ex_count = sum(ex_models.values())
                tr_count = snakes_in_proc - ex_count
                return RustMultiSnakeVecEnv(
                    num_games=games_per_proc,
                    snakes_per_game=args.snakes_per_game,
                    training_count=tr_count,
                    existing_models=ex_models
                )
            return _init

        if args.num_procs == 1:
            env = make_env_fn(0)()
        else:
            env_fns = [make_env_fn(i) for i in range(args.num_procs)]
            env = MultiProcRustVecEnv(env_fns)

        logger.info("Initializing PPO agent with device='cpu'...")
        # The policy is a tiny MLP; GPU host<->device transfer/launch overhead
        # exceeds the compute it would save, so CPU is faster here.
        # batch_size/n_steps tuned for CPU throughput: SB3 defaults (batch_size=64,
        # n_steps=2048) create 512 minibatches * 10 epochs = 5120 tiny optimizer
        # steps per update, which dominates wall-clock on CPU. Larger batches cut
        # that overhead drastically (measured ~4.5x speedup); smaller n_steps gives
        # more frequent policy updates for the same total sample count.
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
            verbose=1,
            device="cpu",
            batch_size=4096,
            n_steps=512,
            ent_coef=0.01,
        )
        
        logger.info(f"Starting training for {args.steps} steps...")
        model.learn(total_timesteps=args.steps, progress_bar=True)
        
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        model.save(args.model_path)
        logger.info(f"Training complete. Saving model to {args.model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
