import argparse
import logging
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("learner.train_amphibia")

# Ensure src directory is in sys.path if running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learner.amphibia_environment import RustAmphibiaVecEnv
from learner.policy import GridCnnExtractor
from learner.constants import PREY_GRID1, PREY_GRID2
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser(description="Train Amphibia agent to survive snakes.")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total timesteps to train.")
    parser.add_argument("--num-games", type=int, default=16, help="Number of parallel games.")
    parser.add_argument("--snakes-per-game", type=int, default=2, help="Number of snakes per game instance.")
    parser.add_argument("--amphibias-per-game", type=int, default=1, help="Number of amphibias per game instance.")
    parser.add_argument("--max-amphibias", type=int, default=20, help="Max amphibias per game instance.")
    parser.add_argument("--snake-model", type=str, default="models/snake_model.zip", help="Path to snake model to use as predator.")
    parser.add_argument("--model-path", type=str, default="models/amphibia_model.zip", help="Path to save the amphibia model.")

    args = parser.parse_args()

    from learner.model_utils import resolve_model_path, normalize_model_path
    args.model_path = normalize_model_path(args.model_path)
    if args.snake_model:
        args.snake_model = resolve_model_path(args.snake_model) or normalize_model_path(args.snake_model)

    try:
        logger.info(f"Creating Amphibia Vector Env with {args.num_games} games...")
        env = RustAmphibiaVecEnv(
            num_games=args.num_games,
            snakes_per_game=args.snakes_per_game,
            amphibias_per_game=args.amphibias_per_game,
            max_amphibias=args.max_amphibias,
            snake_model_path=args.snake_model
        )

        logger.info("Initializing PPO agent for PREY with device='cpu'...")
        # The observation holds two 8x8 grids; a small CNN encoder preserves their
        # spatial structure, then a compact MLP head maps to the 5 discrete moves.
        # Amphibia shares the prey observation layout (species differ only via speed).
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(
                features_extractor_class=GridCnnExtractor,
                features_extractor_kwargs=dict(grid1=PREY_GRID1, grid2=PREY_GRID2),
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
            ),
            verbose=1,
            device="cpu",
            batch_size=2048,
            n_steps=512,
            ent_coef=0.02, # Reward now includes threat-distance shaping (dense signal), so less entropy is needed to find escape routes than with pure sparse survival reward.
            gamma=0.995,   # Match the snake's horizon; prey lifespans run 200-500 steps, so survival is a long-horizon objective.
        )
        
        logger.info(f"Starting Amphibia training for {args.steps} steps...")
        model.learn(total_timesteps=args.steps, progress_bar=True)
        
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        model.save(args.model_path)
        logger.info(f"Amphibia training complete. Saving model to {args.model_path}")
        
    except Exception as e:
        logger.error(f"Amphibia training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
