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
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser(description="Train Amphibia agent to survive snakes.")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total timesteps to train.")
    parser.add_argument("--num-games", type=int, default=16, help="Number of parallel games.")
    parser.add_argument("--snakes-per-game", type=int, default=2, help="Number of snakes per game instance.")
    parser.add_argument("--amphibias-per-game", type=int, default=1, help="Number of amphibias per game instance.")
    parser.add_argument("--snake-model", type=str, default="models/snake_model.zip", help="Path to snake model to use as predator.")
    parser.add_argument("--model-path", type=str, default="models/amphibia_model.zip", help="Path to save the amphibia model.")

    args = parser.parse_args()

    try:
        logger.info(f"Creating Amphibia Vector Env with {args.num_games} games...")
        env = RustAmphibiaVecEnv(
            num_games=args.num_games,
            snakes_per_game=args.snakes_per_game,
            amphibias_per_game=args.amphibias_per_game,
            snake_model_path=args.snake_model
        )

        logger.info("Initializing PPO agent for PREY with device='cpu'...")
        # Amphibia observation is small (64 floats), smaller MLP is faster and sufficient
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            verbose=1,
            device="cpu",
            batch_size=2048,
            n_steps=512,
            ent_coef=0.05, # Higher entropy coefficient to encourage amphibia to explore escaping routes
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
