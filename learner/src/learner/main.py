import argparse
import logging
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("learner.main")

# Ensure src directory is in sys.path if running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learner.environment import RustMultiSnakeVecEnv
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser(description="Reinforcement learning agent for animal behavior simulation.")
    parser.add_argument("--steps", type=int, default=100_000, help="Total timesteps to train.")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environment instances.")
    parser.add_argument("--model-path", type=str, default="models/snake_model.zip", help="Path to save the model.")

    args = parser.parse_args()

    try:
        # Note: num_envs here represents number of games. 
        # Total SB3 environments will be num_envs * 2.
        logger.info(f"Creating MARL Vector Env with {args.num_envs} games ({args.num_envs * 2} parallel snakes)...")
        env = RustMultiSnakeVecEnv(num_games=args.num_envs)

        logger.info("Initializing PPO agent with device='cpu'...")
        # The policy is a tiny MLP; GPU host<->device transfer/launch overhead
        # exceeds the compute it would save, so CPU is faster here.
        model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256, 256]), verbose=1, device="cpu")
        
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
