import argparse
import logging
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("learner.main")

# Ensure src directory is in sys.path if running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learner.environment import SimulationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    """Factory function for creating an environment instance."""
    return SimulationEnv()

def main():
    parser = argparse.ArgumentParser(description="Reinforcement learning agent for animal behavior simulation.")
    parser.add_argument("--steps", type=int, default=100_000, help="Total timesteps to train.")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environment instances.")
    parser.add_argument("--model-path", type=str, default="models/snake_model.zip", help="Path to save the model.")

    args = parser.parse_args()

    try:
        logger.info(f"Creating {args.num_envs} in-process vectorized environments...")
        # DummyVecEnv runs all envs in the training process. The Rust/PyO3 env step
        # is cheap enough that SubprocVecEnv's per-step IPC (pickling + pipes) costs
        # more than it saves from cross-process parallelism.
        env = make_vec_env(make_env, n_envs=args.num_envs, vec_env_cls=DummyVecEnv)

        logger.info("Initializing PPO agent with device='cpu'...")
        # The policy is a tiny MLP; GPU host<->device transfer/launch overhead
        # exceeds the compute it would save, so CPU is faster here.
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")
        
        logger.info(f"Starting training for {args.steps} steps...")
        model.learn(total_timesteps=args.steps, progress_bar=True)
        
        logger.info(f"Training complete. Saving model to {args.model_path}")
        os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
        model.save(args.model_path)
        
        env.close()

    except Exception as e:
        logger.exception(f"Fatal error during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
