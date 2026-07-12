import argparse
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from learner.corpsefag_environment import RustCorpsefagVecEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train a multi-agent PPO model for the Corpsefags.")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total timesteps to train.")
    parser.add_argument("--out", type=str, default="models/corpsefag_model", help="Path to save the trained model.")
    parser.add_argument("--resume", type=str, help="Path to an existing model to resume training from.")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of background processes to spawn for environment stepping.")
    args = parser.parse_args()

    # Fixed hyperparameters
    num_games_per_proc = 16
    snakes_per_game = 4
    num_corpsefags = 10

    if args.num_procs > 1:
        def make_env():
            return RustCorpsefagVecEnv(
                num_games=num_games_per_proc,
                snakes_per_game=snakes_per_game,
                num_corpsefags=num_corpsefags
            )
        env = SubprocVecEnv([make_env for _ in range(args.num_procs)])
    else:
        env = RustCorpsefagVecEnv(
            num_games=num_games_per_proc,
            snakes_per_game=snakes_per_game,
            num_corpsefags=num_corpsefags
        )

    if args.resume:
        logger.info(f"Loading existing model from {args.resume}...")
        model = PPO.load(args.resume, env=env)
    else:
        logger.info("Initializing new PPO model for Corpsefags (MlpPolicy)...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cpu",
            learning_rate=3e-4,
            n_steps=512,
            batch_size=1024,
            n_epochs=4,
            gamma=0.99,
            ent_coef=0.01
        )

    try:
        logger.info(f"Starting Corpsefag training for {args.steps} steps...")
        model.learn(total_timesteps=args.steps, progress_bar=True)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current model state...")
    finally:
        logger.info(f"Saving model to {args.out}.zip")
        model.save(args.out)


if __name__ == "__main__":
    main()
