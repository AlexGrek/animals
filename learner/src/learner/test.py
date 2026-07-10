import argparse
import logging
import os
import json
import sys
import numpy as np
from stable_baselines3 import PPO
import animals_simulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("learner.test")

def main():
    parser = argparse.ArgumentParser(description="Headless Fast-Forward Test for Snake Simulation")
    parser.add_argument("--model", action="append", type=str, help="Path to SB3 model(s)")
    parser.add_argument("--snakes", type=int, default=2, help="Number of snakes in the simulation")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum number of steps before forced termination")
    parser.add_argument("--output", type=str, default="test_results.json", help="Path to dump JSON results")
    args, unknown = parser.parse_known_args()

    model_paths = args.model
    if not model_paths:
        model_paths = ["models/snake_model"]
        
    num_snakes = args.snakes

    if len(model_paths) == 1:
        model_paths = model_paths * num_snakes
    elif len(model_paths) != num_snakes:
        logger.error(f"Number of models ({len(model_paths)}) must be 1 or equal to number of snakes ({num_snakes}).")
        sys.exit(1)

    models = []
    loaded_models = {}
    for path in model_paths:
        if path not in loaded_models:
            if not os.path.exists(path + ".zip") and not os.path.exists(path):
                logger.error(f"Model not found at {path}.")
                sys.exit(1)
            logger.info(f"Loading model from {path}...")
            loaded_models[path] = PPO.load(path)
        models.append(loaded_models[path])

    logger.info(f"Initializing simulation with {num_snakes} snakes...")
    sim = animals_simulation.Simulation(num_snakes)
    obs_list = sim.reset()

    # Snakes respawn in-place on death rather than ending the whole game, so
    # there is no more "game over" step to stop at: we always run the full
    # max_steps budget. `dones`/rewards are used to accumulate per-life totals
    # (apples eaten, kills, deaths) since get_stats() only reflects the
    # current (post-respawn) life for each snake.
    total_apples = [0] * num_snakes
    total_kills_events = [0] * num_snakes
    total_deaths = [0] * num_snakes

    logger.info("Running simulation loop...")
    steps = 0
    while steps < args.max_steps:
        obs = np.array(obs_list, dtype=np.float32)
        actions = []
        for i in range(num_snakes):
            a, _ = models[i].predict(obs[i:i+1], deterministic=True)
            actions.append(int(a[0]))

        obs_list, rewards, dones, _terminal_obs = sim.step(actions)
        steps += 1

        for i in range(num_snakes):
            if dones[i]:
                total_deaths[i] += 1
            elif rewards[i] >= 50.0:
                total_kills_events[i] += 1
            elif rewards[i] >= 10.0:
                total_apples[i] += 1

    logger.info(f"Simulation ended after {steps} steps.")
    stats = sim.get_stats()

    # Associate models and cumulative (across respawns) counters with stats
    # for clarity, since `stats` itself only reflects each snake's current life.
    for i, stat in enumerate(stats):
        stat["model"] = model_paths[i]
        stat["total_apples_eaten"] = total_apples[i]
        stat["total_kills"] = total_kills_events[i]
        stat["total_deaths"] = total_deaths[i]

    with open(args.output, "w") as f:
        json.dump(stats, f, indent=4)

    logger.info(f"Results dumped to {args.output}")

if __name__ == "__main__":
    main()
