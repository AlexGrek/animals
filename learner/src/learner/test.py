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
    parser.add_argument("--prey-model", action="append", type=str, help="Path to prey SB3 model(s)")
    parser.add_argument("--amphibia-model", action="append", type=str, help="Path to amphibia SB3 model(s)")
    parser.add_argument("--snakes", type=int, default=2, help="Number of snakes in the simulation")
    parser.add_argument("--preys", type=int, default=1, help="Number of preys in the simulation")
    parser.add_argument("--amphibias", type=int, default=0, help="Number of amphibias in the simulation")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum number of steps before forced termination")
    parser.add_argument("--output", type=str, default="test_results.json", help="Path to dump JSON results")
    args, unknown = parser.parse_known_args()

    num_snakes = args.snakes
    num_preys = args.preys
    num_amphibias = args.amphibias

    model_paths = args.model
    if not model_paths:
        model_paths = ["models/snake_model"]

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
            logger.info(f"Loading snake model from {path}...")
            loaded_models[path] = PPO.load(path)
        models.append(loaded_models[path])

    # Handle multiple prey models
    prey_model_paths = args.prey_model
    prey_models = []
    if prey_model_paths:
        if len(prey_model_paths) == 1:
            prey_model_paths = prey_model_paths * num_preys
        elif len(prey_model_paths) != num_preys:
            logger.error(f"Number of prey models ({len(prey_model_paths)}) must be 1 or equal to number of preys ({num_preys}).")
            sys.exit(1)

        loaded_prey_models = {}
        for path in prey_model_paths:
            if path not in loaded_prey_models:
                if not os.path.exists(path + ".zip") and not os.path.exists(path):
                    logger.error(f"Prey model not found at {path}.")
                    sys.exit(1)
                logger.info(f"Loading prey model from {path}...")
                loaded_prey_models[path] = PPO.load(path)
            prey_models.append(loaded_prey_models[path])
    else:
        prey_models = [None] * num_preys
        
    # Handle multiple amphibia models
    amphibia_model_paths = args.amphibia_model
    amphibia_models = []
    if amphibia_model_paths:
        if len(amphibia_model_paths) == 1:
            amphibia_model_paths = amphibia_model_paths * num_amphibias
        elif len(amphibia_model_paths) != num_amphibias:
            logger.error(f"Number of amphibia models ({len(amphibia_model_paths)}) must be 1 or equal to number of amphibias ({num_amphibias}).")
            sys.exit(1)

        loaded_amphibia_models = {}
        for path in amphibia_model_paths:
            if path not in loaded_amphibia_models:
                if not os.path.exists(path + ".zip") and not os.path.exists(path):
                    logger.error(f"Amphibia model not found at {path}.")
                    sys.exit(1)
                logger.info(f"Loading amphibia model from {path}...")
                loaded_amphibia_models[path] = PPO.load(path)
            amphibia_models.append(loaded_amphibia_models[path])
    else:
        amphibia_models = [None] * num_amphibias

    logger.info(f"Initializing simulation with {num_snakes} snakes, {num_preys} preys, and {num_amphibias} amphibias...")
    sim = animals_simulation.Simulation(num_snakes, num_preys, num_amphibias)
    obs_list = sim.reset()

    total_apples = [0] * num_snakes
    total_kills_events = [0] * num_snakes
    total_deaths = [0] * num_snakes
    
    prey_deaths = [0] * num_preys
    prey_ticks_survived = [0] * num_preys
    
    amphibia_deaths = [0] * num_amphibias
    amphibia_ticks_survived = [0] * num_amphibias

    logger.info("Running simulation loop...")
    steps = 0
    while steps < args.max_steps:
        obs = np.array(obs_list, dtype=np.float32)
        actions = []
        for i in range(num_snakes):
            a, _ = models[i].predict(obs[i:i+1], deterministic=True)
            actions.append(int(a[0]))

        prey_actions = []
        prey_obs_list = sim.get_all_prey_observations()
        for p_idx in range(num_preys):
            if prey_models[p_idx] is not None:
                p_obs = np.array(prey_obs_list[p_idx], dtype=np.float32).reshape(1, 64)
                pa, _ = prey_models[p_idx].predict(p_obs, deterministic=True)
                prey_actions.append(int(pa[0]))
            else:
                prey_actions.append(0)

        amphibia_actions = []
        amphibia_obs_list = sim.get_all_amphibia_observations()
        for a_idx in range(num_amphibias):
            if amphibia_models[a_idx] is not None:
                a_obs = np.array(amphibia_obs_list[a_idx], dtype=np.float32).reshape(1, 64)
                aa, _ = amphibia_models[a_idx].predict(a_obs, deterministic=True)
                amphibia_actions.append(int(aa[0]))
            else:
                amphibia_actions.append(0)

        obs_list, rewards, dones, _terminal_obs, _, prey_rewards, prey_dones, _, amphibia_rewards, amphibia_dones = sim.step(actions, prey_actions, amphibia_actions)
        steps += 1

        for p_idx in range(num_preys):
            if prey_dones[p_idx]:
                prey_deaths[p_idx] += 1
            else:
                prey_ticks_survived[p_idx] += 1
                
        for a_idx in range(num_amphibias):
            if amphibia_dones[a_idx]:
                amphibia_deaths[a_idx] += 1
            else:
                amphibia_ticks_survived[a_idx] += 1

        for i in range(num_snakes):
            if dones[i]:
                total_deaths[i] += 1
            elif rewards[i] >= 50.0:
                total_kills_events[i] += 1
            elif rewards[i] >= 10.0:
                total_apples[i] += 1

    logger.info(f"Simulation ended after {steps} steps.")
    stats = sim.get_stats()

    for i, stat in enumerate(stats):
        stat["model"] = model_paths[i]
        stat["total_apples_eaten"] = total_apples[i]
        stat["total_kills"] = total_kills_events[i]
        stat["total_deaths"] = total_deaths[i]

    # Add prey stats to output
    prey_stats = []
    for p_idx in range(num_preys):
        prey_stats.append({
            "model": prey_model_paths[p_idx] if prey_model_paths else "static_prey",
            "total_deaths": prey_deaths[p_idx],
            "ticks_survived": prey_ticks_survived[p_idx]
        })
        
    amphibia_stats = []
    for a_idx in range(num_amphibias):
        amphibia_stats.append({
            "model": amphibia_model_paths[a_idx] if amphibia_model_paths else "static_amphibia",
            "total_deaths": amphibia_deaths[a_idx],
            "ticks_survived": amphibia_ticks_survived[a_idx]
        })

    output_data = {
        "snakes": stats,
        "prey": prey_stats,
        "amphibia": amphibia_stats
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=4)

    logger.info(f"Results dumped to {args.output}")

if __name__ == "__main__":
    main()
