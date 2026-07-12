import argparse
import logging
import os
import json
import sys
import numpy as np
from stable_baselines3 import PPO
import animals_simulation

from learner.constants import SNAKE_OBS_SIZE, PREY_OBS_SIZE

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
    parser.add_argument("--deterministic", action="store_true",
                         help="Use argmax actions instead of sampling (matches play.py's default inference mode)")
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

    from learner.model_utils import resolve_model_path

    # Resolve snake model paths
    resolved_model_paths = []
    for path in model_paths:
        resolved = resolve_model_path(path)
        if resolved is None:
            logger.error(f"Model not found at {path}.")
            sys.exit(1)
        resolved_model_paths.append(resolved)
    model_paths = resolved_model_paths

    models = []
    loaded_models = {}
    for path in model_paths:
        if path not in loaded_models:
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

        resolved_prey_paths = []
        for path in prey_model_paths:
            resolved = resolve_model_path(path)
            if resolved is None:
                logger.error(f"Prey model not found at {path}.")
                sys.exit(1)
            resolved_prey_paths.append(resolved)
        prey_model_paths = resolved_prey_paths

        loaded_prey_models = {}
        for path in prey_model_paths:
            if path not in loaded_prey_models:
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

        resolved_amphibia_paths = []
        for path in amphibia_model_paths:
            resolved = resolve_model_path(path)
            if resolved is None:
                logger.error(f"Amphibia model not found at {path}.")
                sys.exit(1)
            resolved_amphibia_paths.append(resolved)
        amphibia_model_paths = resolved_amphibia_paths

        loaded_amphibia_models = {}
        for path in amphibia_model_paths:
            if path not in loaded_amphibia_models:
                logger.info(f"Loading amphibia model from {path}...")
                loaded_amphibia_models[path] = PPO.load(path)
            amphibia_models.append(loaded_amphibia_models[path])
    else:
        amphibia_models = [None] * num_amphibias

    logger.info(f"Initializing simulation with {num_snakes} snakes, {num_preys} preys, and {num_amphibias} amphibias...")
    sim = animals_simulation.Simulation(num_snakes, num_preys, num_preys, num_amphibias, num_amphibias, 0)
    obs_list, _, _, _ = sim.reset()

    total_apples = [0] * num_snakes
    total_kills_events = [0] * num_snakes
    total_deaths = [0] * num_snakes
    
    prey_deaths = [0] * num_preys
    prey_ticks_survived = [0] * num_preys
    
    amphibia_deaths = [0] * num_amphibias
    amphibia_ticks_survived = [0] * num_amphibias

    def batched_predict(models_list, obs_arr, deterministic=args.deterministic):
        """Batch each unique model's predictions in one call instead of one
        forward pass per agent."""
        n = obs_arr.shape[0]
        result = [0] * n
        for model in set(m for m in models_list if m is not None):
            idxs = [i for i in range(n) if models_list[i] is model]
            acts, _ = model.predict(obs_arr[idxs], deterministic=deterministic)
            for i, a in zip(idxs, acts):
                result[i] = int(a)
        return result

    # --- Circling diagnostics -------------------------------------------------
    # Grid is 400x400 everywhere (animals_simulation::Simulation::new / GRID_WIDTH
    # /GRID_HEIGHT in animals_game); torus-wrapped like the engine's torus_delta.
    GRID_WIDTH = GRID_HEIGHT = 400

    def torus_delta(from_xy, to_xy):
        dx = to_xy[0] - from_xy[0]
        dy = to_xy[1] - from_xy[1]
        if dx > GRID_WIDTH / 2: dx -= GRID_WIDTH
        elif dx < -GRID_WIDTH / 2: dx += GRID_WIDTH
        if dy > GRID_HEIGHT / 2: dy -= GRID_HEIGHT
        elif dy < -GRID_HEIGHT / 2: dy += GRID_HEIGHT
        return dx, dy

    action_counts = [[0, 0, 0] for _ in range(num_snakes)]  # straight, right, left
    longest_turn_run = [0] * num_snakes
    cur_turn_run = [0] * num_snakes
    cur_turn_action = [None] * num_snakes
    life_patches = [set() for _ in range(num_snakes)]  # unique 4x4 patches visited this life
    life_patch_counts = [[] for _ in range(num_snakes)]  # completed lives' patch counts
    all_patches = [set() for _ in range(num_snakes)]  # unique patches across the whole run
    displacement_samples = [[] for _ in range(num_snakes)]  # torus displacement over 100-tick windows
    head_at_window_start = [None] * num_snakes

    logger.info("Running simulation loop...")
    steps = 0
    while steps < args.max_steps:
        obs = np.array(obs_list, dtype=np.float32).reshape(num_snakes, SNAKE_OBS_SIZE)
        actions = batched_predict(models, obs)

        prey_obs_arr = np.array(sim.get_all_prey_observations(), dtype=np.float32).reshape(num_preys, PREY_OBS_SIZE)
        prey_actions = batched_predict(prey_models, prey_obs_arr)

        amphibia_obs_arr = np.array(sim.get_all_amphibia_observations(), dtype=np.float32).reshape(num_amphibias, PREY_OBS_SIZE)
        amphibia_actions = batched_predict(amphibia_models, amphibia_obs_arr)

        (obs_list, rewards, dones, _terminal_obs,
         _, prey_rewards, prey_dones,
         _, amphibia_rewards, amphibia_dones,
         _, _) = sim.step(actions, prey_actions, amphibia_actions, [])
        steps += 1

        stats_now = sim.get_stats()
        for i in range(num_snakes):
            action_counts[i][actions[i]] += 1
            if actions[i] == 0:
                cur_turn_run[i] = 0
                cur_turn_action[i] = None
            else:
                if actions[i] == cur_turn_action[i]:
                    cur_turn_run[i] += 1
                else:
                    cur_turn_action[i] = actions[i]
                    cur_turn_run[i] = 1
                longest_turn_run[i] = max(longest_turn_run[i], cur_turn_run[i])

            head = (stats_now[i]["head_x"], stats_now[i]["head_y"])
            if head_at_window_start[i] is None:
                head_at_window_start[i] = head
            if steps % 100 == 0:
                dx, dy = torus_delta(head_at_window_start[i], head)
                displacement_samples[i].append((dx * dx + dy * dy) ** 0.5)
                head_at_window_start[i] = head

            if dones[i]:
                life_patch_counts[i].append(len(life_patches[i]))
                life_patches[i] = set()
            else:
                patch = (head[0] // 4, head[1] // 4)
                life_patches[i].add(patch)
                all_patches[i].add(patch)

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

        # Circling diagnostics (see docs/learning.md "Validation protocol").
        straight, right, left = action_counts[i]
        total_actions = straight + right + left
        turns = right + left
        stat["action_distribution"] = {
            "straight": straight / total_actions if total_actions else 0.0,
            "right": right / total_actions if total_actions else 0.0,
            "left": left / total_actions if total_actions else 0.0,
        }
        stat["turn_bias"] = abs(right - left) / turns if turns else 0.0
        stat["longest_turn_run"] = longest_turn_run[i]
        completed_lives = life_patch_counts[i]
        stat["unique_patches_per_life"] = (sum(completed_lives) / len(completed_lives)) if completed_lives else None
        stat["unique_patches_total"] = len(all_patches[i])
        stat["unique_patches_per_100_ticks"] = len(all_patches[i]) / (steps / 100) if steps else 0.0
        stat["mean_displacement_per_100_ticks"] = (
            sum(displacement_samples[i]) / len(displacement_samples[i]) if displacement_samples[i] else 0.0
        )

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
