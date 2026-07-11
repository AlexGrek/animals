import argparse
import logging
import os
import socket
import struct
import sys
import numpy as np
import torch
from stable_baselines3 import PPO

from learner.constants import SNAKE_OBS_SIZE, PREY_OBS_SIZE
# Ensure the custom feature extractor class is importable when SB3 unpickles a
# checkpoint's policy_kwargs (the class is referenced by name inside the .zip).
import learner.policy  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("learner.play")

# --- Neural-net activation streaming for the Bevy overlay -------------------
# For the ONE snake selected in the visualizer we stream a fixed-length vector
# of per-tick layer activations. Layout (must match `animals_game/src/main.rs`):
#   [128 cnn-feature outputs][256 policy_net Tanh-0][256 policy_net Tanh-1][3 action logits]
# = 643 floats. See CLAUDE.md "cross-language invariants".
_ACT_LAYERS = ("features", "pi0", "pi1", "logits")
_ACT_LEN = 128 + 256 + 256 + 3  # 643


def register_activation_hooks(model, path=""):
    """Attach forward hooks to a snake PPO model, returning a dict that always
    holds the most recent forward pass's activations (flattened per layer).

    Older/other checkpoints may not use the custom `GridCnnExtractor` (e.g. a
    plain SB3 `FlattenExtractor`, or a different `net_arch`) and so won't have
    the exact submodules this schema expects. Rather than crash the whole
    inference server, we skip whichever hooks don't apply to a given model —
    `activation_blob` already treats a missing layer as "no data" (empty blob),
    which the Bevy overlay renders as "waiting for inference" for that snake.
    """
    store = {}

    def make_hook(name):
        def hook(_module, _inp, out):
            store[name] = out.detach().cpu().numpy().reshape(-1)
        return hook

    policy = model.policy

    # `features_extractor.linear` is the GridCnnExtractor's CNN head
    # (Linear(2048->128)+ReLU) => 128. Absent on a stock FlattenExtractor.
    linear = getattr(policy.features_extractor, "linear", None)
    if linear is not None:
        linear.register_forward_hook(make_hook("features"))
    else:
        logger.warning(
            "Snake model '%s' has no GridCnnExtractor.linear; NN overlay "
            "activations for this model will be unavailable.", path,
        )

    # policy_net = [Linear, Tanh, Linear, Tanh]; hook the two Tanh outputs.
    # Guarded in case a checkpoint's net_arch has a different depth.
    policy_net = policy.mlp_extractor.policy_net
    if len(policy_net) > 1:
        policy_net[1].register_forward_hook(make_hook("pi0"))
    if len(policy_net) > 3:
        policy_net[3].register_forward_hook(make_hook("pi1"))
    policy.action_net.register_forward_hook(make_hook("logits"))
    return store


def activation_blob(store):
    """Concatenate the hooked layers into the fixed 643-float vector, or return
    an empty array if any layer hasn't fired yet."""
    parts = []
    for k in _ACT_LAYERS:
        v = store.get(k)
        if v is None:
            return np.empty(0, dtype=np.float32)
        parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
    blob = np.concatenate(parts).astype(np.float32)
    if blob.shape[0] != _ACT_LEN:
        logger.warning("Activation blob size %d != expected %d", blob.shape[0], _ACT_LEN)
    return blob

def main():
    parser = argparse.ArgumentParser(description="TCP Inference Server for Bevy Game")
    parser.add_argument("--model", action="append", type=str, help="Path to SB3 model(s) for snakes")
    parser.add_argument("--prey-model", action="append", type=str, help="Path to SB3 model(s) for preys")
    parser.add_argument("--amphibia-model", action="append", type=str, help="Path to SB3 model(s) for amphibias")
    parser.add_argument("--snakes", type=int, default=2, help="Number of snakes in the simulation")
    parser.add_argument("--preys", type=int, default=1, help="Number of preys in the simulation")
    parser.add_argument("--amphibias", type=int, default=0, help="Number of amphibias in the simulation")
    parser.add_argument("--port", type=int, default=31337, help="TCP port to listen on")
    args = parser.parse_args()

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
    snake_stores = {}  # path -> activation store (for the NN overlay)
    for path in model_paths:
        if path not in loaded_models:
            logger.info(f"Loading snake model from {path}...")
            loaded_models[path] = PPO.load(path)
            snake_stores[path] = register_activation_hooks(loaded_models[path], path)
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

    logger.info("All models loaded successfully.")

    # Start TCP Server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind(("127.0.0.1", args.port))
        server.listen(1)
        logger.info(f"Listening for Bevy connection on 127.0.0.1:{args.port}...")

        def recvall(sock, n):
            data = bytearray()
            while len(data) < n:
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            return data

        while True:
            conn, addr = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            logger.info(f"Bevy connected from {addr}!")

            try:
                while True:
                    header = recvall(conn, 16)
                    if not header:
                        break
                    num_snakes, num_preys, num_amphibias, selected = struct.unpack('<4i', header)

                    bytes_expected = (num_snakes * SNAKE_OBS_SIZE + (num_preys + num_amphibias) * PREY_OBS_SIZE) * 4
                    floats_expected = num_snakes * SNAKE_OBS_SIZE + (num_preys + num_amphibias) * PREY_OBS_SIZE

                    data = recvall(conn, bytes_expected)
                    if not data:
                        break

                    unpacked = struct.unpack(f'<{floats_expected}f', data)
                    snake_obs = np.array(unpacked[:num_snakes * SNAKE_OBS_SIZE], dtype=np.float32).reshape(num_snakes, SNAKE_OBS_SIZE)

                    prey_start = num_snakes * SNAKE_OBS_SIZE
                    prey_end = prey_start + num_preys * PREY_OBS_SIZE
                    amphibia_end = prey_end + num_amphibias * PREY_OBS_SIZE

                    prey_obs = np.array(unpacked[prey_start:prey_end], dtype=np.float32).reshape(num_preys, PREY_OBS_SIZE)
                    amphibia_obs = np.array(unpacked[prey_end:amphibia_end], dtype=np.float32).reshape(num_amphibias, PREY_OBS_SIZE)

                    # Batch each unique loaded model's predictions in one call
                    # instead of one forward pass per agent.
                    snake_action_map = {}
                    for path in loaded_models:
                        idxs = [i for i in range(num_snakes) if model_paths[i % len(model_paths)] == path]
                        if idxs:
                            acts, _ = loaded_models[path].predict(snake_obs[idxs], deterministic=True)
                            for i, a in zip(idxs, acts):
                                snake_action_map[i] = int(a)
                    actions = [snake_action_map[i] for i in range(num_snakes)]

                    prey_action_map = {i: 0 for i in range(num_preys)}
                    if prey_model_paths:
                        for path in set(prey_model_paths):
                            idxs = [i for i in range(num_preys) if prey_model_paths[i % len(prey_model_paths)] == path]
                            if idxs:
                                acts, _ = loaded_prey_models[path].predict(prey_obs[idxs], deterministic=True)
                                for i, a in zip(idxs, acts):
                                    prey_action_map[i] = int(a)
                    actions.extend(prey_action_map[i] for i in range(num_preys))

                    amphibia_action_map = {i: 0 for i in range(num_amphibias)}
                    if amphibia_model_paths:
                        for path in set(amphibia_model_paths):
                            idxs = [i for i in range(num_amphibias) if amphibia_model_paths[i % len(amphibia_model_paths)] == path]
                            if idxs:
                                acts, _ = loaded_amphibia_models[path].predict(amphibia_obs[idxs], deterministic=True)
                                for i, a in zip(idxs, acts):
                                    amphibia_action_map[i] = int(a)
                    actions.extend(amphibia_action_map[i] for i in range(num_amphibias))

                    # For the selected snake only, run one extra single-row
                    # forward so the hooks capture just that snake's activations,
                    # and append them length-prefixed to the response.
                    sel_blob = np.empty(0, dtype=np.float32)
                    if 0 <= selected < num_snakes:
                        sel_path = model_paths[selected]
                        sel_model = loaded_models[sel_path]
                        sel_model.predict(snake_obs[selected:selected + 1], deterministic=True)
                        sel_blob = activation_blob(snake_stores[sel_path])

                    response = struct.pack(f'<{num_snakes + num_preys + num_amphibias}i', *actions)
                    response += struct.pack('<i', int(sel_blob.shape[0]))
                    if sel_blob.shape[0]:
                        response += struct.pack(f'<{sel_blob.shape[0]}f', *sel_blob)
                    conn.sendall(response)

            except ConnectionResetError:
                pass
            finally:
                conn.close()
                logger.info("Bevy disconnected. Waiting for new connection...")

    except KeyboardInterrupt:
        logger.info("Shutting down inference server.")
    finally:
        server.close()

if __name__ == "__main__":
    main()
