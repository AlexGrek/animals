import argparse
import logging
import os
import socket
import struct
import sys
import numpy as np
from stable_baselines3 import PPO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("learner.play")

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

    logger.info("All models loaded successfully.")

    # Start TCP Server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    bytes_expected = (num_snakes * 66 + (num_preys + num_amphibias) * 64) * 4
    floats_expected = num_snakes * 66 + (num_preys + num_amphibias) * 64
    
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
                    data = recvall(conn, bytes_expected)
                    if not data:
                        break
                        
                    unpacked = struct.unpack(f'<{floats_expected}f', data)
                    snake_obs = np.array(unpacked[:num_snakes * 66], dtype=np.float32).reshape(num_snakes, 66)
                    
                    prey_start = num_snakes * 66
                    prey_end = prey_start + num_preys * 64
                    amphibia_end = prey_end + num_amphibias * 64
                    
                    prey_obs = np.array(unpacked[prey_start:prey_end], dtype=np.float32).reshape(num_preys, 64)
                    amphibia_obs = np.array(unpacked[prey_end:amphibia_end], dtype=np.float32).reshape(num_amphibias, 64)
                    
                    actions = []
                    for i in range(num_snakes):
                        a, _ = models[i].predict(snake_obs[i:i+1], deterministic=True)
                        actions.append(int(a[0]))
                    
                    # Predict prey actions
                    for p_idx in range(num_preys):
                        if prey_models[p_idx] is not None:
                            pa, _ = prey_models[p_idx].predict(prey_obs[p_idx:p_idx+1], deterministic=True)
                            prey_action = int(pa[0])
                        else:
                            prey_action = 0 # Stand still
                        actions.append(prey_action)
                        
                    # Predict amphibia actions
                    for a_idx in range(num_amphibias):
                        if amphibia_models[a_idx] is not None:
                            aa, _ = amphibia_models[a_idx].predict(amphibia_obs[a_idx:a_idx+1], deterministic=True)
                            amphibia_action = int(aa[0])
                        else:
                            amphibia_action = 0 # Stand still
                        actions.append(amphibia_action)
                    
                    response = struct.pack(f'<{num_snakes + num_preys + num_amphibias}i', *actions)
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
