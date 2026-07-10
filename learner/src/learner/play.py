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
    parser.add_argument("--model", action="append", type=str, help="Path to SB3 model(s)")
    parser.add_argument("--snakes", type=int, default=2, help="Number of snakes in the simulation")
    parser.add_argument("--port", type=int, default=31337, help="TCP port to listen on")
    args = parser.parse_args()

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
    # Deduplicate loading to save memory
    loaded_models = {}
    for path in model_paths:
        if path not in loaded_models:
            if not os.path.exists(path + ".zip") and not os.path.exists(path):
                logger.error(f"Model not found at {path}.")
                sys.exit(1)
            logger.info(f"Loading model from {path}...")
            loaded_models[path] = PPO.load(path)
        models.append(loaded_models[path])

    logger.info("All models loaded successfully.")

    # Start TCP Server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    bytes_expected = num_snakes * 66 * 4
    floats_expected = num_snakes * 66
    
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
                        break # Connection closed
                        
                    unpacked = struct.unpack(f'<{floats_expected}f', data)
                    obs = np.array(unpacked, dtype=np.float32).reshape(num_snakes, 66)
                    
                    actions = []
                    for i in range(num_snakes):
                        a, _ = models[i].predict(obs[i:i+1], deterministic=True)
                        actions.append(int(a[0]))
                    
                    response = struct.pack(f'<{num_snakes}i', *actions)
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
