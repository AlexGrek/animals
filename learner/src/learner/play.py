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
    parser.add_argument("--model", type=str, default="models/snake_model", help="Path to SB3 model")
    parser.add_argument("--port", type=int, default=31337, help="TCP port to listen on")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Train it first using 'task train'!")
        sys.exit(1)

    logger.info(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    logger.info("Model loaded successfully.")

    # Start TCP Server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind(("127.0.0.1", args.port))
        server.listen(1)
        logger.info(f"Listening for Bevy connection on 127.0.0.1:{args.port}...")
        
        while True:
            conn, addr = server.accept()
            logger.info(f"Bevy connected from {addr}!")
            
            try:
                while True:
                    # Expect exactly 32 bytes (8 floats, little endian)
                    data = conn.recv(32)
                    if not data:
                        break # Connection closed
                        
                    if len(data) != 32:
                        logger.warning(f"Received incomplete packet ({len(data)} bytes). Closing connection.")
                        break

                    # Unpack 8 floats
                    unpacked = struct.unpack('<8f', data)
                    obs = np.array(unpacked, dtype=np.float32)
                    
                    # Predict action
                    action, _ = model.predict(obs, deterministic=True)
                    action_int = int(action)
                    
                    # Pack 1 integer (4 bytes, little endian)
                    response = struct.pack('<i', action_int)
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
