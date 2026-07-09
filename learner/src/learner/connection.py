import asyncio
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("learner.connection")

class UDSConnection:
    """Handles Unix Domain Socket communication with the Bevy game engine."""
    
    def __init__(self, socket_path: str = "/tmp/animals_sim.sock"):
        self.socket_path = socket_path
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None

    async def connect(self, max_retries: int = 5, delay: float = 1.0) -> bool:
        """Establishes connection to the Unix Domain Socket server with retries."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to UDS at {self.socket_path} (Attempt {attempt + 1}/{max_retries})...")
                self.reader, self.writer = await asyncio.open_unix_connection(self.socket_path)
                logger.info("Connected successfully.")
                return True
            except Exception as e:
                logger.warning(f"Failed to connect to UDS at {self.socket_path}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    
        logger.error(f"Could not connect to {self.socket_path} after {max_retries} attempts.")
        self.reader = None
        self.writer = None
        return False

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Sends a JSON-serialized message frame followed by a newline."""
        if not self.writer:
            logger.error("Cannot send message: not connected.")
            return False
        try:
            data = json.dumps(message) + "\n"
            self.writer.write(data.encode('utf-8'))
            await self.writer.drain()
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            await self.close()
            return False

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receives a single JSON-serialized message frame terminated by a newline."""
        if not self.reader:
            logger.error("Cannot receive message: not connected.")
            return None
        try:
            line = await self.reader.readline()
            if not line:
                logger.warning("Connection closed by server.")
                await self.close()
                return None
            return json.loads(line.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            await self.close()
            return None

    async def close(self):
        """Closes the connection resources."""
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
            self.writer = None
        self.reader = None
        logger.info("UDS connection closed.")
