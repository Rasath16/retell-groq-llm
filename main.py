import os
import logging

from dotenv import load_dotenv

# Load environment variables
env_file = ".env.development" if os.getenv("NODE_ENV") == "development" else ".env"
load_dotenv(env_file)

import uvicorn
from server import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"Server running on port {port}")
    print(f"WebSocket endpoint: ws://localhost:{port}/llm-websocket/{{call_id}}")
    uvicorn.run(app, host="0.0.0.0", port=port)
