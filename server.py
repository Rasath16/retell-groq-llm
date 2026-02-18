import json
import os
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from retell import Retell

from llm_groq import GroqLlmClient

logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager
from groq import AsyncGroq

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Groq client on startup
    app.state.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    yield
    # Cleanup on shutdown
    await app.state.groq_client.close()

app = FastAPI(lifespan=lifespan)

retell_client = None
if os.getenv("RETELL_API_KEY"):
    retell_client = Retell(api_key=os.getenv("RETELL_API_KEY"))


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Retell Groq Custom LLM Server"}


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()

    # Verify webhook signature if Retell API key is configured
    if retell_client and request.headers.get("x-retell-signature"):
        valid = Retell.verify(
            json.dumps(body),
            os.getenv("RETELL_API_KEY"),
            request.headers["x-retell-signature"],
        )
        if not valid:
            logger.error("Invalid webhook signature")
            return JSONResponse(
                status_code=401, content={"error": "Invalid signature"}
            )

    event = body.get("event")
    call_id = body.get("data", {}).get("call_id")

    if event == "call_started":
        logger.info("Call started: %s", call_id)
    elif event == "call_ended":
        logger.info("Call ended: %s", call_id)
    elif event == "call_analyzed":
        logger.info("Call analyzed: %s", call_id)
    else:
        logger.info("Unknown webhook event: %s", event)

    return {"received": True}


@app.websocket("/llm-websocket/{call_id}")
async def websocket_endpoint(ws: WebSocket, call_id: str):
    await ws.accept()
    logger.info("WebSocket connected for call: %s", call_id)

    # Use the shared Groq client from app state
    llm_client = GroqLlmClient(client=app.state.groq_client)

    # Send config event to Retell
    config_event = {
        "response_type": "config",
        "config": {
            "auto_reconnect": True,
            "call_details": True,
        },
    }
    await ws.send_json(config_event)

    try:
        while True:
            data = await ws.receive_text()
            request = json.loads(data)
            interaction_type = request.get("interaction_type")

            if interaction_type == "call_details":
                logger.info("Call details received for %s", call_id)
                # Agent speaks first — send the opening message
                await llm_client.begin_message(ws)

            elif interaction_type == "ping_pong":
                pong = {
                    "response_type": "ping_pong",
                    "timestamp": request.get("timestamp"),
                }
                await ws.send_json(pong)

            elif interaction_type == "update_only":
                # Transcript update, no response needed
                pass

            elif interaction_type in ("response_required", "reminder_required"):
                # Generate and stream LLM response
                await llm_client.draft_response(request, ws)

            else:
                logger.warning("Unknown interaction type: %s", interaction_type)

    except WebSocketDisconnect as e:
        logger.info("WebSocket closed for call %s: %s %s", call_id, e.code, e.reason)
    except Exception as e:
        logger.error("WebSocket error for call %s: %s", call_id, e)
