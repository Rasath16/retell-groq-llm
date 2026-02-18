import json
import os
import logging

from groq import AsyncGroq
from fastapi import WebSocket

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Role: Megan, Go Green Solar. Human-like, warm, upbeat, not pushy.
Style: Natural, spontaneous. Variable backchannels ("mhm", "yeah"). Light fillers ("um", "well"). Max 1-3 sentences. No lists/formatting. Spoken numbers ($80 -> eighty dollars).
Goal: Qualify Marcus for solar. Protocol:
1. Intro: "Hey there, Am I speaking with Marcus?" -> Go Green Solar intro, tax credits.
2. Zip: Ask/Check.
3. Homeowner: Must own.
4. Property: Type?
5. Bill: >$80?
6. Credit: >640?
7. Qualify: Celebrate! Expert call in 48hrs.
8. Close.
Edge Cases:
- Source: "Data suggests eligibility."
- Scam?: "Just checking eligibility."
- Cost?: "Specialist provides quote."
- Already has solar: "Happy with it?" -> Exit.
- Not interested: Polite exit.
- Angry: De-escalate/Exit.
Constraint: Be concise. Lower latency is priority."""


class GroqLlmClient:
    def __init__(self, client: AsyncGroq):
        self.client = client

    async def begin_message(self, ws: WebSocket):
        """Send the first message to start the conversation (agent speaks first)."""
        begin_message = {
            "response_type": "response",
            "response_id": 0,
            "content": "Hey there, am I speaking with Marcus?",
            "content_complete": True,
            "end_call": False,
        }
        await ws.send_json(begin_message)

    def _conversation_to_chat_messages(self, transcript: list[dict]) -> list[dict]:
        """Convert Retell transcript to Groq/OpenAI-compatible messages."""
        messages = []
        for utterance in transcript:
            messages.append({
                "role": "assistant" if utterance["role"] == "agent" else "user",
                "content": utterance["content"],
            })
        return messages

    def _prepare_prompt(self, request: dict) -> list[dict]:
        """Build the full prompt for Groq."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add transcript history
        transcript = request.get("transcript")
        if transcript:
            # Optimize: Keep only the last 10 turns to reduce input tokens and latency
            # while maintaining recent context.
            chat_messages = self._conversation_to_chat_messages(transcript)
            messages.extend(chat_messages[-10:])

        # If this is a reminder, nudge the agent to re-engage
        if request.get("interaction_type") == "reminder_required":
            messages.append({
                "role": "user",
                "content": "(The caller has been silent for a moment. Gently check in or re-engage them.)",
            })

        return messages

    async def draft_response(self, request: dict, ws: WebSocket):
        """Stream response from Groq back to Retell."""
        messages = self._prepare_prompt(request)
        response_id = request["response_id"]

        try:
            stream = await self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                stream=True,
                temperature=0.2,
                max_tokens=100,
                top_p=0.9,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta or not delta.content:
                    continue

                # Stream each chunk back to Retell
                response = {
                    "response_type": "response",
                    "response_id": response_id,
                    "content": delta.content,
                    "content_complete": False,
                    "end_call": False,
                }
                await ws.send_json(response)

            # Send final message to signal completion
            final_response = {
                "response_type": "response",
                "response_id": response_id,
                "content": "",
                "content_complete": True,
                "end_call": False,
            }
            await ws.send_json(final_response)

        except Exception as e:
            logger.error("Error streaming from Groq: %s", e, exc_info=True)
            # We don't send an error response to the user to avoid breaking character,
            # but we log the full traceback for debugging.

