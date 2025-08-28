"""
Aiohttp server that streams LLM responses to the browser using Server-Sent Events (SSE).
"""

import json
from typing import Any
from aiohttp import web
from aiohttp_cors import setup as cors_setup, ResourceOptions
from Task4_chat import online_help_desk_chat

async def sse_event(response: web.StreamResponse, event: str, data: Any) -> None:
    """
    Send a single SSE event to the client.

    Args:
        response: The prepared StreamResponse to write to.
        event: The SSE "event" name (e.g., "data", "error", "end").
        data: The JSON-serializable payload for the "data:" line.
    """
    sse_format = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    await response.write(sse_format.encode("utf-8"))

async def chat_handler(request: web.Request) -> web.StreamResponse:
    """
    POST/chat
    Accepts JSON with `query` and `history`, then streams LLM output as SSE.

    Returns:
        StreamResponse producing SSE frames until completion.
    """
    try:
        # Parse and validate JSON input
        body = await request.json()
        question = body.get("query", "")
        history = body.get("history", [])

        # Convert frontend history (list[dict]) -> list[tuple] required by `online_help_desk_chat()`
        formatted_history = [(item["role"], item["content"]) for item in history]

        # Prepare SSE stream response
        response = web.StreamResponse(status=200, reason="OK", headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        })

        await response.prepare(request)

        try:
            # Stream chunks from the LLM and forward them as SSE `data` events
            async for chunk in online_help_desk_chat(question, formatted_history):
                await sse_event(response, "data", {"action": "response", "message": chunk})
        except Exception as e:
            # Notify the client with an SSE `error` event
            await sse_event(response, "error", {"action": "error", "message": str(e)})

        # Send an `end` event
        await sse_event(response, "end", {})
        await response.write_eof()
        return response

    except Exception as e:
        # Fail before `prepare()`, return a normal JSON error (not SSE)
        return web.json_response({"error": str(e)}, status=500)

# App Setup
app = web.Application()
app.router.add_post("/chat", chat_handler)

# Configure CORS for all routes and origins using aiohttp_cors
cors = cors_setup(app, defaults={
    "*": ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*" )
})
for route in list(app.router.routes()):
    cors.add(route)

if __name__ == "__main__":
    web.run_app(app, host='0.0.0.0', port=8002)
