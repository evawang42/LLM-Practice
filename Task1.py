"""
Streaming chat with Ollama (async)
This module exposes an async generator `chat()` that streams model output chunk-by-chunk from a running Ollama server.
"""

from typing import AsyncGenerator
import asyncio
from ollama import AsyncClient
from config import OLLAMA_URL

async def chat(input: dict[str, str]) -> AsyncGenerator[str, None]:
    """
    Stream model chunks from Ollama as they arrive.

    Args:
        input (dict[str, str]): A small dictionary carrying the user prompt.

    Yields:
        str: Incremental text pieces (chunks) produced by the model.
    """
    # Build the single user message from the provided `input` dictionary.
    message = {'role': 'user', 'content': input.get('input', '')}

    # Call Ollama's async chat API with streaming enabled.
    async for part in await AsyncClient(host=OLLAMA_URL).chat(model='llama3', messages=[message], stream=True):
        yield part['message']['content']

async def main():
    async for chunk in chat({"input": "Who are you?"}):
        print(chunk, end='', flush=True)

if __name__ == "__main__":
    asyncio.run(main())