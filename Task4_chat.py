"""
Streaming chat with Ollama
"""

from typing import AsyncGenerator, Tuple, Literal, List
from ollama import AsyncClient
from config import OLLAMA_URL, OLLAMA_MODEL

IPromptMessage = Tuple[Literal['system', 'user', 'assistant', 'tool'], str]
"""The type of a prompt message."""

async def chat(messages: list[IPromptMessage], input_data: dict[str, str]) -> AsyncGenerator[str, None]:
    """
    Stream model output from Ollama for a rendered prompt.

    Args:
        messages: A list of `(role, content)` prompt messages. `content` may
            include placeholders (e.g., `{question}`, `{context}`) that will be
            substituted using `input_data`.
        input_data: Variables used to render the message templates via `content.format(**input_data)`. 
            Example: `{'question': '...', 'context': '...'}`.

    Yields:
        str: Incremental text chunks produced by the model.
    """
    rendered = [{'role': role, 'content': content.format(**input_data)} for role, content in messages]
    async for part in await AsyncClient(host=OLLAMA_URL).chat(model=OLLAMA_MODEL, messages=rendered, stream=True):
        yield part['message']['content']

async def online_help_desk_chat(question: str, history: List[Tuple[str, str]]) -> AsyncGenerator[str, None]:
    """
    Chat interface for an online help desk.
    Includes prior turns (history) and a predefined system prompt, then delegates to `chat()`.

    Args:
        question: The current user question to be answered by the model.
        history:  Prior dialogue turns as a list of `(role, content)`.

    Yields:
        str: Incremental text chunks produced by the model.
    """
    system_prompt: IPromptMessage = (
        "system",
        "You are a helpful assistant. Answer in Traditional Chinese (Taiwanese usage). "
        "If you don't know the answer, say '不知道'."
    )

    # messages：[system] + prior history + the current user question
    messages = [system_prompt]
    messages.extend([(role, content) for role, content in history])
    messages.append(("user", "{question}"))

    async for chunk in chat(messages, {'question': question}):
        yield chunk