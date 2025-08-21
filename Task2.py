"""
Streaming Q&A over local documents with Ollama
This script demonstrates a minimal *retrieval-free* Q&A workflow where the model answers questions strictly based on the provided document text.
"""

from typing import Literal, AsyncGenerator, Tuple
import asyncio
from ollama import AsyncClient
from config import OLLAMA_URL

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
    async for part in await AsyncClient(host=OLLAMA_URL).chat(model='llama3', messages=rendered, stream=True):
        yield part['message']['content']

async def q_a(question: str, document: str) -> AsyncGenerator[str, None]:
    """
    Build a strict Q&A prompt and stream the answer.

    Args:
        question: The user's question to be answered from `document`.
        document: The plain-text content used as the *only* source of truth.

    Yields:
        str: Incremental text chunks produced by the model.
    """
    # Prompt
    qa_prompt: list[IPromptMessage] = [
        ('system', 'Answer strictly using ONLY the provided content. '
                   'If the answer is not present, reply with "不知道". '
                   'Respond in Traditional Chinese (zh-Hant).'),
        ('user',   'Content:\n{context}\n---\nQuestion: {question}\n'
                   'Answer strictly using the content above.')
    ]
    # Call the `chat()` function.
    async for chunk in chat(qa_prompt, {'question': question, 'context': document}):
        yield chunk

async def main():
    sources = [
        ('data/your_file.txt', ['Question 1', 'Question 2']),
        ('data/your_file.md', ['Question 1', 'Question 2']),
    ]
    for file_path, questions in sources:
        document = open(file_path, 'r', encoding='utf-8').read()
        for question in questions:
            print(f'Question: {question}\nAnswer:')
            async for chunk in q_a(question, document):
                print(chunk, end='', flush=True)
            print('\n')

if __name__ == '__main__':
    asyncio.run(main())
