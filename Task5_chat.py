"""
Streaming chat with Ollama
"""

from typing import AsyncGenerator, Tuple, Literal, List, Dict
from ollama import AsyncClient
from config import OLLAMA_URL, OLLAMA_MODEL

IPromptMessage = Tuple[Literal['system', 'user', 'assistant', 'tool'], str]
"""The type of a prompt message."""

# Route classification for user queries
ROUTES: Dict[int, str] = {
    1: "Food Ordering",
    2: "Product Query",
    3: "Event Query",
    4: "Shop Query",
    5: "Product Recommendation",
    6: "Corporate Information",
    7: "Others",
}

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

async def input_router(question: str) -> int:
    """
    Classify the user input into one of 7 predefined query routes.

    Args:
        question (str): The user’s input question.

    Returns:
        int: A number between 1–7 indicating the intent category.
    """
    # Prompt
    system_text = (
        "You are an assistant for a fast-food brand. First interpret the message in Chinese, "
        "but your final output must be exactly one Arabic digit (1-7), with no other text, punctuation, or explanation.\n"
        "[ROUTES & Scope]\n"
        "1 = Order flow: place order / pre-order / delivery / modify order / cancel / checkout (if there is clear ordering intent, choose 1 over 2/3/4)\n"
        "2 = Menu & products: availability / price / size-specs / ingredients / allergens / serving time (product level)\n"
        "3 = Marketing events: coupons / discounts / promotions / membership / seasonal campaigns\n"
        "4 = Store operations: store location / business hours / directions / delivery coverage / phone / parking\n"
        "5 = Personalized recommendation: user explicitly asks you to recommend or ‘what to eat’ based on taste/budget/restrictions/history (only when there is an explicit request for recommendations)\n"
        "6 = Brand & company: brand story / policies / recruiting / brand-level comparisons with other chains\n"
        "7 = Others / off-topic: non-food topics or unclear/ambiguous expressions\n"
    )


    # few-shot examples
    prompt: list[IPromptMessage] = [
        ('system', system_text),

        # 1 Food Ordering
        ('user', '請幫我外送兩份經典牛肉堡到內湖，另外加一份薯條。'),
        ('assistant', '1'),

        # 2 Product Query
        ('user', '小杯可樂現在多少錢？'),
        ('assistant', '2'),

        # 3 Event Query
        ('user', '本月是否有折扣碼或會員加碼活動？'),
        ('assistant', '3'),

        # 4 Shop Query
        ('user', '台北車站附近的門市今天營業到幾點？'),
        ('assistant', '4'),

        # 5 Product Recommendation
        ('user', '我不吃牛而且怕辣，預算兩百內，有沒有推薦？'),
        ('assistant', '5'),

        # 6 Corporate Information
        ('user', '品牌的創立故事與核心價值是什麼？'),
        ('assistant', '6'),

        # 7 Others
        ('user', '你覺得最近股市會上漲嗎？'),
        ('assistant', '7'),

        # user input
        ('user', '{question}'),
    ]

    # Call the `chat()` function.
    chunks = []
    async for chunk in chat(prompt, {'question': question}):
        chunks.append(chunk)
    text = "".join(chunks)

    return int(text)


async def chitchat(question: str, history: List[Tuple[str, str]]) -> AsyncGenerator[str, None]:
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

    messages = [system_prompt]
    messages.extend([(role, content) for role, content in history])
    messages.append(("user", "{question}"))

    async for chunk in chat(messages, {'question': question}):
        yield chunk
    
async def online_help_desk_chat(question: str, history: List[Tuple[str, str]]) -> AsyncGenerator[str, None]:
    """
    Main entry point that routes user input to appropriate logic (only proceeds for route 5 = recommendation).

    Args:
        question: The current user question to be answered by the model.
        history:  Prior dialogue turns as a list of `(role, content)`.

    Yields:
        str: Either `Others` output or fallback route label.
    """
    route = await input_router(question)
    if route != 7:
        yield f"No response: {route} - {ROUTES[route]}"
    else:
        async for chunk in chitchat(question, history):
            yield chunk