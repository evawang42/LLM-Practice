# LLM-Practice
This repository contains hands-on practices with LLMs via the Ollama ecosystem, focused on streaming interaction, document Q&A, and prompt engineering.

## Task 1 - Streaming chat with Ollama
This task demonstrates how to build a async streaming chat function using the `ollama-python` client. The core component is an `async generator` that emits LLM-generated text in real-time, chunk by chunk.
### Code Summary
```python
async def chat(input: dict[str, str]) -> AsyncGenerator[str, None]:
```
- Uses `ollama.AsyncClient.chat(...)` with `stream=True`.
- Supports streaming responses from local or remote Ollama server.

`config.py` example:
```py
OLLAMA_URL = "http://127.0.0.1:11434"
```

## Task 2 - Streaming Q&A Over Local Documents (Write prompts)
This task demonstrates a retrieval-free, streaming question-answering system where the LLM answers questions strictly based on the content of local text files.
### How to Use
#### 1. Prepare Your Files
- Place your .txt or .md files in the data/ directory.
- Each file should contain clean, plain-text content.
#### 2. Modify the Script
- Update the sources list in the `main()` function:
```python
sources = [
    ('data/your_file.txt', ['Question 1', 'Question 2']),
    ('data/your_file.md', ['Question 1', 'Question 2']),
]
```
### 3. Prompt Template
- You may edit the prompt to change the language, style, or response behavior.
```python
qa_prompt: list[IPromptMessage] = [...]
```

## Task 3 - Streaming Helpdesk Chatbot for Fast-Food Restaurant (Designing a Router)
This task implements a multi-intent virtual assistant for a fast-food restaurant using Ollama. It classifies user queries into different categories—like ordering, product inquiries, event promos, store logistics, or food recommendations—and responds accordingly using an LLM.
### How It Works
#### 1. Intent Routing (`input_router`)
- Uses few-shot prompts to classify the user message into one of 7 predefined query routes.
- Only Route 5 (Recommendation) leads to a full answer.
#### 2. Food Recommendation Logic
- This module generates food recommendations by analyzing the following inputs:
    - User’s message
    - Full menu (CSV)
    - Purchase history (list of past meals)
### How to Use
#### 1. Add Your Own Questions
- Modify the questions list in main():
```python
questions = [
    ("Question 1", []),
    ("Question 2", []),
    ("Question 3 (Recommendation)", []),
    ("Question 4 (Recommendation)", [["Item 1", "Item 2"], ["Item 3"]]),
]
```
- The `questions` variable allows you to define a sequence of user queries, optionally enhanced with historical context. This setup supports both simple Q&A interactions and more personalized recommendation tasks.
- Each element in the questions list is a tuple:
    - question_text: A string representing the user's question or prompt.
    - history: An optional list representing prior interactions, preferences, or meal history.
```python
(question_text: str, history: list)
```
#### 2. Use Your Own Menu
- Replace your_file.csv with your own CSV file.
- Update path in code:
```python
menu_path = Path("data/your_file.csv")
```
### ⚠ Known Issues
- Despite explicitly prompting the LLM to respond in Traditional Chinese (zh-Hant), some responses are still occasionally returned in English.

## Task 4 – Streaming Chat (SSE) with Ollama
This task shows how to build a streaming chat backend with aiohttp, plus a frontend that renders the model’s answer as it streams.
It consists of two files:
- Task4_chat.py: chat pipeline to Ollama (LLM)
- Task4_server.py: aiohttp server that exposes POST/chat and streams text via Server-Sent Events (SSE)
### How It Works
1. Frontend → /chat: sends JSON `{ "query": "…", "history": [ { "role": "user", "content": "…" }] }`
2. Server (`chat_handler`): prepares an SSE response, calls `online_help_desk_chat()`.
3. Chat core (`online_help_desk_chat`): builds messages `[(system, …), (user|assistant, …), …, (user, "{question}")]`, then delegates to `chat()`.
4. `chat()`: calls Ollama AsyncClient.chat(stream=True), yields chunks.
5. Server: wraps each chunk into SSE frames: `event: data\ndata: {"action":"response","message":"…"}\n\n`
6. Frontend: appends `message` text to the chat window as it arrives.
7. Server emits `event: end` to close the stream.
### Frontend example
```javascript
const response = await fetch('http://localhost:8002/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({query, history: chatHistory.slice(0, -1)})
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

let buffer = '';
while (true) {
    const {done, value} = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, {stream: true});
    const lines = buffer.split('\n');
    buffer = lines.pop();

    for (const line of lines) {
        if (line.startsWith('event: data')) continue;
        if (line.startsWith('event: end')) return;
        if (line.startsWith('data:')) {
            const payload = JSON.parse(line.slice(5).trim());
            if (payload.action === 'response') console.log(payload.message);
        }
    }
}
```
### API Contract
Endpoint: `POST /chat`

Request body: `{"query": "string","history": [{"role":"user|assistant","content":"string"}]}`
SSE Response:
- Streamed chunk:
```
event: data
data: {"action":"response","message":"<partial text>"}
```
- Error:
```
event: error
data: {"action":"error","message":"<reason>"}
```
- End:
```
event: end
data: {}
```

## Reference
- [Ollama Python Library](https://github.com/ollama/ollama-python)
- [Understanding Server-Sent Events (SSE) with Node.js](https://itsfuad.medium.com/understanding-server-sent-events-sse-with-node-js-3e881c533081)
- [Web Server Quickstart](https://docs.aiohttp.org/en/stable/web_quickstart.html)