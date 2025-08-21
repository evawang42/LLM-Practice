# LLM-Practice
This repository contains hands-on practices with LLMs via the Ollama ecosystem, focused on streaming interaction, document Q&A, and prompt engineering.

## Task 1 - Streaming chat with Ollama
This task demonstrates how to build a minimal async streaming chat function using the `ollama-python` client. The core component is an `async generator` that emits LLM-generated text in real-time, chunk by chunk.
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


## Reference
- https://github.com/ollama/ollama-python