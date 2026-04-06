"""
Direct httpx-based Groq API client.
Avoids SDK compatibility issues on Vercel serverless.
"""
import os
import json
from typing import AsyncGenerator
import httpx

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', '')}",
        "Content-Type": "application/json",
    }


async def chat(
    messages: list[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 600,
    response_format: dict | None = None,
) -> str:
    """Make a single chat completion request and return the content string."""
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        body["response_format"] = response_format

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(GROQ_API_URL, headers=_headers(), json=body)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def stream_chat(
    messages: list[dict],
    model: str,
    temperature: float = 0.4,
    max_tokens: int = 2000,
) -> AsyncGenerator[str, None]:
    """Stream a chat completion, yielding text chunks."""
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", GROQ_API_URL, headers=_headers(), json=body
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
