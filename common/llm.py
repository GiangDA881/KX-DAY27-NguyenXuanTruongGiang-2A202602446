"""LLM factory. Returns an OpenAI-compatible chat model wired to OpenRouter or xKiro."""

import itertools
import os

from langchain_openai import ChatOpenAI

_KEY_CYCLE = None


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    global _KEY_CYCLE
    api_key = os.environ.get("OPENROUTER_API_KEY")
    pool = os.environ.get("XKIRO_KEY_POOL")
    if pool:
        keys = [k.strip() for k in pool.split(",") if k.strip()]
        if keys:
            if _KEY_CYCLE is None:
                _KEY_CYCLE = itertools.cycle(keys)
            api_key = next(_KEY_CYCLE)

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY or XKIRO_KEY_POOL is not set — check .env")

    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "mistralai/mistral-large-2512"),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.xkiro.com/v1"),
        api_key=api_key,
        temperature=temperature,
    )
