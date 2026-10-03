from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map provider names and common aliases / typos to a canonical provider name.

    Aliases handled:
    - openai, open_ai, open-ai, chatgpt -> 'openai'
    - custom, openai_compatible, openai-compatible, vllm, local, self-hosted -> 'custom'
    - gemini, google, google_genai, google-genai, google_generative_ai -> 'gemini'
    - anthropic, anthorpic, claude -> 'anthropic'
    - ollama, olama -> 'ollama'
    - openrouter, open_router, open-router -> 'openrouter'
    """
    if not value or not isinstance(value, str):
        raise ValueError("Provider name must be a non-empty string.")

    cleaned = value.strip().lower()
    alias_map = {
        "openai": "openai",
        "open_ai": "openai",
        "open-ai": "openai",
        "chatgpt": "openai",
        "custom": "custom",
        "openai_compatible": "custom",
        "openai-compatible": "custom",
        "vllm": "custom",
        "local": "custom",
        "self-hosted": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "google_genai": "gemini",
        "google-genai": "gemini",
        "google_generative_ai": "gemini",
        "anthropic": "anthropic",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "ollama": "ollama",
        "olama": "ollama",
        "openrouter": "openrouter",
        "open_router": "openrouter",
        "open-router": "openrouter",
    }

    if cleaned in alias_map:
        return alias_map[cleaned]

    raise ValueError(
        f"Unsupported provider: '{value}'. Expected one of: openai, custom, gemini, anthropic, ollama, openrouter."
    )


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Mapping:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    elif provider == "custom":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "base_url": config.base_url or "http://localhost:8000/v1",
            "api_key": config.api_key or "EMPTY",
        }
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatGoogleGenerativeAI(**kwargs)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["anthropic_api_url"] = config.base_url
        return ChatAnthropic(**kwargs)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    elif provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenRouter(**kwargs)

    else:
        raise ValueError(f"Unsupported provider: {provider}")
