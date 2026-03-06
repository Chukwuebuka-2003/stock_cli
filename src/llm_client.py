"""
LLM Client Factory Module

This module provides a unified interface for creating LLM clients that support
multiple providers (Groq and Docker Model Runner). The factory pattern allows
easy switching between providers via configuration.
"""

import logging
from typing import Union

logger = logging.getLogger(__name__)

# Default configurations
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_DOCKER_MODEL = "ai/qwen2.5:latestQ4_0"
DEFAULT_DOCKER_URL = "http://localhost:12434/engines/v1"


class LLMClientWrapper:
    """
    A wrapper that provides a unified interface for LLM clients.

    This class wraps either a Groq or OpenAI client and exposes:
    - chat.completions.create(): The standard completion API
    - model: The configured model name for this provider
    """

    def __init__(self, client, model: str):
        """
        Initialize the wrapper.

        Args:
            client: The underlying LLM client (Groq or OpenAI)
            model: The model name to use for completions
        """
        self._client = client
        self._model = model

        # Expose the chat.completions interface directly
        self.chat = client.chat

    @property
    def model(self) -> str:
        """Return the configured model name for this provider."""
        return self._model


def create_llm_client(config) -> LLMClientWrapper:
    """
    Create an LLM client based on configuration.

    This factory function selects the appropriate LLM provider based on
    the LLM_PROVIDER environment variable or config setting.

    Provider Selection Priority:
    1. LLM_PROVIDER=docker -> Use Docker Model Runner
    2. GROQ_API_KEY set -> Use Groq (default)
    3. Neither configured -> Raise clear error

    Args:
        config: Config object with get() method for retrieving settings

    Returns:
        LLMClientWrapper: A wrapped client with unified interface

    Raises:
        ValueError: If no valid provider is configured

    Environment Variables:
        LLM_PROVIDER: "groq" or "docker" (default: "groq")
        GROQ_API_KEY: API key for Groq (required if LLM_PROVIDER=groq)
        GROQ_MODEL: Model name for Groq (default: "openai/gpt-oss-120b")
        DOCKER_MODEL_RUNNER_URL: Docker Model Runner endpoint
        DOCKER_MODEL_RUNNER_MODEL: Model name for Docker (default: "ai/qwen2.5:latestQ4_0")
    """
    import os

    # Determine provider from config or environment
    provider = os.getenv("LLM_PROVIDER") or config.get("llm_provider", "groq")
    provider = provider.lower().strip()

    if provider == "docker":
        return _create_docker_client(config)
    elif provider == "groq":
        return _create_groq_client(config)
    else:
        raise ValueError(
            f"Invalid LLM_PROVIDER '{provider}'. "
            f"Supported providers: 'groq', 'docker'. "
            f"Set LLM_PROVIDER environment variable or config setting."
        )


def _create_groq_client(config) -> LLMClientWrapper:
    """
    Create a Groq LLM client.

    Args:
        config: Config object with get() method

    Returns:
        LLMClientWrapper wrapping the Groq client

    Raises:
        ValueError: If GROQ_API_KEY is not configured
    """
    import os
    from groq import Groq

    # Get API key from config or environment
    api_key = config.get("groq_api_key") or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not configured. "
            "Please set the GROQ_API_KEY environment variable "
            "or run 'stock-tracker setup-ai' to configure it."
        )

    # Get model from config or environment, or use default
    model = os.getenv("GROQ_MODEL") or config.get("groq_model", DEFAULT_GROQ_MODEL)

    logger.info(f"Creating Groq LLM client with model: {model}")

    client = Groq(api_key=api_key)
    return LLMClientWrapper(client, model)


def _create_docker_client(config) -> LLMClientWrapper:
    """
    Create a Docker Model Runner LLM client.

    Docker Model Runner uses an OpenAI-compatible API, so we use the
    OpenAI SDK with a custom base_url.

    Args:
        config: Config object with get() method

    Returns:
        LLMClientWrapper wrapping the OpenAI client

    Raises:
        ValueError: If Docker Model Runner is not accessible
    """
    import os
    from openai import OpenAI

    # Get endpoint URL from config or environment
    base_url = (
        os.getenv("DOCKER_MODEL_RUNNER_URL") or
        config.get("docker_model_runner_url", DEFAULT_DOCKER_URL)
    )

    # Get model name from config or environment
    model = (
        os.getenv("DOCKER_MODEL_RUNNER_MODEL") or
        config.get("docker_model_runner_model", DEFAULT_DOCKER_MODEL)
    )

    logger.info(f"Creating Docker Model Runner LLM client")
    logger.info(f"  Endpoint: {base_url}")
    logger.info(f"  Model: {model}")

    # Docker Model Runner doesn't require an API key, but the OpenAI SDK
    # requires one to be passed. We use a placeholder value.
    client = OpenAI(
        base_url=base_url,
        api_key="not-needed"  # Docker Model Runner doesn't require authentication
    )

    return LLMClientWrapper(client, model)