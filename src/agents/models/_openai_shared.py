from __future__ import annotations

from openai import AsyncOpenAI

_default_openai_key: str | None = None
_default_openai_client: AsyncOpenAI | None = None
_use_responses_by_default: bool = True

# Azure OpenAI specific defaults
_default_azure_endpoint: str | None = None
_default_api_version: str | None = None
_default_azure_deployment: str | None = None


def set_default_openai_key(key: str) -> None:
    global _default_openai_key
    _default_openai_key = key


def get_default_openai_key() -> str | None:
    return _default_openai_key


def set_default_openai_client(client: AsyncOpenAI) -> None:
    global _default_openai_client
    _default_openai_client = client


def get_default_openai_client() -> AsyncOpenAI | None:
    return _default_openai_client


def set_use_responses_by_default(use_responses: bool) -> None:
    global _use_responses_by_default
    _use_responses_by_default = use_responses


def get_use_responses_by_default() -> bool:
    return _use_responses_by_default


# Azure OpenAI specific functions
def set_default_azure_endpoint(endpoint: str) -> None:
    global _default_azure_endpoint
    _default_azure_endpoint = endpoint


def get_default_azure_endpoint() -> str | None:
    return _default_azure_endpoint


def set_default_api_version(api_version: str) -> None:
    global _default_api_version
    _default_api_version = api_version


def get_default_api_version() -> str | None:
    return _default_api_version


def set_default_azure_deployment(deployment: str) -> None:
    global _default_azure_deployment
    _default_azure_deployment = deployment


def get_default_azure_deployment() -> str | None:
    return _default_azure_deployment
