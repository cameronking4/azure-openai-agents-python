from __future__ import annotations

import httpx
from openai import AsyncOpenAI, AzureOpenAI, DefaultAsyncHttpxClient

from . import _openai_shared
from .interface import Model, ModelProvider
from .openai_chatcompletions import OpenAIChatCompletionsModel
from .openai_responses import OpenAIResponsesModel

DEFAULT_MODEL: str = "gpt-4o"


_http_client: httpx.AsyncClient | None = None


# If we create a new httpx client for each request, that would mean no sharing of connection pools,
# which would mean worse latency and resource usage. So, we share the client across requests.
def shared_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = DefaultAsyncHttpxClient()
    return _http_client


class OpenAIProvider(ModelProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | AzureOpenAI | None = None,
        organization: str | None = None,
        project: str | None = None,
        use_responses: bool | None = None,
        # Azure OpenAI specific parameters
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        azure_deployment: str | None = None,
    ) -> None:
        if openai_client is not None:
            assert api_key is None and base_url is None and azure_endpoint is None, (
                "Don't provide api_key, base_url, or azure_endpoint if you provide openai_client"
            )
            self._client: AsyncOpenAI | AzureOpenAI | None = openai_client
        else:
            self._client = None
            self._stored_api_key = api_key
            self._stored_base_url = base_url
            self._stored_organization = organization
            self._stored_project = project
            # Store Azure specific parameters
            self._stored_azure_endpoint = azure_endpoint
            self._stored_api_version = api_version
            self._stored_azure_deployment = azure_deployment

        if use_responses is not None:
            self._use_responses = use_responses
        else:
            self._use_responses = _openai_shared.get_use_responses_by_default()

    # We lazy load the client in case you never actually use OpenAIProvider(). Otherwise
    # AsyncOpenAI() raises an error if you don't have an API key set.
    def _get_client(self) -> AsyncOpenAI | AzureOpenAI:
        if self._client is None:
            # Check if we should use Azure OpenAI
            if self._stored_azure_endpoint or _openai_shared.get_default_azure_endpoint():
                # Create Azure OpenAI client
                azure_endpoint = self._stored_azure_endpoint or _openai_shared.get_default_azure_endpoint()
                api_version = self._stored_api_version or _openai_shared.get_default_api_version() or "2023-07-01-preview"
                
                self._client = AzureOpenAI(
                    api_key=self._stored_api_key or _openai_shared.get_default_openai_key(),
                    azure_endpoint=azure_endpoint,
                    api_version=api_version,
                    azure_deployment=self._stored_azure_deployment or _openai_shared.get_default_azure_deployment(),
                    http_client=shared_http_client(),
                )
            else:
                # Create standard OpenAI client
                self._client = _openai_shared.get_default_openai_client() or AsyncOpenAI(
                    api_key=self._stored_api_key or _openai_shared.get_default_openai_key(),
                    base_url=self._stored_base_url,
                    organization=self._stored_organization,
                    project=self._stored_project,
                    http_client=shared_http_client(),
                )

        return self._client

    def get_model(self, model_name: str | None) -> Model:
        if model_name is None:
            model_name = DEFAULT_MODEL

        client = self._get_client()
        
        # For Azure OpenAI, if a deployment is specified and no model is provided,
        # use the deployment name as the model name
        if hasattr(client, "azure_deployment") and client.azure_deployment and model_name == DEFAULT_MODEL:
            model_name = client.azure_deployment

        return (
            OpenAIResponsesModel(model=model_name, openai_client=client)
            if self._use_responses
            else OpenAIChatCompletionsModel(model=model_name, openai_client=client)
        )
