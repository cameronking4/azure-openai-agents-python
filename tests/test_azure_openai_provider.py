import os
import unittest
from unittest.mock import MagicMock, patch

from openai import AzureOpenAI, AsyncOpenAI

from agents.models.openai_provider import OpenAIProvider
from agents.models._openai_shared import (
    set_default_azure_endpoint,
    set_default_api_version,
    set_default_azure_deployment,
    get_default_azure_endpoint,
    get_default_api_version,
    get_default_azure_deployment,
)


class TestAzureOpenAIProvider(unittest.TestCase):
    def setUp(self):
        # Clear any default settings before each test
        self._clear_defaults()

    def tearDown(self):
        # Clear any default settings after each test
        self._clear_defaults()

    def _clear_defaults(self):
        # Reset all default settings to None
        set_default_azure_endpoint(None)  # type: ignore
        set_default_api_version(None)  # type: ignore
        set_default_azure_deployment(None)  # type: ignore

    @patch("agents.models.openai_provider.AzureOpenAI")
    def test_azure_openai_provider_direct_config(self, mock_azure_openai):
        # Test creating an OpenAI provider with Azure configuration
        provider = OpenAIProvider(
            api_key="test-api-key",
            azure_endpoint="https://test-endpoint.openai.azure.com/",
            api_version="2023-07-01-preview",
            azure_deployment="test-deployment",
        )

        # Access the client to trigger lazy loading
        provider._get_client()

        # Verify AzureOpenAI was called with the correct parameters
        mock_azure_openai.assert_called_once()
        args, kwargs = mock_azure_openai.call_args
        self.assertEqual(kwargs["api_key"], "test-api-key")
        self.assertEqual(kwargs["azure_endpoint"], "https://test-endpoint.openai.azure.com/")
        self.assertEqual(kwargs["api_version"], "2023-07-01-preview")
        self.assertEqual(kwargs["azure_deployment"], "test-deployment")

    @patch("agents.models.openai_provider.AzureOpenAI")
    def test_azure_openai_provider_global_config(self, mock_azure_openai):
        # Set global Azure OpenAI configuration
        set_default_azure_endpoint("https://global-endpoint.openai.azure.com/")
        set_default_api_version("2023-07-01-preview")
        set_default_azure_deployment("global-deployment")

        # Create provider without direct Azure configuration
        provider = OpenAIProvider(api_key="test-api-key")

        # Access the client to trigger lazy loading
        provider._get_client()

        # Verify AzureOpenAI was called with the global configuration
        mock_azure_openai.assert_called_once()
        args, kwargs = mock_azure_openai.call_args
        self.assertEqual(kwargs["api_key"], "test-api-key")
        self.assertEqual(kwargs["azure_endpoint"], "https://global-endpoint.openai.azure.com/")
        self.assertEqual(kwargs["api_version"], "2023-07-01-preview")
        self.assertEqual(kwargs["azure_deployment"], "global-deployment")

    @patch("agents.models.openai_provider.AsyncOpenAI")
    def test_standard_openai_when_no_azure_config(self, mock_async_openai):
        # Create provider without Azure configuration
        provider = OpenAIProvider(api_key="test-api-key", base_url="https://api.openai.com/v1")

        # Access the client to trigger lazy loading
        provider._get_client()

        # Verify AsyncOpenAI was called, not AzureOpenAI
        mock_async_openai.assert_called_once()
        args, kwargs = mock_async_openai.call_args
        self.assertEqual(kwargs["api_key"], "test-api-key")
        self.assertEqual(kwargs["base_url"], "https://api.openai.com/v1")

    @patch("agents.models.openai_provider.AzureOpenAI")
    def test_model_name_with_azure_deployment(self, mock_azure_openai):
        # Mock the AzureOpenAI instance
        mock_client = MagicMock()
        mock_client.azure_deployment = "test-deployment"
        mock_azure_openai.return_value = mock_client

        # Create provider with Azure configuration
        provider = OpenAIProvider(
            api_key="test-api-key",
            azure_endpoint="https://test-endpoint.openai.azure.com/",
            api_version="2023-07-01-preview",
            azure_deployment="test-deployment",
        )

        # Get a model without specifying a model name
        model = provider.get_model(None)

        # Verify the model name is set to the deployment name
        self.assertEqual(model.model, "test-deployment")

    def test_default_getters_and_setters(self):
        # Test the getter and setter functions for Azure defaults
        set_default_azure_endpoint("https://test-endpoint.openai.azure.com/")
        set_default_api_version("2023-07-01-preview")
        set_default_azure_deployment("test-deployment")

        self.assertEqual(get_default_azure_endpoint(), "https://test-endpoint.openai.azure.com/")
        self.assertEqual(get_default_api_version(), "2023-07-01-preview")
        self.assertEqual(get_default_azure_deployment(), "test-deployment")


if __name__ == "__main__":
    unittest.main()
