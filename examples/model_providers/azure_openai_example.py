#!/usr/bin/env python
"""
Example of using Azure OpenAI with the agents library.

This example demonstrates how to configure and use Azure OpenAI as the model provider
for agents. It shows both direct provider configuration and global configuration options.

Requirements:
- Set the AZURE_OPENAI_API_KEY environment variable with your Azure OpenAI API key
- Replace the example azure_endpoint with your actual Azure OpenAI endpoint
- Replace the deployment_name with your actual Azure OpenAI deployment name
"""

import os
from agents import Agent, run
from agents.models import OpenAIProvider
from agents.models._openai_shared import (
    set_default_azure_endpoint,
    set_default_api_version,
    set_default_azure_deployment,
    set_default_openai_key,
)

# Example 1: Configure Azure OpenAI directly in the agent
def example_direct_configuration():
    # Create an OpenAI provider configured for Azure
    provider = OpenAIProvider(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint="https://your-resource-name.openai.azure.com/",
        api_version="2023-07-01-preview",  # Update to the latest API version as needed
        azure_deployment="deployment-name",  # e.g., gpt-4, gpt-35-turbo, etc.
    )
    
    # Create an agent using the Azure OpenAI provider
    agent = Agent(
        model_provider=provider,
        system_prompt="You are a helpful assistant.",
    )
    
    # Run the agent
    result = run(agent, "What is the capital of France?")
    print(result.output)


# Example 2: Configure Azure OpenAI globally
def example_global_configuration():
    # Set global Azure OpenAI configuration
    set_default_openai_key(os.environ.get("AZURE_OPENAI_API_KEY"))
    set_default_azure_endpoint("https://your-resource-name.openai.azure.com/")
    set_default_api_version("2023-07-01-preview")
    set_default_azure_deployment("deployment-name")
    
    # Create an agent (will use the global Azure OpenAI configuration)
    agent = Agent(
        system_prompt="You are a helpful assistant.",
    )
    
    # Run the agent
    result = run(agent, "What is the capital of Germany?")
    print(result.output)


# Example 3: Using Azure OpenAI with a specific deployment but without specifying model
def example_with_deployment_no_model():
    provider = OpenAIProvider(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint="https://your-resource-name.openai.azure.com/",
        api_version="2023-07-01-preview",
        azure_deployment="deployment-name",
    )
    
    # Note: No model specified, will use the deployment name as the model
    agent = Agent(
        model_provider=provider,
        system_prompt="You are a helpful assistant.",
    )
    
    result = run(agent, "What is the capital of Italy?")
    print(result.output)


if __name__ == "__main__":
    # Uncomment the example you want to run
    # example_direct_configuration()
    # example_global_configuration()
    # example_with_deployment_no_model()
    
    print("Please uncomment one of the example functions to run it.")
    print("Make sure to update the Azure OpenAI configuration with your actual values.")
