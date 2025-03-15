# Custom LLM providers

## Custom Provider Examples

The examples in this directory demonstrate how you might use a non-OpenAI LLM provider. To run them, first set a base URL, API key and model.

```bash
export EXAMPLE_BASE_URL="..."
export EXAMPLE_API_KEY="..."
export EXAMPLE_MODEL_NAME"..."
```

Then run the examples, e.g.:

```
python examples/model_providers/custom_example_provider.py

Loops within themselves,
Function calls its own being,
Depth without ending.
```

## Azure OpenAI Example

The `azure_openai_example.py` file demonstrates how to use Azure OpenAI with the agents library. To run this example:

1. Set your Azure OpenAI API key:
   ```bash
   export AZURE_OPENAI_API_KEY="your-api-key"
   ```

2. Edit the example file to update:
   - Your Azure OpenAI endpoint URL
   - Your deployment name

3. Run the example:
   ```bash
   python examples/model_providers/azure_openai_example.py
   ```

The example shows three different ways to configure Azure OpenAI:
- Direct configuration in the agent
- Global configuration
- Using a deployment without specifying a model
