# Anthropic Claude API - Quick Start Guide

*Last Updated: May 23, 2025*

This quick start guide provides practical examples and common use cases for the Anthropic Claude API and MCP integration.

## Table of Contents

1. [Installation](#installation)
2. [Basic API Usage](#basic-api-usage)
3. [MCP Integration](#mcp-integration)
4. [Message Batches](#message-batches)
5. [Vertex AI Integration](#vertex-ai-integration)
6. [Common Use Cases](#common-use-cases)

## Installation

Using uv (recommended):

```bash
# Basic installation
uv add anthropic

# With Vertex AI support
uv add "anthropic[vertex]"

# With AWS Bedrock support
uv add "anthropic[bedrock]"
```

Set your API key as an environment variable:

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "your_api_key"

# Linux/macOS
export ANTHROPIC_API_KEY="your_api_key"
```

## Basic API Usage

### Simple Message

```python
from anthropic import Anthropic

# Initialize the client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Create a simple message
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Explain the concept of neural networks"}
    ]
)

# Print the response
print(response.content[0].text)
```

### With System Prompt

```python
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    system="You are an AI assistant with expertise in machine learning. Provide clear, concise explanations with practical examples.",
    messages=[
        {"role": "user", "content": "Explain the concept of neural networks"}
    ]
)
```

### Multi-turn Conversation

```python
# First message
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "What are the best practices for Python code?"}
    ]
)

# Continue the conversation
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "What are the best practices for Python code?"},
        {"role": "assistant", "content": response.content[0].text},
        {"role": "user", "content": "Can you provide examples of each practice?"}
    ]
)
```

## MCP Integration

### Basic MCP Integration

```python
# Define MCP tools
tools = [
    {
        "name": "mcp__crawl4ai-rag__perform_rag_query",
        "description": "Search the RAG system for relevant information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
]

# Create a message with tools
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Find information about Python best practices"}
    ],
    tools=tools
)

# Check for tool calls
for content_block in response.content:
    if content_block.type == "tool_use":
        tool_call = content_block.tool_use
        print(f"Tool: {tool_call.name}")
        print(f"Input: {tool_call.input}")
        
        # Process the tool call and get the output
        tool_output = "..." # Your tool processing logic here
        
        # Continue the conversation with the tool output
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": "Find information about Python best practices"},
                {"role": "assistant", "content": [content_block]},
                {"role": "user", "content": ""}
            ],
            tool_outputs=[
                {
                    "tool_call_id": tool_call.id,
                    "output": tool_output
                }
            ]
        )
```

### MCP Connector

```python
# Configure MCP servers
mcp_servers = [
    {
        "server_url": "https://mcp.zapier.com/",
        "name": "zapier",
        "auth": {
            "type": "oauth",
            "token": "your_zapier_oauth_token"
        }
    }
]

# Create a message with MCP servers
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Create a Zapier automation that sends an email when a new file is uploaded to Google Drive"}
    ],
    mcp_servers=mcp_servers
)
```

## Message Batches

### Creating a Batch

```python
# Create a message batch
batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": "request-1",
            "params": {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Write a short poem about AI"}
                ]
            }
        },
        {
            "custom_id": "request-2",
            "params": {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Explain quantum computing"}
                ]
            }
        }
    ]
)

print(f"Batch ID: {batch.id}")
```

### Polling for Completion

```python
import time

# Poll for batch completion
def poll_batch_completion(batch_id, max_attempts=10, delay=2):
    attempts = 0
    while attempts < max_attempts:
        # Get the batch status
        batch_status = client.messages.batches.retrieve(batch_id)
        
        # Check if the batch is complete
        if batch_status.status == "completed":
            return batch_status
        
        print(f"Batch status: {batch_status.status}, waiting...")
        
        # Wait before the next attempt
        time.sleep(delay)
        attempts += 1
    
    raise Exception(f"Batch {batch_id} did not complete within the expected time")

# Poll for completion
batch_status = poll_batch_completion(batch.id)
```

### Retrieving Results

```python
# Retrieve batch results
results = {}
for result in batch_status.results:
    custom_id = result.custom_id
    message = result.message
    results[custom_id] = message

# Process the results
for custom_id, message in results.items():
    print(f"Result for {custom_id}:")
    print(message.content[0].text)
```

## Vertex AI Integration

### Setting Up the Client

```python
from anthropic import AnthropicVertex

# Set up the Anthropic Vertex AI client
client = AnthropicVertex(
    project_id="your_google_cloud_project_id",
    region="us-east5"
)
```

### Creating a Message

```python
# Create a message using Claude on Vertex AI
response = client.messages.create(
    model="claude-opus-4@20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain machine learning"}
    ]
)

# Process the response
content = response.content
text_content = ""

for item in content:
    if item.get("type") == "text":
        text_content += item.get("text", "")

print(text_content)
```

## Common Use Cases

### Content Generation

```python
# Generate creative content
response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=2000,
    system="You are a creative writer specializing in short stories.",
    messages=[
        {"role": "user", "content": "Write a short story about a robot that develops consciousness"}
    ]
)
```

### Data Analysis

```python
# Analyze data
data = """
Year, Revenue, Expenses, Profit
2020, 1000000, 800000, 200000
2021, 1200000, 900000, 300000
2022, 1500000, 1100000, 400000
2023, 1800000, 1300000, 500000
2024, 2000000, 1500000, 500000
"""

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    system="You are a data analyst. Analyze the data and provide insights.",
    messages=[
        {"role": "user", "content": f"Analyze this financial data and provide insights:\n\n{data}"}
    ]
)
```

### Code Generation

```python
# Generate code
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=2000,
    system="You are an expert Python programmer.",
    messages=[
        {"role": "user", "content": "Write a Python function that calculates the Fibonacci sequence up to n terms"}
    ]
)
```

### RAG with MCP

```python
# Configure MCP servers for RAG
mcp_servers = [
    {
        "server_url": "https://your-rag-server.com/",
        "name": "rag-server"
    }
]

# Create a message with RAG
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    system="You are an assistant that answers questions based on retrieved information.",
    messages=[
        {"role": "user", "content": "What are the key features of the new product?"}
    ],
    mcp_servers=mcp_servers
)
```

### Multi-modal Input

```python
import base64

# Read image file
with open("image.jpg", "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode("utf-8")

# Create a message with image
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": "Describe what you see in this image"
                }
            ]
        }
    ]
)
```

## Additional Resources

- [Anthropic API Documentation](https://docs.anthropic.com/en/api/overview)
- [MCP Connector Documentation](https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector)
- [Remote MCP Servers](https://docs.anthropic.com/en/docs/agents-and-tools/remote-mcp-servers)
- [Message Batches Documentation](https://docs.anthropic.com/en/api/messages-batch-examples)
- [Claude on Vertex AI](https://docs.anthropic.com/en/api/claude-on-vertex-ai)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
