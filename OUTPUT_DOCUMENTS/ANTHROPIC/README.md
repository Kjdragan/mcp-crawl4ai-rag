# Anthropic Claude API Integration Guide

*Last Updated: May 23, 2025*

## Table of Contents

1. [Introduction](#introduction)
2. [Scripts Overview](#scripts-overview)
3. [Installation Requirements](#installation-requirements)
4. [MCP Integration](#mcp-integration)
5. [Message Batches](#message-batches)
6. [Vertex AI Integration](#vertex-ai-integration)
7. [Remote MCP Servers](#remote-mcp-servers)
8. [Usage Examples](#usage-examples)
9. [Troubleshooting](#troubleshooting)
10. [Resources](#resources)

## Introduction

This documentation provides a comprehensive guide to the Anthropic Claude API integration examples and scripts in this directory. These scripts demonstrate various ways to interact with Anthropic's Claude models, including direct API access, Model Context Protocol (MCP) integration, message batches, and integration with Google Cloud's Vertex AI.

## Scripts Overview

| Script Name | Description |
|-------------|-------------|
| `anthropic_mcp_connector_example.py` | Demonstrates how to use Anthropic's MCP connector feature to connect to remote MCP servers directly from the Messages API |
| `anthropic_mcp_example.py` | Shows how to set up the Anthropic API client with MCP tools and handle tool calls |
| `anthropic_remote_mcp_example.py` | Illustrates how to connect to third-party remote MCP servers like Zapier, Stripe, etc. |
| `anthropic_message_batch_example.py` | Demonstrates how to use Anthropic's Message Batches API for processing multiple requests efficiently |
| `anthropic_streaming_example.py` | Shows how to use Anthropic's streaming API for real-time responses |
| `anthropic_tool_use_example.py` | Demonstrates how to implement tool use with Claude, including defining tools and handling tool calls |
| `anthropic_vision_example.py` | Shows how to use Claude's vision capabilities to analyze and describe images |
| `anthropic_embeddings_example.py` | Demonstrates how to use Claude's embeddings for semantic search and RAG systems |
| `claude_vertex_ai_example.py` | Shows how to use Anthropic's Claude models via Google Cloud's Vertex AI platform |
| `using_mcp_with_anthropic_api.md` | Comprehensive guide on using the Model Context Protocol (MCP) with Anthropic's Claude API |

## Installation Requirements

All scripts require the Anthropic Python SDK. Install using uv (as per user preferences):

```bash
uv add anthropic
```

For specific integrations, additional packages may be required:

- **Vertex AI**: `uv add "anthropic[vertex]"`
- **AWS Bedrock**: `uv add "anthropic[bedrock]"`

## MCP Integration

The Model Context Protocol (MCP) allows Claude to connect to remote servers and access external tools and data sources. Our scripts demonstrate three key aspects of MCP integration:

### 1. Basic MCP Integration (`anthropic_mcp_example.py`)

This script shows how to:
- Set up the Anthropic API client
- Configure MCP tools for use with Claude
- Make API calls to Claude with MCP tool definitions
- Handle tool responses and continue the conversation

### 2. MCP Connector (`anthropic_mcp_connector_example.py`)

This script demonstrates Anthropic's MCP connector feature, which enables:
- Direct API integration without implementing a separate MCP client
- Tool calling support through the Messages API
- OAuth authentication for authenticated servers
- Connecting to multiple MCP servers in a single request

### 3. Remote MCP Servers (`anthropic_remote_mcp_example.py`)

This script shows how to connect to third-party remote MCP servers, including:
- Zapier - `https://mcp.zapier.com/`
- Stripe - `https://mcp.stripe.com/v1/sse`
- Square - `https://mcp.squareup.com/sse`
- PayPal - `https://mcp.paypal.com/sse`
- And others

## Message Batches

The `anthropic_message_batch_example.py` script demonstrates how to use Anthropic's Message Batches API for processing multiple Claude requests efficiently in parallel. Key features include:

- Creating a message batch with multiple requests
- Polling for batch completion with status updates
- Retrieving and processing batch results
- Listing recent message batches in your workspace

This is particularly useful for high-throughput applications that need to process many Claude requests efficiently.

## Vertex AI Integration

The `claude_vertex_ai_example.py` script shows how to use Anthropic's Claude models via Google Cloud's Vertex AI platform. Key features include:

- Setting up the Anthropic Vertex AI client
- Creating messages using Claude on Vertex AI
- Processing responses from Claude
- Handling both simple and multimodal messages

This integration is ideal for users who are already using Google Cloud and want to access Claude models through Vertex AI.

## Remote MCP Servers

Several companies have deployed remote MCP servers that you can connect to via the Anthropic MCP connector API. Our examples demonstrate how to connect to these servers, including:

- **Zapier**: Connect to nearly 8,000 apps through Zapier's automation platform
- **Stripe**: Access Stripe APIs for payments, subscriptions, etc.
- **Square**: Use an agent to build on Square APIs for payments, inventory, orders, etc.
- **PayPal**: Integrate PayPal commerce capabilities
- **Plaid**: Analyze, troubleshoot, and optimize Plaid integrations
- **Atlassian**: Interact with Jira, Confluence, and other Atlassian products
- **Intercom**: Access Intercom customer messaging platform

## Usage Examples

### Basic MCP Example

```python
from anthropic import Anthropic

client = Anthropic(api_key="your_api_key")

# Define MCP tools
mcp_tools = [
    {
        "name": "mcp__crawl4ai-rag__perform_rag_query",
        "description": "Search the RAG system for relevant information"
    }
]

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Search for information about Python best practices"}
    ],
    tools=mcp_tools
)

print(response.content)
```

### Message Batches Example

```python
from anthropic import Anthropic

client = Anthropic(api_key="your_api_key")

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

### Vertex AI Example

```python
from anthropic import AnthropicVertex

client = AnthropicVertex(
    project_id="your_google_cloud_project_id",
    region="us-east5"
)

response = client.messages.create(
    model="claude-opus-4@20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain machine learning"}
    ]
)

print(response.content)
```

## Troubleshooting

### Common Issues

1. **API Key Authentication**:
   - Ensure your Anthropic API key is correctly set as an environment variable
   - Check that the API key has the necessary permissions

2. **MCP Integration**:
   - Verify that the beta header is correctly set: `"anthropic-beta": "mcp-client-2025-04-04"`
   - Ensure MCP server URLs are accessible and correctly formatted

3. **Message Batches**:
   - Check batch status regularly to identify any failed requests
   - Use custom IDs to match results to original requests

4. **Vertex AI**:
   - Ensure Google Cloud authentication is properly set up
   - Verify that Claude is available in your selected region

## Resources

- [Anthropic API Documentation](https://docs.anthropic.com/en/api/overview)
- [MCP Connector Documentation](https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector)
- [Remote MCP Servers](https://docs.anthropic.com/en/docs/agents-and-tools/remote-mcp-servers)
- [Message Batches Documentation](https://docs.anthropic.com/en/api/messages-batch-examples)
- [Streaming Documentation](https://docs.anthropic.com/en/docs/build-with-claude/streaming)
- [Tool Use Documentation](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use)
- [Vision Documentation](https://docs.anthropic.com/en/docs/build-with-claude/vision)
- [Embeddings Documentation](https://docs.anthropic.com/en/docs/build-with-claude/embeddings)
- [Claude on Vertex AI](https://docs.anthropic.com/en/api/claude-on-vertex-ai)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
