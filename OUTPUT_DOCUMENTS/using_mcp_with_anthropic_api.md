# Using Model Context Protocol (MCP) with Anthropic API and Python

*Last Updated: May 23, 2025*

## Table of Contents

1. [Introduction](#introduction)
2. [What is Model Context Protocol (MCP)?](#what-is-model-context-protocol-mcp)
3. [Setting Up Your Environment](#setting-up-your-environment)
4. [Basic MCP Integration](#basic-mcp-integration)
5. [Advanced MCP Usage](#advanced-mcp-usage)
6. [Available Remote MCP Servers](#available-remote-mcp-servers)
7. [Authentication with MCP Servers](#authentication-with-mcp-servers)
8. [Handling Tool Responses](#handling-tool-responses)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)
11. [Complete Example](#complete-example)
12. [Resources](#resources)

## Introduction

This document provides a comprehensive guide on how to use the Model Context Protocol (MCP) with Anthropic's Claude API using Python. MCP allows Claude to connect to remote servers and access external tools and data sources, significantly expanding its capabilities beyond what's possible with the standard API.

## What is Model Context Protocol (MCP)?

The Model Context Protocol (MCP) is a standard that connects AI systems with external tools and data sources. When integrated with Anthropic's Claude API, MCP enables:

- **Direct API integration**: Connect to MCP servers without implementing an MCP client
- **Tool calling support**: Access MCP tools through the Messages API
- **OAuth authentication**: Support for OAuth Bearer tokens for authenticated servers
- **Multiple servers**: Connect to multiple MCP servers in a single request

MCP follows a client-server architecture where Claude (the client) can connect to multiple specialized servers that provide different capabilities.

## Setting Up Your Environment

### Prerequisites

- Python 3.8 or later
- An Anthropic API key
- Access to one or more MCP servers (optional)

### Installation

Use `uv` to install the Anthropic Python SDK:

```bash
uv add anthropic
```

### Environment Setup

Set up your environment variables:

```python
import os

# Set your API key
os.environ["ANTHROPIC_API_KEY"] = "your_api_key"
```

## Basic MCP Integration

### Importing the Anthropic Client

```python
from anthropic import Anthropic

# Initialize the client
client = Anthropic()
```

### Connecting to an MCP Server

To connect to an MCP server, include the `mcp_servers` parameter in your Messages API request:

```python
# Define MCP server configuration
mcp_servers = [
    {
        "type": "url",
        "url": "https://example-server.modelcontextprotocol.io/sse",
        "name": "example-mcp"
        # Optional: "authorization_token": "YOUR_TOKEN"
    }
]

# Create a message with MCP connector
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "What tools do you have available?"}
    ],
    mcp_servers=mcp_servers
)

print(response.content)
```

> **Important**: The MCP connector feature requires the beta header: `"anthropic-beta": "mcp-client-2025-04-04"`. Depending on the SDK version, you may need to pass this as a custom header.

## Advanced MCP Usage

### Using Multiple MCP Servers

You can connect to multiple MCP servers in a single request:

```python
mcp_servers = [
    {
        "type": "url",
        "url": "https://mcp.zapier.com/",
        "name": "zapier-mcp",
        "authorization_token": "ZAPIER_TOKEN"
    },
    {
        "type": "url",
        "url": "https://mcp.stripe.com/v1/sse",
        "name": "stripe-mcp",
        "authorization_token": "STRIPE_TOKEN"
    }
]

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Create a Zap that triggers when a new payment is received in Stripe"}
    ],
    mcp_servers=mcp_servers
)
```

### MCP Server Configuration Options

Each MCP server in the `mcp_servers` array supports the following configuration:

```python
{
    "type": "url",                                     # Always "url" for remote MCP servers
    "url": "https://example-server.example.com/sse",   # The URL of the MCP server
    "name": "example-mcp",                             # A unique name for the server
    "authorization_token": "YOUR_TOKEN"                # Optional OAuth token for authenticated servers
}
```

## Available Remote MCP Servers

Several companies have deployed remote MCP servers that you can connect to via the Anthropic MCP connector API:

| Company | Description | Server URL |
|---------|-------------|------------|
| Zapier | Connect to nearly 8,000 apps through Zapier's automation platform | `https://mcp.zapier.com/` |
| Stripe | Access Stripe APIs for payments, subscriptions, etc. | `https://mcp.stripe.com/v1/sse` |
| Square | Use an agent to build on Square APIs for payments, inventory, orders, etc. | `https://mcp.squareup.com/sse` |
| PayPal | Integrate PayPal commerce capabilities | `https://mcp.paypal.com/sse` |
| Plaid | Analyze, troubleshoot, and optimize Plaid integrations | `https://api.dashboard.plaid.com/mcp/sse` |
| Atlassian | Interact with Jira, Confluence, and other Atlassian products | `https://mcp.atlassian.com/sse` |
| Intercom | Access Intercom customer messaging platform | `https://mcp.intercom.io/sse` |

> **Note**: These servers are third-party services designed to work with the Anthropic API. They are not owned, operated, or endorsed by Anthropic. Only connect to remote MCP servers you trust and review each server's security practices and terms before connecting.

## Authentication with MCP Servers

For MCP servers that require OAuth authentication, you'll need to obtain an access token. The MCP connector beta supports passing an `authorization_token` parameter in the MCP server definition:

```python
mcp_servers = [
    {
        "type": "url",
        "url": "https://mcp.example.com/sse",
        "name": "example-mcp",
        "authorization_token": "YOUR_OAUTH_TOKEN"
    }
]
```

API consumers are expected to handle the OAuth flow and obtain the access token prior to making the API call, as well as refreshing the token as needed.

## Handling Tool Responses

When Claude uses tools from MCP servers, the responses are handled automatically by the MCP connector. However, you may want to extract and process the tool uses and results from Claude's response:

```python
def handle_tool_responses(response):
    """Extract and process tool uses and results from Claude's response."""
    content = response.content
    
    # Check for any tool uses in the response
    for item in content:
        if item.get("type") == "tool_use":
            tool_use = item.get("tool_use", {})
            tool_name = tool_use.get("name")
            tool_input = tool_use.get("input")
            
            print(f"Tool used: {tool_name}")
            print(f"Tool input: {tool_input}")
        
        elif item.get("type") == "tool_result":
            tool_result = item.get("tool_result", {})
            tool_use_id = tool_result.get("tool_use_id")
            result_content = tool_result.get("content")
            
            print(f"Tool result for {tool_use_id}: {result_content}")
        
        elif item.get("type") == "text":
            print(f"Text response: {item.get('text')}")
```

## Best Practices

1. **Error Handling**: Implement robust error handling for MCP server connections and responses.

2. **Security**: 
   - Use a dedicated virtual machine or container with minimal privileges
   - Avoid giving Claude access to sensitive data
   - Limit internet access to an allowlist of domains
   - Have a human confirm decisions with meaningful real-world consequences

3. **Token Management**: Implement proper token refresh mechanisms for OAuth authentication.

4. **Testing**: Test your MCP integration thoroughly before deploying to production.

5. **Monitoring**: Monitor MCP server connections and responses for errors or unexpected behavior.

## Troubleshooting

### Common Issues

1. **Connection Errors**:
   - Ensure the MCP server URL is correct and accessible
   - Check that your authorization token is valid and not expired
   - Verify that the beta header is correctly set

2. **Tool Use Errors**:
   - Make sure the tool name is correctly specified
   - Verify that the tool input matches the expected format
   - Check the MCP server logs for more detailed error information

3. **Response Parsing Errors**:
   - Ensure you're correctly parsing the response content
   - Check for changes in the response format in newer API versions

## Complete Example

Here's a complete example that demonstrates how to use MCP with the Anthropic API:

```python
#!/usr/bin/env python3
"""
Example script demonstrating how to use Anthropic's Claude API with MCP integration.
"""

import os
import json
from typing import Dict, Any, List

from anthropic import Anthropic

def connect_to_mcp_servers(
    api_key: str,
    user_message: str,
    model: str = "claude-3-opus-20240229"
) -> Dict[str, Any]:
    """
    Connect to MCP servers using Anthropic's API.
    
    Args:
        api_key: Anthropic API key
        user_message: The message from the user
        model: The Claude model to use
        
    Returns:
        The response from Claude
    """
    # Initialize the Anthropic client
    client = Anthropic(api_key=api_key)
    
    # Define MCP servers to connect to
    mcp_servers = [
        {
            "type": "url",
            "url": "https://mcp.zapier.com/",
            "name": "zapier-mcp"
            # No authorization token in this example, but would be required in production
        },
        {
            "type": "url",
            "url": "https://mcp.stripe.com/v1/sse",
            "name": "stripe-mcp"
            # No authorization token in this example, but would be required in production
        }
    ]
    
    # Create the message with MCP connector
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            # Add the MCP servers to the request
            mcp_servers=mcp_servers
        )
        
        return response
    except Exception as e:
        print(f"Error connecting to MCP servers: {e}")
        return {"error": str(e)}

def process_response(response):
    """Process the response from Claude, extracting tool uses and results."""
    if hasattr(response, 'error'):
        print(f"Error: {response['error']}")
        return
    
    content = response.content
    
    for item in content:
        if item.get("type") == "tool_use":
            tool_use = item.get("tool_use", {})
            print(f"\nTool used: {tool_use.get('name')}")
            print(f"Tool input: {json.dumps(tool_use.get('input'), indent=2)}")
        elif item.get("type") == "tool_result":
            tool_result = item.get("tool_result", {})
            print(f"\nTool result for {tool_result.get('tool_use_id')}")
            print(f"Result: {tool_result.get('content')}")
        elif item.get("type") == "text":
            print(f"\nText response: {item.get('text')}")

def main():
    """Main function to demonstrate MCP integration with Anthropic's API."""
    # Get API key from environment variable
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return
    
    # Example user message that might require tools from MCP servers
    user_message = "I need to create a Zap that triggers when a new payment is received in Stripe. Can you help me set this up?"
    
    print("Connecting to MCP servers...")
    response = connect_to_mcp_servers(
        api_key=api_key,
        user_message=user_message
    )
    
    # Process the response
    process_response(response)

if __name__ == "__main__":
    main()
```

## Resources

- [Anthropic API Documentation](https://docs.anthropic.com/en/api/overview)
- [MCP Connector Documentation](https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector)
- [Remote MCP Servers](https://docs.anthropic.com/en/docs/agents-and-tools/remote-mcp-servers)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
