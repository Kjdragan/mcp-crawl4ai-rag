# Model Context Protocol (MCP) Integration with Anthropic's Claude API

*Last Updated: May 23, 2025*

This guide provides a comprehensive overview of integrating the Model Context Protocol (MCP) with Anthropic's Claude API, including implementation details, best practices, and examples.

## Table of Contents

1. [Introduction to MCP](#introduction-to-mcp)
2. [MCP Integration Methods](#mcp-integration-methods)
3. [MCP Connector](#mcp-connector)
4. [Remote MCP Servers](#remote-mcp-servers)
5. [Implementation Examples](#implementation-examples)
6. [Advanced Use Cases](#advanced-use-cases)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Introduction to MCP

The Model Context Protocol (MCP) is a standardized protocol that enables AI models to connect with external tools and data sources. It allows Claude to:

- Access real-time information beyond its training data
- Perform actions in external systems
- Use specialized tools for specific tasks
- Integrate with third-party services

MCP works by establishing a communication channel between Claude and external servers that implement the MCP specification. This enables Claude to:

1. Request information from external sources
2. Call functions in external systems
3. Receive and process responses from external tools
4. Continue conversations with the context from external tools

## MCP Integration Methods

There are two primary methods for integrating MCP with Anthropic's Claude API:

### 1. Direct Tool Definition

This method involves defining tools directly in the API request using the `tools` parameter. Claude can then use these tools during the conversation.

```python
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Define tools
tools = [
    {
        "name": "search_database",
        "description": "Search a database for information",
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
        {"role": "user", "content": "Find information about quantum computing"}
    ],
    tools=tools
)
```

### 2. MCP Connector

This method uses Anthropic's MCP connector feature to connect to remote MCP servers directly from the Messages API. The MCP connector handles the communication protocol between Claude and the MCP server.

```python
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Configure MCP servers
mcp_servers = [
    {
        "server_url": "https://mcp.example.com/",
        "name": "example-server"
    }
]

# Create a message with MCP servers
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Find information about quantum computing"}
    ],
    mcp_servers=mcp_servers
)
```

## MCP Connector

The MCP connector is a feature of Anthropic's Messages API that simplifies integration with external tools by handling the server connection and communication protocol.

### Key Features

- **Direct API Integration**: Connect to MCP servers directly from the Messages API
- **Tool Calling Support**: Enable Claude to call tools on MCP servers
- **OAuth Authentication**: Support for authenticated MCP servers
- **Multiple MCP Servers**: Connect to multiple MCP servers in a single request

### Configuration Options

The MCP connector supports the following configuration options:

```python
mcp_servers = [
    {
        "server_url": "https://mcp.example.com/",  # Required: URL of the MCP server
        "name": "example-server",                 # Required: Name of the MCP server
        "auth": {                                 # Optional: Authentication details
            "type": "oauth",                      # Authentication type (oauth, basic, etc.)
            "token": "YOUR_OAUTH_TOKEN"           # Authentication token
        },
        "headers": {                              # Optional: Additional headers
            "X-Custom-Header": "value"
        },
        "timeout": 30                             # Optional: Timeout in seconds
    }
]
```

### Authentication Methods

The MCP connector supports several authentication methods:

#### OAuth Authentication

```python
"auth": {
    "type": "oauth",
    "token": "YOUR_OAUTH_TOKEN"
}
```

#### Basic Authentication

```python
"auth": {
    "type": "basic",
    "username": "your_username",
    "password": "your_password"
}
```

#### API Key Authentication

```python
"auth": {
    "type": "api_key",
    "key": "YOUR_API_KEY"
}
```

## Remote MCP Servers

Several companies have deployed remote MCP servers that you can connect to via the Anthropic MCP connector API.

### Available Remote MCP Servers

| Company | Server URL | Description |
|---------|------------|-------------|
| Zapier | `https://mcp.zapier.com/` | Connect to nearly 8,000 apps through Zapier's automation platform |
| Stripe | `https://mcp.stripe.com/v1/sse` | Access Stripe APIs for payments, subscriptions, etc. |
| Square | `https://mcp.squareup.com/sse` | Use an agent to build on Square APIs for payments, inventory, orders, etc. |
| PayPal | `https://mcp.paypal.com/sse` | Integrate PayPal commerce capabilities |
| Plaid | `https://mcp.plaid.com/sse` | Analyze, troubleshoot, and optimize Plaid integrations |
| Atlassian | `https://mcp.atlassian.com/sse` | Interact with Jira, Confluence, and other Atlassian products |
| Intercom | `https://mcp.intercom.com/sse` | Access Intercom customer messaging platform |

### Connecting to Remote MCP Servers

```python
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Configure remote MCP servers
mcp_servers = [
    {
        "server_url": "https://mcp.zapier.com/",
        "name": "zapier",
        "auth": {
            "type": "oauth",
            "token": os.environ.get("ZAPIER_OAUTH_TOKEN")
        }
    },
    {
        "server_url": "https://mcp.stripe.com/v1/sse",
        "name": "stripe",
        "auth": {
            "type": "oauth",
            "token": os.environ.get("STRIPE_OAUTH_TOKEN")
        }
    }
]

# Create a message with remote MCP servers
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Create a Zapier automation that sends an email when a new Stripe payment is received"}
    ],
    mcp_servers=mcp_servers
)
```

## Implementation Examples

### Basic MCP Integration

```python
from anthropic import Anthropic
import os
import json

# Initialize the Anthropic client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Define MCP tools
def get_mcp_tools():
    return [
        {
            "name": "search_database",
            "description": "Search a database for information",
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
        {"role": "user", "content": "Find information about quantum computing"}
    ],
    tools=get_mcp_tools()
)

# Handle tool calls
def handle_tool_calls(response):
    if hasattr(response, "content") and len(response.content) > 0:
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_call = content_block.tool_use
                tool_name = tool_call.name
                tool_input = tool_call.input
                
                print(f"Tool call: {tool_name}")
                print(f"Input: {json.dumps(tool_input, indent=2)}")
                
                # Process the tool call
                if tool_name == "search_database":
                    tool_output = search_database(tool_input.get("query"))
                else:
                    tool_output = {"error": f"Unknown tool: {tool_name}"}
                
                # Continue the conversation with the tool output
                continue_conversation(response.id, tool_call.id, tool_output)

# Process tool call
def search_database(query):
    # This is a mock implementation
    return {
        "results": [
            {
                "title": "Introduction to Quantum Computing",
                "content": "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations."
            },
            {
                "title": "Quantum Bits (Qubits)",
                "content": "The basic unit of quantum information is the qubit, which can represent a 0, a 1, or any quantum superposition of these two states."
            }
        ]
    }

# Continue the conversation with the tool output
def continue_conversation(message_id, tool_call_id, tool_output):
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": "Find information about quantum computing"},
            {"role": "assistant", "content": [{"type": "tool_use", "tool_use": {"id": tool_call_id, "name": "search_database", "input": {"query": "quantum computing"}}}]},
            {"role": "user", "content": ""}
        ],
        tool_outputs=[
            {
                "tool_call_id": tool_call_id,
                "output": json.dumps(tool_output)
            }
        ]
    )
    
    print("Response after tool call:")
    for content_block in response.content:
        if content_block.type == "text":
            print(content_block.text)

# Handle tool calls
handle_tool_calls(response)
```

### MCP Connector Example

```python
from anthropic import Anthropic
import os

# Initialize the Anthropic client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Configure MCP servers
def get_mcp_servers():
    return [
        {
            "server_url": "https://mcp.example.com/",
            "name": "example-server",
            "auth": {
                "type": "oauth",
                "token": os.environ.get("EXAMPLE_SERVER_TOKEN")
            }
        }
    ]

# Create a message with MCP servers
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Find information about quantum computing"}
    ],
    mcp_servers=get_mcp_servers()
)

# Print the response
print("Response:")
for content_block in response.content:
    if content_block.type == "text":
        print(content_block.text)
```

### Multiple MCP Servers Example

```python
from anthropic import Anthropic
import os

# Initialize the Anthropic client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Configure multiple MCP servers
def get_multiple_mcp_servers():
    return [
        {
            "server_url": "https://mcp.zapier.com/",
            "name": "zapier",
            "auth": {
                "type": "oauth",
                "token": os.environ.get("ZAPIER_OAUTH_TOKEN")
            }
        },
        {
            "server_url": "https://mcp.stripe.com/v1/sse",
            "name": "stripe",
            "auth": {
                "type": "oauth",
                "token": os.environ.get("STRIPE_OAUTH_TOKEN")
            }
        }
    ]

# Create a message with multiple MCP servers
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Create a Zapier automation that sends an email when a new Stripe payment is received"}
    ],
    mcp_servers=get_multiple_mcp_servers()
)

# Print the response
print("Response:")
for content_block in response.content:
    if content_block.type == "text":
        print(content_block.text)
```

## Advanced Use Cases

### RAG with MCP

```python
from anthropic import Anthropic
import os

# Initialize the Anthropic client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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

# Print the response
print("Response:")
for content_block in response.content:
    if content_block.type == "text":
        print(content_block.text)
```

### Multi-step Tool Calls

```python
from anthropic import Anthropic
import os
import json

# Initialize the Anthropic client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Define MCP tools for multi-step process
tools = [
    {
        "name": "search_products",
        "description": "Search for products in the catalog",
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
    },
    {
        "name": "get_product_details",
        "description": "Get detailed information about a product",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The product ID"
                }
            },
            "required": ["product_id"]
        }
    },
    {
        "name": "add_to_cart",
        "description": "Add a product to the shopping cart",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The product ID"
                },
                "quantity": {
                    "type": "integer",
                    "description": "The quantity to add"
                }
            },
            "required": ["product_id", "quantity"]
        }
    }
]

# Create a message with multi-step tools
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "I'm looking for a new laptop. Can you help me find one and add it to my cart?"}
    ],
    tools=tools
)

# Process the multi-step tool calls
def process_multi_step_tool_calls(response, conversation_history=None):
    if conversation_history is None:
        conversation_history = [
            {"role": "user", "content": "I'm looking for a new laptop. Can you help me find one and add it to my cart?"}
        ]
    
    if hasattr(response, "content") and len(response.content) > 0:
        # Add assistant response to conversation history
        assistant_content = []
        for content_block in response.content:
            assistant_content.append(content_block)
        
        conversation_history.append({"role": "assistant", "content": assistant_content})
        
        # Check for tool calls
        tool_outputs = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_call = content_block.tool_use
                tool_name = tool_call.name
                tool_input = tool_call.input
                
                print(f"Tool call: {tool_name}")
                print(f"Input: {json.dumps(tool_input, indent=2)}")
                
                # Process the tool call
                if tool_name == "search_products":
                    tool_output = search_products(tool_input.get("query"))
                elif tool_name == "get_product_details":
                    tool_output = get_product_details(tool_input.get("product_id"))
                elif tool_name == "add_to_cart":
                    tool_output = add_to_cart(tool_input.get("product_id"), tool_input.get("quantity"))
                else:
                    tool_output = {"error": f"Unknown tool: {tool_name}"}
                
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": json.dumps(tool_output)
                })
        
        # If there are tool calls, continue the conversation
        if tool_outputs:
            conversation_history.append({"role": "user", "content": ""})
            
            # Continue the conversation with the tool outputs
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=conversation_history,
                tool_outputs=tool_outputs
            )
            
            # Process the next response
            return process_multi_step_tool_calls(response, conversation_history)
        
        # No tool calls, return the final response
        return response, conversation_history
    
    return response, conversation_history

# Mock implementations of tool functions
def search_products(query):
    return {
        "products": [
            {"id": "laptop-001", "name": "MacBook Pro 16-inch", "price": 2399.99},
            {"id": "laptop-002", "name": "Dell XPS 15", "price": 1799.99},
            {"id": "laptop-003", "name": "ThinkPad X1 Carbon", "price": 1599.99}
        ]
    }

def get_product_details(product_id):
    products = {
        "laptop-001": {
            "id": "laptop-001",
            "name": "MacBook Pro 16-inch",
            "price": 2399.99,
            "description": "16-inch Retina display, Apple M3 Pro chip, 32GB RAM, 1TB SSD",
            "rating": 4.8,
            "reviews": 1250
        },
        "laptop-002": {
            "id": "laptop-002",
            "name": "Dell XPS 15",
            "price": 1799.99,
            "description": "15.6-inch 4K UHD+ display, Intel Core i9, 32GB RAM, 1TB SSD",
            "rating": 4.6,
            "reviews": 980
        },
        "laptop-003": {
            "id": "laptop-003",
            "name": "ThinkPad X1 Carbon",
            "price": 1599.99,
            "description": "14-inch 4K display, Intel Core i7, 16GB RAM, 512GB SSD",
            "rating": 4.7,
            "reviews": 1050
        }
    }
    
    return products.get(product_id, {"error": f"Product not found: {product_id}"})

def add_to_cart(product_id, quantity):
    return {
        "success": True,
        "message": f"Added {quantity} of product {product_id} to cart",
        "cart_total": {
            "items": quantity,
            "price": get_product_details(product_id).get("price", 0) * quantity
        }
    }

# Process the multi-step tool calls
final_response, conversation_history = process_multi_step_tool_calls(response)

# Print the final response
print("\nFinal Response:")
for content_block in final_response.content:
    if content_block.type == "text":
        print(content_block.text)
```

## Best Practices

### Security

1. **Use OAuth tokens** for authenticated MCP servers
2. **Validate server URLs** before connecting
3. **Set appropriate timeouts** to prevent hanging requests
4. **Use HTTPS** for all MCP server connections
5. **Implement rate limiting** to prevent abuse

### Performance

1. **Set appropriate timeouts** to prevent long-running requests
2. **Cache tool responses** when appropriate
3. **Use streaming** for long-running tool calls
4. **Batch tool calls** when possible
5. **Monitor tool call performance** to identify bottlenecks

### Reliability

1. **Implement error handling** for tool calls
2. **Retry failed tool calls** with exponential backoff
3. **Provide fallback options** for unavailable tools
4. **Log tool call errors** for debugging
5. **Monitor tool availability** to detect outages

### User Experience

1. **Provide clear instructions** for tool usage
2. **Show progress indicators** for long-running tool calls
3. **Handle partial results** gracefully
4. **Provide fallback responses** when tools fail
5. **Explain tool capabilities** to users

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Check that OAuth tokens are valid and not expired
   - Verify that the authentication type is correct
   - Ensure that the token has the necessary permissions

2. **Connection Errors**
   - Verify that the server URL is correct and accessible
   - Check network connectivity to the MCP server
   - Ensure that the MCP server is running and healthy

3. **Timeout Errors**
   - Increase the timeout value for long-running tool calls
   - Optimize tool performance to reduce response time
   - Consider using streaming for long-running tool calls

4. **Tool Call Errors**
   - Check that the tool name is correct and matches the server's tools
   - Verify that the tool input matches the expected schema
   - Ensure that the tool is available on the MCP server

5. **Response Parsing Errors**
   - Verify that the tool output is valid JSON
   - Check that the tool output matches the expected schema
   - Ensure that the tool output is not too large

### Debugging Tips

1. **Enable verbose logging** to see detailed request and response information
2. **Use a network proxy** to inspect MCP server communication
3. **Test tools individually** to isolate issues
4. **Check server logs** for error messages
5. **Verify API versions** to ensure compatibility

## Resources

- [Anthropic API Documentation](https://docs.anthropic.com/en/api/overview)
- [MCP Connector Documentation](https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector)
- [Remote MCP Servers](https://docs.anthropic.com/en/docs/agents-and-tools/remote-mcp-servers)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
