#!/usr/bin/env python3
"""
Example script demonstrating how to connect to remote MCP servers with Anthropic's Claude API.

This example shows how to:
1. Connect to remote MCP servers using the MCP connector
2. Access tools from different providers (like Zapier, Stripe, etc.)
3. Process tool responses

Requirements:
- anthropic Python package

Install with: uv add anthropic
"""

import os
import json
from typing import Dict, Any, List, Optional

# Import the Anthropic client
from anthropic import Anthropic


def connect_to_remote_mcp_servers(
    api_key: str,
    user_message: str,
    model: str = "claude-3-opus-20240229"
) -> Dict[str, Any]:
    """
    Connect to remote MCP servers using Anthropic's MCP connector.
    
    Args:
        api_key: Anthropic API key
        user_message: The message from the user
        model: The Claude model to use
        
    Returns:
        The response from Claude
    """
    # Initialize the Anthropic client
    client = Anthropic(api_key=api_key)
    
    # Define remote MCP servers to connect to
    mcp_servers = [
        # Zapier MCP server example
        {
            "type": "url",
            "url": "https://mcp.zapier.com/",
            "name": "zapier-mcp"
            # No authorization token in this example, but would be required in production
        },
        # Stripe MCP server example
        {
            "type": "url",
            "url": "https://mcp.stripe.com/v1/sse",
            "name": "stripe-mcp"
            # No authorization token in this example, but would be required in production
        }
    ]
    
    # Required beta header for MCP connector
    headers = {
        "anthropic-beta": "mcp-client-2025-04-04"
    }
    
    # In a real implementation, you would pass the headers to the client
    # For demonstration purposes only
    print(f"Using beta header: {headers}")
    
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
        
        return response.model_dump()
    except Exception as e:
        print(f"Error connecting to remote MCP servers: {e}")
        return {"error": str(e)}


def handle_tool_responses(response: Dict[str, Any], client: Anthropic) -> Dict[str, Any]:
    """
    Handle tool responses from Claude's interaction with remote MCP servers.
    
    Args:
        response: The response from Claude
        client: The Anthropic client
        
    Returns:
        The updated response after handling tool responses
    """
    content = response.get("content", [])
    
    # Check for any tool uses in the response
    tool_uses = []
    for item in content:
        if item.get("type") == "tool_use":
            tool_uses.append(item)
    
    if not tool_uses:
        return response
    
    # In a real implementation, you would:
    # 1. Extract the tool use details
    # 2. Get the tool result (this would happen automatically with the MCP connector)
    # 3. Send the tool result back to Claude
    
    print(f"Found {len(tool_uses)} tool uses in the response")
    for i, tool_use in enumerate(tool_uses):
        tool_name = tool_use.get("tool_use", {}).get("name")
        tool_input = tool_use.get("tool_use", {}).get("input")
        print(f"Tool {i+1}: {tool_name}")
        print(f"Input: {json.dumps(tool_input, indent=2)}")
    
    return response


def main():
    """Main function to demonstrate connecting to remote MCP servers."""
    # Get API key from environment variable
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return
    
    # Example user message that might require tools from remote MCP servers
    user_message = "I need to create a Zap that triggers when a new payment is received in Stripe. Can you help me set this up?"
    
    print("Connecting to remote MCP servers...")
    response = connect_to_remote_mcp_servers(
        api_key=api_key,
        user_message=user_message
    )
    
    # Process the response
    print("\nResponse from Claude:")
    
    # Handle any tool responses
    response = handle_tool_responses(response, Anthropic(api_key=api_key))
    
    # Extract and display the text content
    content = response.get("content", [])
    for item in content:
        if item.get("type") == "text":
            print(f"\nText response: {item.get('text')}")
    
    print("\nAvailable Remote MCP Servers Examples:")
    print("1. Zapier - https://mcp.zapier.com/")
    print("2. Stripe - https://mcp.stripe.com/v1/sse")
    print("3. Square - https://mcp.squareup.com/sse")
    print("4. PayPal - https://mcp.paypal.com/sse")
    print("5. Plaid - https://api.dashboard.plaid.com/mcp/sse")
    print("6. Atlassian - https://mcp.atlassian.com/sse")
    print("7. Intercom - https://mcp.intercom.io/sse")


if __name__ == "__main__":
    main()
