#!/usr/bin/env python3
"""
Example script demonstrating how to use Anthropic's MCP connector feature.

This example shows how to:
1. Connect to remote MCP servers directly from the Messages API
2. Access MCP tools through the Messages API
3. Support OAuth Bearer tokens for authenticated servers
4. Connect to multiple MCP servers in a single request

Requirements:
- anthropic Python package

Install with: uv add anthropic
"""

import os
import json
from typing import Dict, Any, List, Optional

# Import the Anthropic client
from anthropic import Anthropic


def create_message_with_mcp_connector(
    api_key: str,
    user_message: str,
    mcp_servers: List[Dict[str, str]],
    model: str = "claude-3-opus-20240229"
) -> Dict[str, Any]:
    """
    Create a message using Anthropic's MCP connector feature.
    
    Args:
        api_key: Anthropic API key
        user_message: The message from the user
        mcp_servers: List of MCP server configurations
        model: The Claude model to use
        
    Returns:
        The response from Claude
    """
    # Initialize the Anthropic client
    client = Anthropic(api_key=api_key)
    
    # Create the message with MCP connector
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


def main():
    """Main function to demonstrate the Anthropic MCP connector."""
    # Get API key from environment variable
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return
    
    # Example user message that might require MCP tools
    user_message = "Can you search for information about Python best practices?"
    
    # Example MCP server configurations
    mcp_servers = [
        {
            "type": "url",
            "url": "https://mcp.example.com/sse",
            "name": "search-server",
            "authorization_token": "YOUR_OAUTH_TOKEN"  # Replace with actual token
        },
        {
            "type": "url",
            "url": "https://mcp.another-example.com/sse",
            "name": "rag-server"
            # No authorization token needed for this example server
        }
    ]
    
    try:
        # Make the API request with MCP connector
        headers = {
            "anthropic-beta": "mcp-client-2025-04-04"  # Required beta header
        }
        
        # Note: In a real implementation, you would pass the headers to the client
        # The Anthropic Python SDK may not yet support the beta header directly
        # This is a conceptual example based on the documentation
        print("Making API request with MCP connector...")
        
        # For demonstration purposes - in a real implementation, you would use:
        # client = Anthropic(api_key=api_key, headers=headers)
        response = create_message_with_mcp_connector(
            api_key=api_key,
            user_message=user_message,
            mcp_servers=mcp_servers
        )
        
        # Process the response
        print("\nResponse from Claude:")
        print(json.dumps(response, indent=2))
        
        # Extract and display any tool uses from the response
        content = response.get("content", [])
        for item in content:
            if item.get("type") == "tool_use":
                tool_use = item.get("tool_use", {})
                print(f"\nTool used: {tool_use.get('name')}")
                print(f"Tool input: {json.dumps(tool_use.get('input'), indent=2)}")
            elif item.get("type") == "text":
                print(f"\nText response: {item.get('text')}")
        
    except Exception as e:
        print(f"Error: {e}")


# Example of handling MCP tool results
def handle_mcp_tool_results(response: Dict[str, Any], client: Anthropic) -> Dict[str, Any]:
    """
    Handle MCP tool results from Claude's response.
    
    Args:
        response: The response from Claude
        client: The Anthropic client
        
    Returns:
        The updated response after handling tool results
    """
    content = response.get("content", [])
    
    # Check if there are any tool uses in the response
    tool_uses = []
    for item in content:
        if item.get("type") == "tool_use":
            tool_uses.append(item)
    
    if not tool_uses:
        return response
    
    # In a real implementation, you would:
    # 1. Extract the tool use details
    # 2. Call the appropriate MCP server to get the tool result
    # 3. Send the tool result back to Claude
    
    # This is a simplified example
    tool_results = []
    for tool_use in tool_uses:
        # Simulate getting a tool result
        tool_name = tool_use.get("tool_use", {}).get("name")
        tool_results.append({
            "type": "tool_result",
            "tool_result": {
                "tool_use_id": tool_use.get("tool_use", {}).get("id"),
                "content": f"Simulated result for {tool_name}"
            }
        })
    
    # Send the tool results back to Claude
    updated_response = client.messages.create(
        model=response.get("model"),
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Initial query"
            },
            {
                "role": "assistant",
                "content": content
            },
            {
                "role": "user",
                "content": tool_results
            }
        ]
    )
    
    return updated_response.model_dump()


if __name__ == "__main__":
    main()
