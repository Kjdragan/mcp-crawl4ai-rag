#!/usr/bin/env python3
"""
Example script demonstrating how to use Anthropic's Claude API with MCP (Model Context Protocol)
and the Crawl4AI RAG system.

This script shows how to:
1. Set up the Anthropic API client
2. Create a message with MCP tools
3. Process the response and handle tool calls

Requirements:
- anthropic Python package
- requests

Install with: uv add anthropic requests
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional

# Use the anthropic package for API calls
from anthropic import Anthropic

# MCP server configuration
MCP_SERVER_URL = "http://localhost:8051"  # Default Crawl4AI MCP server URL

class MCPTool:
    """Represents an MCP tool configuration for the Anthropic API."""
    
    def __init__(self, name: str, description: str, server_name: str = "crawl4ai-rag"):
        self.name = name
        self.description = description
        self.server_name = server_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool to the format expected by Anthropic's API."""
        return {
            "name": f"mcp__{self.server_name}__{self.name}",
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }


def create_mcp_connector(server_url: str) -> Dict[str, Any]:
    """Create an MCP connector configuration for the Anthropic API."""
    return {
        "type": "mcp",
        "servers": [
            {
                "name": "crawl4ai-rag",
                "url": server_url
            }
        ]
    }


def handle_tool_calls(client: Anthropic, message_response: Dict[str, Any]) -> Dict[str, Any]:
    """Handle any tool calls in the message response."""
    content = message_response.get("content", [])
    
    for item in content:
        if item.get("type") == "tool_use":
            tool_call = item.get("tool_use", {})
            tool_name = tool_call.get("name", "")
            tool_input = tool_call.get("input", {})
            
            # Check if this is an MCP tool call
            if tool_name.startswith("mcp__"):
                # Parse the MCP tool name
                parts = tool_name.split("__")
                if len(parts) == 3:
                    _, server_name, actual_tool_name = parts
                    
                    # Make the request to the MCP server
                    response = call_mcp_server(
                        server_name=server_name,
                        tool_name=actual_tool_name,
                        parameters=tool_input
                    )
                    
                    # Send the tool response back to Claude
                    return client.messages.create(
                        model="claude-3-opus-20240229",
                        messages=[
                            {"role": "user", "content": message_response["content"]},
                            {"role": "assistant", "content": [item]},
                            {"role": "user", "content": f"Tool response: {json.dumps(response)}"}
                        ],
                        max_tokens=1024,
                        tools=get_mcp_tools()
                    )
    
    return message_response


def call_mcp_server(server_name: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Call the MCP server with the given tool name and parameters."""
    # In a real implementation, you would look up the server URL from a configuration
    server_url = MCP_SERVER_URL
    
    # Construct the endpoint URL
    endpoint = f"{server_url}/tools/{tool_name}"
    
    try:
        # Make the request to the MCP server
        response = requests.post(
            endpoint,
            json=parameters,
            headers={"Content-Type": "application/json"}
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse and return the response
        return response.json()
    except Exception as e:
        print(f"Error calling MCP server: {e}")
        return {"error": str(e)}


def get_mcp_tools() -> List[Dict[str, Any]]:
    """Get the list of MCP tools to include in the API request."""
    tools = [
        MCPTool(
            name="crawl_single_page",
            description="Crawl a single web page and store its content in the RAG system"
        ),
        MCPTool(
            name="smart_crawl_url",
            description="Intelligently crawl a URL or website and store content in the RAG system"
        ),
        MCPTool(
            name="perform_rag_query",
            description="Search the RAG system for relevant information"
        ),
        MCPTool(
            name="get_available_sources",
            description="Get a list of all available sources in the RAG system"
        ),
        MCPTool(
            name="crawl_local_file",
            description="Process a local file and add it to the RAG system"
        )
    ]
    
    return [tool.to_dict() for tool in tools]


def main():
    """Main function to demonstrate the Anthropic API with MCP integration."""
    # Get API key from environment variable
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return
    
    # Initialize the Anthropic client
    client = Anthropic(api_key=api_key)
    
    # Create a message with MCP tools
    try:
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Can you search for information about Python programming best practices?"
                }
            ],
            tools=get_mcp_tools(),
            tool_choice="auto",
            connectors=[create_mcp_connector(MCP_SERVER_URL)]
        )
        
        # Handle any tool calls in the response
        processed_response = handle_tool_calls(client, response.model_dump())
        
        # Print the final response
        print(json.dumps(processed_response, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
