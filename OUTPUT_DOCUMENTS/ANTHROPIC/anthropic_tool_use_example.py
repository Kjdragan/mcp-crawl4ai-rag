#!/usr/bin/env python3
"""
Example script demonstrating how to use Anthropic's Claude API with tool use.

This example shows how to:
1. Set up the Anthropic client
2. Define tools for Claude to use
3. Create a message with tools
4. Process tool use responses
5. Implement a complete tool use loop

Requirements:
- anthropic Python package

Install with: uv add anthropic
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Callable

from anthropic import Anthropic


def setup_client() -> Anthropic:
    """
    Set up the Anthropic client with API key from environment variable.
    
    Returns:
        The Anthropic client
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    return Anthropic(api_key=api_key)


def define_tools() -> List[Dict[str, Any]]:
    """
    Define tools for Claude to use.
    
    Returns:
        List of tool definitions
    """
    return [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature to return"
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "search_database",
            "description": "Search a database for information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "calculate",
            "description": "Perform a calculation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    ]


def create_message_with_tools(
    client: Anthropic,
    prompt: str,
    tools: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    model: str = "claude-3-opus-20240229",
    max_tokens: int = 1024
) -> Dict[str, Any]:
    """
    Create a message with tools using Claude.
    
    Args:
        client: Anthropic client
        prompt: User message
        tools: List of tool definitions
        system_prompt: Optional system prompt
        model: Claude model to use
        max_tokens: Maximum tokens to generate
        
    Returns:
        The response from Claude
    """
    # Prepare message parameters
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "tools": tools
    }
    
    # Add system prompt if provided
    if system_prompt:
        params["system"] = system_prompt
    
    # Create the message
    return client.messages.create(**params)


def process_tool_use_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process a response from Claude that may contain tool use requests.
    
    Args:
        response: Response from Claude
        
    Returns:
        List of tool use requests
    """
    tool_uses = []
    
    # Check if the response has content
    if hasattr(response, "content"):
        # Iterate through content blocks
        for content_block in response.content:
            # Check if the content block is a tool use
            if content_block.type == "tool_use":
                tool_use = {
                    "id": content_block.id,
                    "name": content_block.tool_use.name,
                    "input": content_block.tool_use.input
                }
                tool_uses.append(tool_use)
    
    return tool_uses


def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool based on its name and input.
    
    Args:
        tool_name: Name of the tool to execute
        tool_input: Input parameters for the tool
        
    Returns:
        The tool output
    """
    # Mock implementations for demonstration purposes
    if tool_name == "get_weather":
        location = tool_input.get("location", "Unknown")
        unit = tool_input.get("unit", "celsius")
        
        # In a real implementation, you would call a weather API here
        return {
            "temperature": 22 if unit == "celsius" else 72,
            "condition": "Sunny",
            "humidity": 45,
            "wind_speed": 10,
            "location": location,
            "unit": unit
        }
    
    elif tool_name == "search_database":
        query = tool_input.get("query", "")
        limit = tool_input.get("limit", 3)
        
        # In a real implementation, you would query a database here
        return {
            "results": [
                {"id": 1, "title": f"Result 1 for {query}", "content": "This is the first result"},
                {"id": 2, "title": f"Result 2 for {query}", "content": "This is the second result"},
                {"id": 3, "title": f"Result 3 for {query}", "content": "This is the third result"}
            ][:limit],
            "total_results": 100,
            "query": query
        }
    
    elif tool_name == "calculate":
        expression = tool_input.get("expression", "")
        
        try:
            # SECURITY WARNING: In a real implementation, you would need to
            # sanitize the expression to prevent code injection
            # This is just for demonstration purposes
            result = eval(expression)
            return {
                "expression": expression,
                "result": result
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e)
            }
    
    else:
        return {
            "error": f"Unknown tool: {tool_name}"
        }


def continue_conversation_with_tool_output(
    client: Anthropic,
    conversation_history: List[Dict[str, Any]],
    tool_outputs: List[Dict[str, Any]],
    model: str = "claude-3-opus-20240229",
    max_tokens: int = 1024
) -> Dict[str, Any]:
    """
    Continue the conversation with tool outputs.
    
    Args:
        client: Anthropic client
        conversation_history: Conversation history
        tool_outputs: Tool outputs to send to Claude
        model: Claude model to use
        max_tokens: Maximum tokens to generate
        
    Returns:
        The response from Claude
    """
    # Create the message with tool outputs
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=conversation_history,
        tool_outputs=tool_outputs
    )


def run_tool_use_loop(
    client: Anthropic,
    initial_prompt: str,
    tools: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    model: str = "claude-3-opus-20240229",
    max_tokens: int = 1024,
    max_iterations: int = 5
) -> List[Dict[str, Any]]:
    """
    Run a complete tool use loop with Claude.
    
    Args:
        client: Anthropic client
        initial_prompt: Initial user prompt
        tools: List of tool definitions
        system_prompt: Optional system prompt
        model: Claude model to use
        max_tokens: Maximum tokens to generate
        max_iterations: Maximum number of iterations
        
    Returns:
        The complete conversation history
    """
    # Initialize conversation history
    conversation_history = [
        {"role": "user", "content": initial_prompt}
    ]
    
    # Prepare message parameters
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": conversation_history,
        "tools": tools
    }
    
    # Add system prompt if provided
    if system_prompt:
        params["system"] = system_prompt
    
    # Run the tool use loop
    iterations = 0
    while iterations < max_iterations:
        # Create the message
        response = client.messages.create(**params)
        
        # Add Claude's response to the conversation history
        assistant_message = {"role": "assistant", "content": response.content}
        conversation_history.append(assistant_message)
        
        # Process tool use requests
        tool_uses = process_tool_use_response(response)
        
        # If no tools were used, we're done
        if not tool_uses:
            break
        
        # Execute tools and prepare tool outputs
        tool_outputs = []
        for tool_use in tool_uses:
            # Execute the tool
            tool_output = execute_tool(tool_use["name"], tool_use["input"])
            
            # Add the tool output
            tool_outputs.append({
                "tool_call_id": tool_use["id"],
                "output": json.dumps(tool_output)
            })
        
        # Add an empty user message with tool outputs
        conversation_history.append({"role": "user", "content": ""})
        
        # Update message parameters for the next iteration
        params = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": conversation_history,
            "tool_outputs": tool_outputs
        }
        
        iterations += 1
    
    return conversation_history


def main():
    """Main function to demonstrate Claude tool use."""
    # Set up the Anthropic client
    client = setup_client()
    
    # Define tools
    tools = define_tools()
    
    print("Example 1: Basic Tool Use")
    print("------------------------")
    
    # Create a message with tools
    prompt = "What's the weather like in San Francisco and New York? Also, calculate 15% of 85."
    system_prompt = "You are a helpful AI assistant that uses tools when appropriate."
    
    response = create_message_with_tools(
        client=client,
        prompt=prompt,
        tools=tools,
        system_prompt=system_prompt
    )
    
    # Process tool use requests
    tool_uses = process_tool_use_response(response)
    
    # Print Claude's response
    print("Claude's response:")
    for content_block in response.content:
        if content_block.type == "text":
            print(content_block.text)
        elif content_block.type == "tool_use":
            print(f"\nTool use requested: {content_block.tool_use.name}")
            print(f"Input: {json.dumps(content_block.tool_use.input, indent=2)}")
    
    # Execute tools and prepare tool outputs
    if tool_uses:
        print("\nExecuting tools:")
        tool_outputs = []
        for tool_use in tool_uses:
            print(f"- Executing {tool_use['name']} with input: {json.dumps(tool_use['input'], indent=2)}")
            tool_output = execute_tool(tool_use["name"], tool_use["input"])
            print(f"  Output: {json.dumps(tool_output, indent=2)}")
            
            tool_outputs.append({
                "tool_call_id": tool_use["id"],
                "output": json.dumps(tool_output)
            })
        
        # Continue the conversation with tool outputs
        conversation_history = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": ""}
        ]
        
        print("\nContinuing conversation with tool outputs...")
        
        continued_response = continue_conversation_with_tool_output(
            client=client,
            conversation_history=conversation_history,
            tool_outputs=tool_outputs
        )
        
        print("\nClaude's response after tool execution:")
        for content_block in continued_response.content:
            if content_block.type == "text":
                print(content_block.text)
    
    print("\n\nExample 2: Complete Tool Use Loop")
    print("-------------------------------")
    
    # Run a complete tool use loop
    initial_prompt = "I need to know the weather in Chicago and Miami. Also, search for information about machine learning."
    
    print(f"Initial prompt: {initial_prompt}")
    print("Running tool use loop...")
    
    conversation_history = run_tool_use_loop(
        client=client,
        initial_prompt=initial_prompt,
        tools=tools,
        system_prompt=system_prompt
    )
    
    # Print the complete conversation
    print("\nComplete conversation:")
    for i, message in enumerate(conversation_history):
        role = message["role"]
        
        if role == "user":
            if i == 0 or message["content"]:
                print(f"\nUser: {message['content']}")
            else:
                print("\nUser: [Tool outputs provided]")
        
        elif role == "assistant":
            print("\nClaude:")
            for content_block in message["content"]:
                if content_block.type == "text":
                    print(content_block.text)
                elif content_block.type == "tool_use":
                    print(f"[Using tool: {content_block.tool_use.name}]")
                    print(f"[Tool input: {json.dumps(content_block.tool_use.input, indent=2)}]")


if __name__ == "__main__":
    main()
