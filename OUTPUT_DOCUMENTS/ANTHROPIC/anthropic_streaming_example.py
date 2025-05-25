#!/usr/bin/env python3
"""
Example script demonstrating how to use Anthropic's Claude API with streaming.

This example shows how to:
1. Set up the Anthropic client
2. Create a streaming message request
3. Process the streaming response events
4. Handle different event types in the stream

Requirements:
- anthropic Python package

Install with: uv add anthropic
"""

import os
import json
from typing import Dict, Any, List, Optional, Generator

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


def create_streaming_message(
    client: Anthropic,
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "claude-3-opus-20240229",
    max_tokens: int = 1024
) -> Generator:
    """
    Create a streaming message using Claude.
    
    Args:
        client: Anthropic client
        prompt: User message
        system_prompt: Optional system prompt
        model: Claude model to use
        max_tokens: Maximum tokens to generate
        
    Returns:
        A generator that yields streaming events
    """
    # Prepare message parameters
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }
    
    # Add system prompt if provided
    if system_prompt:
        params["system"] = system_prompt
    
    # Create the streaming message
    return client.messages.create(**params)


def process_streaming_events(stream: Generator) -> Dict[str, Any]:
    """
    Process streaming events from Claude.
    
    Args:
        stream: Generator yielding streaming events
        
    Returns:
        The final response content
    """
    # Initialize variables to track the response
    full_response = ""
    current_content_block = ""
    content_blocks = []
    
    # Process each event in the stream
    for event in stream:
        event_type = getattr(event, "type", None)
        
        if event_type == "message_start":
            print("Message started")
            
        elif event_type == "content_block_start":
            print(f"Content block started: {event.content_block.type}")
            current_content_block = ""
            
        elif event_type == "content_block_delta":
            # Extract the text delta and add it to the current content block
            if event.delta.type == "text_delta":
                text_delta = event.delta.text
                current_content_block += text_delta
                full_response += text_delta
                # Print the text delta as it comes in
                print(text_delta, end="", flush=True)
                
        elif event_type == "content_block_stop":
            # Add the completed content block to the list
            content_blocks.append({
                "type": event.content_block.type,
                "text": current_content_block
            })
            
        elif event_type == "message_delta":
            # This event contains usage information
            if hasattr(event, "usage"):
                print(f"\nUsage: {event.usage.output_tokens} output tokens")
                
        elif event_type == "message_stop":
            print("\nMessage completed")
    
    print("\n")
    return {
        "full_response": full_response,
        "content_blocks": content_blocks
    }


def streaming_with_tool_use(
    client: Anthropic,
    prompt: str,
    tools: List[Dict[str, Any]],
    model: str = "claude-3-opus-20240229",
    max_tokens: int = 1024
) -> Generator:
    """
    Create a streaming message with tool use.
    
    Args:
        client: Anthropic client
        prompt: User message
        tools: List of tool definitions
        model: Claude model to use
        max_tokens: Maximum tokens to generate
        
    Returns:
        A generator that yields streaming events
    """
    # Create the streaming message with tools
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt}
        ],
        tools=tools,
        stream=True
    )


def process_tool_use_stream(stream: Generator) -> Dict[str, Any]:
    """
    Process streaming events from Claude with tool use.
    
    Args:
        stream: Generator yielding streaming events
        
    Returns:
        Dictionary containing tool use information and response content
    """
    # Initialize variables to track the response
    full_response = ""
    tool_uses = []
    current_tool_use = None
    
    # Process each event in the stream
    for event in stream:
        event_type = getattr(event, "type", None)
        
        if event_type == "content_block_start" and event.content_block.type == "tool_use":
            # Start of a tool use block
            current_tool_use = {
                "id": event.content_block.id,
                "name": event.content_block.tool_use.name,
                "input": event.content_block.tool_use.input
            }
            print(f"Tool use requested: {current_tool_use['name']}")
            print(f"Input: {json.dumps(current_tool_use['input'], indent=2)}")
            
        elif event_type == "content_block_stop" and current_tool_use:
            # End of a tool use block
            tool_uses.append(current_tool_use)
            current_tool_use = None
            
        elif event_type == "content_block_delta" and event.delta.type == "text_delta":
            # Text response
            text_delta = event.delta.text
            full_response += text_delta
            print(text_delta, end="", flush=True)
            
        elif event_type == "message_stop":
            print("\nMessage completed")
    
    print("\n")
    return {
        "full_response": full_response,
        "tool_uses": tool_uses
    }


def main():
    """Main function to demonstrate Claude streaming."""
    # Set up the Anthropic client
    client = setup_client()
    
    print("Example 1: Basic Streaming")
    print("-------------------------")
    
    # Create a streaming message
    prompt = "Explain the concept of streaming APIs and their benefits in 3-4 sentences."
    system_prompt = "You are a helpful AI assistant that provides clear, concise explanations."
    
    stream = create_streaming_message(
        client=client,
        prompt=prompt,
        system_prompt=system_prompt
    )
    
    # Process the streaming events
    response = process_streaming_events(stream)
    
    print("\nExample 2: Streaming with Tool Use")
    print("--------------------------------")
    
    # Define a simple weather tool
    weather_tool = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    ]
    
    # Create a streaming message with tool use
    tool_prompt = "What is the weather like in San Francisco?"
    
    tool_stream = streaming_with_tool_use(
        client=client,
        prompt=tool_prompt,
        tools=weather_tool
    )
    
    # Process the tool use streaming events
    tool_response = process_tool_use_stream(tool_stream)
    
    # In a real application, you would now:
    # 1. Execute the tool with the input from tool_response["tool_uses"]
    # 2. Get the tool output
    # 3. Continue the conversation with the tool output
    
    print("\nTool use information:")
    for tool_use in tool_response["tool_uses"]:
        print(f"Tool: {tool_use['name']}")
        print(f"Input: {json.dumps(tool_use['input'], indent=2)}")
    
    print("\nTo continue the conversation with tool results, you would:")
    print("1. Execute the tool with the input")
    print("2. Send the tool results back to Claude")
    print("3. Continue the conversation")


if __name__ == "__main__":
    main()
