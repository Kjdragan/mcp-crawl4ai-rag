"""
Claude Sonnet 4 API Example

This script demonstrates how to use Anthropic's Claude Sonnet 4 model
for various natural language processing tasks.
"""

import os
import asyncio
import json
from dotenv import load_dotenv
import anthropic
from anthropic import Anthropic
from anthropic.types import MessageParam

# Load environment variables
load_dotenv()

# Get API key from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

def initialize_client():
    """Initialize the Anthropic client."""
    print("Initializing Anthropic client for Claude Sonnet 4")
    return Anthropic(api_key=ANTHROPIC_API_KEY)

def text_generation_example(client):
    """Example of text generation with Claude Sonnet 4."""
    print("\n=== Text Generation Example ===")
    
    # Create a message
    prompt = "Write a short story about a robot that learns to paint."
    print(f"Prompt: {prompt}")
    
    # Generate content
    message = client.messages.create(
        model="claude-3-sonnet-20240229",  # Use claude-3-5-sonnet-20240620 when available
        max_tokens=1024,
        temperature=0.7,
        system="You are a creative writing assistant that specializes in short stories.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    print("\nGenerated Story:")
    print(message.content[0].text)
    
    return message

def conversation_example(client):
    """Example of a multi-turn conversation with Claude Sonnet 4."""
    print("\n=== Conversation Example ===")
    
    # Start a conversation
    messages = [
        {"role": "user", "content": "I'm planning a trip to Japan. What are some must-visit places?"}
    ]
    
    print(f"User: {messages[0]['content']}")
    
    # First response
    response = client.messages.create(
        model="claude-3-sonnet-20240229",  # Use claude-3-5-sonnet-20240620 when available
        max_tokens=1024,
        temperature=0.7,
        messages=messages
    )
    
    assistant_response = response.content[0].text
    print(f"\nClaude: {assistant_response}")
    
    # Add the assistant's response to the conversation
    messages.append({"role": "assistant", "content": assistant_response})
    
    # Continue the conversation
    follow_up = "I'm particularly interested in historical sites and traditional culture. I'll be there for 10 days."
    messages.append({"role": "user", "content": follow_up})
    
    print(f"\nUser: {follow_up}")
    
    # Get the next response
    response = client.messages.create(
        model="claude-3-sonnet-20240229",  # Use claude-3-5-sonnet-20240620 when available
        max_tokens=1024,
        temperature=0.7,
        messages=messages
    )
    
    print(f"\nClaude: {response.content[0].text}")
    
    return messages

def structured_output_example(client):
    """Example of generating structured JSON output with Claude Sonnet 4."""
    print("\n=== Structured Output Example ===")
    
    # Define the structure we want
    system_prompt = """
    You are an assistant that provides information in structured JSON format.
    Always respond with valid JSON that follows the requested schema.
    Do not include any explanatory text outside the JSON structure.
    """
    
    # Request for book recommendations
    prompt = """
    Provide 3 book recommendations for science fiction novels about time travel.
    Format the response as a JSON array with objects having the following structure:
    {
        "title": "Book title",
        "author": "Author name",
        "year_published": year as number,
        "summary": "Brief summary of the book",
        "themes": ["theme1", "theme2", ...]
    }
    """
    
    print(f"Prompt: {prompt}")
    
    # Generate structured output
    response = client.messages.create(
        model="claude-3-sonnet-20240229",  # Use claude-3-5-sonnet-20240620 when available
        max_tokens=1024,
        temperature=0.2,  # Lower temperature for more deterministic output
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Parse the JSON response
    try:
        # Extract the text content
        response_text = response.content[0].text
        
        # Parse the JSON
        json_response = json.loads(response_text)
        
        print("\nStructured Book Recommendations:")
        print(json.dumps(json_response, indent=2))
        
        return json_response
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print("Raw response:", response_text)
        return None

async def streaming_example(client):
    """Example of streaming responses with Claude Sonnet 4."""
    print("\n=== Streaming Example ===")
    
    prompt = "Explain the concept of quantum computing to a high school student."
    print(f"Prompt: {prompt}")
    
    print("\nStreaming response:")
    
    # Create a streaming response
    with client.messages.stream(
        model="claude-3-sonnet-20240229",  # Use claude-3-5-sonnet-20240620 when available
        max_tokens=1024,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ]
    ) as stream:
        # Process the streaming response
        for text in stream.text_stream:
            print(text, end="", flush=True)
    
    print("\n\nStreaming complete!")

def tool_use_example(client):
    """Example of tool use with Claude Sonnet 4."""
    print("\n=== Tool Use Example ===")
    
    # Define a tool for weather information
    weather_tool = {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g., 'San Francisco, CA, USA'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit"
                }
            },
            "required": ["location"]
        }
    }
    
    # Define the tool use handler
    def tool_handler(tool_name, tool_input):
        """Handle tool calls."""
        if tool_name == "get_weather":
            location = tool_input.get("location")
            unit = tool_input.get("unit", "celsius")
            
            # In a real implementation, you would call a weather API here
            print(f"Called get_weather with location={location}, unit={unit}")
            
            # Mock response
            return {
                "location": location,
                "temperature": "22" if unit == "celsius" else "72",
                "unit": unit,
                "condition": "Sunny",
                "humidity": "45%"
            }
        
        return {"error": f"Unknown tool: {tool_name}"}
    
    # Prompt that should trigger tool use
    prompt = "What's the weather like in Chicago right now?"
    print(f"Prompt: {prompt}")
    
    # Create a message with tools
    message = client.messages.create(
        model="claude-3-sonnet-20240229",  # Use claude-3-5-sonnet-20240620 when available
        max_tokens=1024,
        temperature=0.7,
        tools=[weather_tool],
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Check if tool use was triggered
    if hasattr(message, "content") and any(content.type == "tool_use" for content in message.content):
        for content in message.content:
            if content.type == "tool_use":
                tool_use = content.tool_use
                print(f"\nTool use detected: {tool_use.name}")
                print(f"Tool input: {json.dumps(tool_use.input, indent=2)}")
                
                # Execute the tool
                tool_result = tool_handler(tool_use.name, tool_use.input)
                
                # Send the tool result back to Claude
                message = client.messages.create(
                    model="claude-3-sonnet-20240229",  # Use claude-3-5-sonnet-20240620 when available
                    max_tokens=1024,
                    temperature=0.7,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": [content]},
                        {"role": "user", "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "result": json.dumps(tool_result)
                            }
                        ]}
                    ]
                )
                
                print("\nFinal response with tool results:")
                print(message.content[0].text)
                return message
    else:
        print("\nNo tool use detected. Response:")
        print(message.content[0].text)
        return message

async def main():
    """Run the Claude Sonnet 4 examples."""
    # Initialize the client
    client = initialize_client()
    
    # Run examples
    text_generation_example(client)
    conversation_example(client)
    structured_output_example(client)
    await streaming_example(client)
    tool_use_example(client)
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    asyncio.run(main())
