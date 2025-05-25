#!/usr/bin/env python3
"""
Example script demonstrating how to use Anthropic's Claude models via Google Cloud's Vertex AI.

This example shows how to:
1. Set up the Anthropic Vertex AI client
2. Create a message using Claude on Vertex AI
3. Process the response

Requirements:
- anthropic Python package with Vertex AI support

Install with: uv add "anthropic[vertex]"
"""

import os
from typing import Dict, Any, List, Optional

# Import the Anthropic Vertex client
from anthropic import AnthropicVertex


def setup_vertex_client(project_id: str, region: str = "us-east5") -> AnthropicVertex:
    """
    Set up the Anthropic Vertex AI client.
    
    Args:
        project_id: Google Cloud project ID
        region: Google Cloud region where Claude is available
        
    Returns:
        The Anthropic Vertex client
    """
    # Initialize the Anthropic Vertex client
    # Note: This uses Google Cloud default authentication
    client = AnthropicVertex(
        project_id=project_id,
        region=region
    )
    
    return client


def create_message(
    client: AnthropicVertex,
    user_message: str,
    system_prompt: Optional[str] = None,
    model: str = "claude-opus-4@20250514",
    max_tokens: int = 1024
) -> Dict[str, Any]:
    """
    Create a message using Claude on Vertex AI.
    
    Args:
        client: Anthropic Vertex client
        user_message: The message from the user
        system_prompt: Optional system prompt
        model: The Claude model to use
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        The response from Claude
    """
    try:
        # Create the message parameters
        message_params = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        }
        
        # Add system prompt if provided
        if system_prompt:
            message_params["system"] = system_prompt
        
        # Create the message
        response = client.messages.create(**message_params)
        
        return response
    except Exception as e:
        print(f"Error creating message: {e}")
        return {"error": str(e)}


def process_response(response: Dict[str, Any]) -> str:
    """
    Process the response from Claude.
    
    Args:
        response: The response from Claude
        
    Returns:
        The extracted text content
    """
    if hasattr(response, "error"):
        return f"Error: {response['error']}"
    
    # Extract text content from the response
    content = response.content
    text_content = ""
    
    for item in content:
        if item.get("type") == "text":
            text_content += item.get("text", "")
    
    return text_content


def main():
    """Main function to demonstrate Claude on Vertex AI."""
    # Set your Google Cloud project ID
    # This should be the project where Claude is enabled
    project_id = "YOUR_GOOGLE_CLOUD_PROJECT_ID"
    
    # Set the region where Claude is available
    # Common regions include: us-east5, us-central1, europe-west4, asia-southeast1
    region = "us-east5"
    
    print(f"Setting up Anthropic Vertex AI client for project {project_id} in region {region}...")
    
    # Set up the Anthropic Vertex client
    client = setup_vertex_client(project_id, region)
    
    # Example user message
    user_message = "What are the key differences between traditional machine learning and deep learning approaches?"
    
    # Example system prompt
    system_prompt = "You are an AI assistant with expertise in machine learning and data science. Provide clear, concise explanations with practical examples when possible."
    
    print("Creating message using Claude on Vertex AI...")
    
    # Create a message
    response = create_message(
        client=client,
        user_message=user_message,
        system_prompt=system_prompt
    )
    
    # Process the response
    text_content = process_response(response)
    
    # Print the response
    print("\nResponse from Claude on Vertex AI:")
    print("=================================")
    print(text_content)
    
    # Example of a more complex message with multimodal content
    print("\nCreating a multimodal message...")
    
    # Note: Multimodal content requires base64 encoding of images
    # This is a simplified example without actual image content
    multimodal_message = "Explain the concept of transfer learning in deep neural networks."
    
    multimodal_response = create_message(
        client=client,
        user_message=multimodal_message,
        system_prompt="You are an AI assistant with expertise in deep learning. Provide detailed technical explanations with examples."
    )
    
    # Process the multimodal response
    multimodal_text = process_response(multimodal_response)
    
    # Print the multimodal response
    print("\nMultimodal Response from Claude on Vertex AI:")
    print("===========================================")
    print(multimodal_text)


if __name__ == "__main__":
    main()
