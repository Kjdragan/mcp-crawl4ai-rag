#!/usr/bin/env python3
"""
Example script demonstrating how to use Anthropic's Message Batches API.

This example shows how to:
1. Create a message batch with multiple requests
2. Poll for batch completion
3. Retrieve and process batch results

Requirements:
- anthropic Python package

Install with: uv add anthropic
"""

import os
import time
import json
from typing import Dict, Any, List, Optional
import requests

# Import the Anthropic client
from anthropic import Anthropic


def create_message_batch(client: Anthropic, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a message batch with multiple requests.
    
    Args:
        client: Anthropic client
        requests: List of request objects with custom_id and params
        
    Returns:
        The message batch response
    """
    try:
        batch = client.messages.batches.create(
            requests=requests
        )
        return batch
    except Exception as e:
        print(f"Error creating message batch: {e}")
        return {"error": str(e)}


def poll_for_batch_completion(client: Anthropic, batch_id: str, poll_interval: int = 60) -> Dict[str, Any]:
    """
    Poll for message batch completion.
    
    Args:
        client: Anthropic client
        batch_id: The ID of the message batch
        poll_interval: Time in seconds between polling attempts
        
    Returns:
        The completed message batch
    """
    print(f"Polling for batch {batch_id} completion...")
    
    while True:
        try:
            message_batch = client.messages.batches.retrieve(batch_id)
            
            # Print current status
            print(f"Batch status: {message_batch.processing_status}")
            print(f"Request counts: {message_batch.request_counts}")
            
            if message_batch.processing_status == "ended":
                print(f"Batch {batch_id} processing has ended.")
                return message_batch
                
            print(f"Batch {batch_id} is still processing... Checking again in {poll_interval} seconds.")
            time.sleep(poll_interval)
            
        except Exception as e:
            print(f"Error polling batch {batch_id}: {e}")
            return {"error": str(e)}


def retrieve_batch_results(client: Anthropic, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Retrieve and process batch results.
    
    Args:
        client: Anthropic client
        batch: The completed message batch
        
    Returns:
        List of results from the batch
    """
    results_url = batch.get("results_url")
    if not results_url:
        print("No results URL available.")
        return []
    
    print(f"Retrieving results from {results_url}")
    
    try:
        # Get the API key for authentication
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        # Make a request to the results URL
        response = requests.get(
            results_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the JSONL response
        results = []
        for line in response.text.strip().split("\n"):
            if line:
                results.append(json.loads(line))
        
        return results
    
    except Exception as e:
        print(f"Error retrieving batch results: {e}")
        return []


def list_message_batches(client: Anthropic, limit: int = 10) -> List[Dict[str, Any]]:
    """
    List all message batches in the workspace.
    
    Args:
        client: Anthropic client
        limit: Maximum number of batches to retrieve
        
    Returns:
        List of message batches
    """
    try:
        batches = client.messages.batches.list(
            limit=limit
        )
        return batches
    except Exception as e:
        print(f"Error listing message batches: {e}")
        return []


def main():
    """Main function to demonstrate the Message Batches API."""
    # Get API key from environment variable
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return
    
    # Initialize the Anthropic client
    client = Anthropic(api_key=api_key)
    
    # Example batch requests
    batch_requests = [
        {
            "custom_id": "request-1",
            "params": {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Write a short poem about artificial intelligence."}
                ]
            }
        },
        {
            "custom_id": "request-2",
            "params": {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Explain quantum computing in simple terms."}
                ]
            }
        },
        {
            "custom_id": "request-3",
            "params": {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "What are the best practices for Python code documentation?"}
                ]
            }
        }
    ]
    
    # Create a message batch
    print("Creating message batch...")
    batch = create_message_batch(client, batch_requests)
    
    if "error" in batch:
        print(f"Failed to create batch: {batch['error']}")
        return
    
    print(f"Created batch with ID: {batch.id}")
    print(f"Initial status: {batch.processing_status}")
    
    # For demonstration purposes, we'll use a shorter polling interval
    # In production, you might want to use a longer interval
    poll_interval = 10  # seconds
    
    # Poll for batch completion
    completed_batch = poll_for_batch_completion(client, batch.id, poll_interval)
    
    if "error" in completed_batch:
        print(f"Error polling batch: {completed_batch['error']}")
        return
    
    # Retrieve and process batch results
    results = retrieve_batch_results(client, completed_batch)
    
    # Process and display results
    print("\nBatch Results:")
    print("==============")
    
    for result in results:
        custom_id = result.get("custom_id", "Unknown")
        content = result.get("content", [])
        
        print(f"\nResult for request '{custom_id}':")
        
        # Extract text content
        text_content = ""
        for item in content:
            if item.get("type") == "text":
                text_content += item.get("text", "")
        
        # Print a preview of the content
        preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
        print(f"Content preview: {preview}")
    
    # List recent message batches
    print("\nRecent Message Batches:")
    print("======================")
    
    batches = list_message_batches(client, limit=5)
    
    for batch in batches:
        print(f"Batch ID: {batch.id}")
        print(f"Created at: {batch.created_at}")
        print(f"Status: {batch.processing_status}")
        print(f"Request counts: {batch.request_counts}")
        print("---")


if __name__ == "__main__":
    main()
