#!/usr/bin/env python3
"""
Example script demonstrating how to use Anthropic's Claude API with batch processing.

This example shows how to:
1. Set up the Anthropic client
2. Create a message batch with multiple requests
3. Poll for batch completion
4. Retrieve and process batch results

Requirements:
- anthropic Python package

Install with: uv add anthropic
"""

import os
import time
import json
from typing import Dict, Any, List, Optional

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


def create_message_batch(
    client: Anthropic,
    requests: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create a message batch with multiple requests.
    
    Args:
        client: Anthropic client
        requests: List of message requests
        
    Returns:
        The batch response
    """
    try:
        # Create the message batch
        batch = client.messages.batches.create(
            requests=requests
        )
        
        return batch
    except Exception as e:
        print(f"Error creating message batch: {e}")
        return {"error": str(e)}


def poll_batch_completion(
    client: Anthropic,
    batch_id: str,
    max_attempts: int = 30,
    delay: int = 2
) -> Dict[str, Any]:
    """
    Poll for batch completion with exponential backoff.
    
    Args:
        client: Anthropic client
        batch_id: Batch ID to poll
        max_attempts: Maximum number of polling attempts
        delay: Initial delay between polling attempts in seconds
        
    Returns:
        The completed batch
    """
    attempts = 0
    current_delay = delay
    
    while attempts < max_attempts:
        # Get the batch status
        batch_status = client.messages.batches.retrieve(batch_id)
        
        # Check if the batch is complete
        if batch_status.status == "completed":
            print(f"Batch completed after {attempts + 1} attempts")
            return batch_status
        
        # Print the current status
        print(f"Batch status: {batch_status.status}, waiting {current_delay}s before next check...")
        
        # Wait before the next attempt with exponential backoff
        time.sleep(current_delay)
        current_delay = min(current_delay * 1.5, 10)  # Cap at 10 seconds
        attempts += 1
    
    raise TimeoutError(f"Batch {batch_id} did not complete within the expected time")


def retrieve_batch_results(
    client: Anthropic,
    batch_id: str
) -> Dict[str, Any]:
    """
    Retrieve results from a completed batch.
    
    Args:
        client: Anthropic client
        batch_id: Batch ID to retrieve results from
        
    Returns:
        Dictionary mapping custom IDs to message results
    """
    # Get the batch status
    batch_status = client.messages.batches.retrieve(batch_id)
    
    # Check if the batch is complete
    if batch_status.status != "completed":
        raise ValueError(f"Batch {batch_id} is not complete (status: {batch_status.status})")
    
    # Get the batch results
    results = {}
    for result in batch_status.results:
        custom_id = result.custom_id
        message = result.message
        results[custom_id] = message
    
    return results


def list_recent_batches(
    client: Anthropic,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    List recent message batches.
    
    Args:
        client: Anthropic client
        limit: Maximum number of batches to list
        
    Returns:
        List of recent batches
    """
    try:
        # List recent batches
        batches = client.messages.batches.list(
            limit=limit
        )
        
        return batches
    except Exception as e:
        print(f"Error listing batches: {e}")
        return []


def process_batch_results(results: Dict[str, Any]) -> None:
    """
    Process and display batch results.
    
    Args:
        results: Dictionary mapping custom IDs to message results
    """
    for custom_id, message in results.items():
        print(f"\n=== Result for {custom_id} ===")
        
        # Extract and display the content
        if hasattr(message, "content"):
            for content_block in message.content:
                if content_block.type == "text":
                    print(content_block.text)
        
        # Display usage information if available
        if hasattr(message, "usage"):
            print(f"\nUsage: {message.usage.input_tokens} input tokens, {message.usage.output_tokens} output tokens")
        
        print("=" * 50)


def main():
    """Main function to demonstrate Claude batch processing."""
    # Set up the Anthropic client
    client = setup_client()
    
    print("Creating a message batch with multiple requests...")
    
    # Prepare batch requests
    batch_requests = [
        {
            "custom_id": "request-1",
            "params": {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Write a short poem about artificial intelligence"}
                ]
            }
        },
        {
            "custom_id": "request-2",
            "params": {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Explain quantum computing in simple terms"}
                ]
            }
        },
        {
            "custom_id": "request-3",
            "params": {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Provide 3 tips for effective time management"}
                ],
                "system": "You are a productivity expert. Keep your answers concise and actionable."
            }
        }
    ]
    
    # Create the message batch
    batch = create_message_batch(client, batch_requests)
    
    if "error" in batch:
        print(f"Failed to create batch: {batch['error']}")
        return
    
    print(f"Batch created with ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Created at: {batch.created_at}")
    print(f"Number of requests: {len(batch_requests)}")
    
    # Poll for batch completion
    try:
        completed_batch = poll_batch_completion(client, batch.id)
        print("Batch processing completed!")
        
        # Retrieve and process batch results
        results = retrieve_batch_results(client, batch.id)
        process_batch_results(results)
        
        # Example of listing recent batches
        print("\nListing recent batches:")
        recent_batches = list_recent_batches(client, limit=5)
        
        for recent_batch in recent_batches:
            print(f"Batch ID: {recent_batch.id}, Status: {recent_batch.status}, Created: {recent_batch.created_at}")
        
    except TimeoutError as e:
        print(f"Timeout error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
