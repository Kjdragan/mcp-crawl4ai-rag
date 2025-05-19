#!/usr/bin/env python3
"""
Helper script to encode local files and send them to the Crawl4AI MCP server.
"""
import argparse
import base64
import json
import os
import requests
import sys
from sseclient import SSEClient  # Import SSEClient for Server-Sent Events support

def encode_file(file_path):
    """Read a file and encode its content as base64."""
    with open(file_path, 'rb') as f:
        content = f.read()
    return base64.b64encode(content).decode('utf-8')

def process_local_file(file_path, chunk_size=5000, mcp_url="http://localhost:8051"):
    """Process a local file using the Crawl4AI MCP server with SSE support."""
    # Extract filename from path
    filename = os.path.basename(file_path)
    
    # Read and encode the file
    encoded_content = encode_file(file_path)
    print(f"File encoded successfully: {len(encoded_content)} bytes")
    
    # Prepare the tool request
    tool_request = {
        "tool": "crawl_local_file",
        "args": {
            "file_content": encoded_content,
            "filename": filename,
            "chunk_size": chunk_size
        }
    }
    
    # Convert to JSON
    tool_request_json = json.dumps(tool_request)
    
    # Set headers for SSE connection
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
    }
    
    try:
        print("Connecting to MCP server via SSE...")
        
        # First try direct tool endpoint with SSE
        sse_url = f"{mcp_url}/run_sse"
        response = requests.post(sse_url, data=tool_request_json, headers=headers, stream=True)
        
        if response.status_code == 404:
            # Try alternative endpoint
            print("Endpoint not found, trying alternative...")
            sse_url = f"{mcp_url}/api/run_sse"
            response = requests.post(sse_url, data=tool_request_json, headers=headers, stream=True) 
            
            if response.status_code == 404:
                # Try one more endpoint format
                print("Endpoint still not found, trying one more format...")
                # Adapt for MCP protocol that might be using a different format
                adapted_request = {
                    "name": "crawl_local_file",
                    "input": {
                        "file_content": encoded_content,
                        "filename": filename,
                        "chunk_size": chunk_size
                    }
                }
                sse_url = f"{mcp_url}/api/tools/run"
                response = requests.post(sse_url, json=adapted_request, headers={
                    'Content-Type': 'application/json'
                })
                
                # Process non-SSE response
                if response.status_code == 200:
                    try:
                        result = response.json()
                        print("Success! File processed:")
                        print(json.dumps(result, indent=2))
                        return True
                    except json.JSONDecodeError:
                        print(f"Error: Received a 200 response but couldn't parse JSON.")
                        print(f"Response: {response.text[:500]}...")
                        return False
                else:
                    print(f"Error: HTTP {response.status_code}")
                    print(f"Response: {response.text[:500]}...")
                    return False
        
        # Process SSE response
        if response.status_code == 200:
            print("Connected successfully to SSE endpoint. Processing events...")
            
            # Process the SSE events
            client = SSEClient(response)
            for event in client.events():
                if event.event == 'error':
                    print(f"Error from server: {event.data}")
                    return False
                elif event.event == 'complete':
                    try:
                        result = json.loads(event.data)
                        print("Processing completed successfully!")
                        print(json.dumps(result, indent=2))
                        return True
                    except json.JSONDecodeError:
                        print(f"Error parsing completion data: {event.data}")
                        return False
                else:
                    # Progress updates
                    print(f"Progress update: {event.data}")
            
            return True
        else:
            print(f"Error: HTTP {response.status_code}")
            print(f"Response: {response.text[:500]}...")
            return False
            
    except Exception as e:
        print(f"Error connecting to MCP server: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process local files with Crawl4AI MCP")
    parser.add_argument("file_path", help="Path to the local file to process")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for processing (default: 5000)")
    parser.add_argument("--mcp-url", default="http://localhost:8051", help="URL of the MCP server")
    
    args = parser.parse_args()
    
    # Process the file
    success = process_local_file(args.file_path, args.chunk_size, args.mcp_url)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
