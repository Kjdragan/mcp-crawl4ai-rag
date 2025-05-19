#!/usr/bin/env python3
"""
Simplified script to crawl a local markdown file via the MCP server.
"""
import argparse
import base64
import json
import os
import requests
import sys

def encode_file(file_path):
    """Read a file and encode its content as base64."""
    with open(file_path, 'rb') as f:
        content = f.read()
    return base64.b64encode(content).decode('utf-8')

def crawl_local_file(file_path, chunk_size=5000, mcp_url="http://localhost:8051"):
    """Process a local file using the MCP crawl_local_file tool."""
    # Read and encode the file
    encoded_content = encode_file(file_path)
    filename = os.path.basename(file_path)
    
    print(f"File encoded successfully: {len(encoded_content)} bytes")
    print(f"Sending file {filename} to MCP server...")
    
    # Format for standard MCP tool invocation
    payload = {
        "id": "1",  # Request ID
        "jsonrpc": "2.0",  # JSON-RPC version
        "method": "invoke_tool",
        "params": {
            "name": "crawl_local_file",
            "args": {
                "file_content": encoded_content,
                "filename": filename,
                "chunk_size": chunk_size
            }
        }
    }
    
    # Make the request
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        # Try standard JSON-RPC endpoint
        response = requests.post(f"{mcp_url}/jsonrpc", json=payload, headers=headers)
        
        if response.status_code == 404:
            # Try alternate endpoint
            print("Endpoint not found, trying direct tool access...")
            direct_payload = {
                "file_content": encoded_content,
                "filename": filename,
                "chunk_size": chunk_size
            }
            response = requests.post(f"{mcp_url}/api/tools/crawl_local_file", json=direct_payload, headers=headers)
        
        # Process the response
        if response.status_code == 200:
            try:
                result = response.json()
                print("Success! File processed!")
                print(json.dumps(result, indent=2))
                return True
            except json.JSONDecodeError:
                print(f"Received a non-JSON response: {response.text[:500]}...")
                return False
        else:
            print(f"Error: HTTP {response.status_code}")
            print(f"Response: {response.text[:500]}...")
            return False
            
    except Exception as e:
        print(f"Error making request: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process local files with Crawl4AI MCP")
    parser.add_argument("file_path", help="Path to the local file to process")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for processing")
    parser.add_argument("--mcp-url", default="http://localhost:8051", help="URL of the MCP server")
    
    args = parser.parse_args()
    
    # Process the file
    success = crawl_local_file(args.file_path, args.chunk_size, args.mcp_url)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
