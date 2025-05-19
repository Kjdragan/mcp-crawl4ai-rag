#!/usr/bin/env python
"""
Helper script to crawl local files using the MCP server.
This script:
1. Reads a local file
2. Base64 encodes the content
3. Sends it to the MCP crawl_local_file tool
"""

import argparse
import base64
import json
import os
import sys
import requests

def read_and_encode_file(file_path):
    """Read a file and encode its content as base64."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        encoded_content = base64.b64encode(content).decode('utf-8')
        return encoded_content
    except Exception as e:
        print(f"Error reading or encoding file: {e}")
        sys.exit(1)

def crawl_local_file(file_path, chunk_size=5000, mcp_url="http://localhost:8051"):
    """Send the file to the MCP crawl_local_file tool."""
    # Read and encode the file
    encoded_content = read_and_encode_file(file_path)
    
    # Prepare the request
    filename = os.path.basename(file_path)
    payload = {
        "name": "crawl_local_file",
        "input": {
            "file_content": encoded_content,
            "filename": filename,
            "chunk_size": chunk_size
        }
    }
    
    # Send the request
    try:
        # MCP servers use a specific endpoint format
        response = requests.post(f"{mcp_url}/api/v1/mcp/tools/invoke", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error sending request to MCP server: {e}")
        return None

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Crawl a local file using the MCP server')
    parser.add_argument('file_path', help='Path to the local file to crawl')
    parser.add_argument('--chunk-size', type=int, default=5000, help='Size of chunks for processing')
    parser.add_argument('--mcp-url', default='http://localhost:8051', help='URL of the MCP server')
    
    args = parser.parse_args()
    
    # Crawl the file
    result = crawl_local_file(args.file_path, args.chunk_size, args.mcp_url)
    
    # Check the result
    if result and result.get("output", {}).get("success"):
        print("File successfully crawled!")
    else:
        print("Failed to crawl file.")

if __name__ == "__main__":
    main()
