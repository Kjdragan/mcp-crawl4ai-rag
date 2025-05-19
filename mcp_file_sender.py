#!/usr/bin/env python3
"""
Simple script to send a local file to the Crawl4AI MCP server.
"""
import argparse
import base64
import json
import os
import requests
import sys

def main():
    parser = argparse.ArgumentParser(description="Send a local file to Crawl4AI MCP")
    parser.add_argument("file_path", help="Path to the local file to process")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for processing")
    parser.add_argument("--mcp-url", default="http://localhost:8051", help="URL of the MCP server")
    
    args = parser.parse_args()
    
    # Read and encode the file
    with open(args.file_path, 'rb') as f:
        content = f.read()
    encoded_content = base64.b64encode(content).decode('utf-8')
    
    # Get filename
    filename = os.path.basename(args.file_path)
    
    print(f"File: {filename}")
    print(f"Encoded size: {len(encoded_content)} bytes")
    
    # Prepare payload
    payload = {
        "name": "crawl_local_file",
        "input": {
            "file_content": encoded_content,
            "filename": filename,
            "chunk_size": args.chunk_size
        }
    }
    
    # Try to send to MCP server
    try:
        # Standard MCP endpoint
        url = f"{args.mcp_url}/api/tools/run"
        print(f"Sending to {url}...")
        
        response = requests.post(
            url, 
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("SUCCESS! Response:")
                print(json.dumps(result, indent=2))
            except:
                print(f"Response (not JSON): {response.text[:1000]}...")
        else:
            print(f"Error: {response.text[:1000]}...")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
