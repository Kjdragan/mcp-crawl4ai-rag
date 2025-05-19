#!/usr/bin/env python
"""
Script to process the Google ADK markdown file using our MCP tool
"""

import base64
import json
import requests
import sys

# File path to process
file_path = "C:\\Users\\kevin\\repos\\TOOLS\\mcp-crawl4ai-rag\\AI DOC LIBRARY\\google_ADK_Agent_Development_Kit.md"
output_name = "google_ADK_Agent_Development_Kit.md"

# Read and encode the file
try:
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Base64 encode the content
    encoded_content = base64.b64encode(content).decode('utf-8')
    print(f"Successfully read and encoded file: {len(encoded_content)} bytes")
    
    # Prepare the payload for the MCP tool
    payload = {
        "name": "crawl_local_file",
        "input": {
            "file_content": encoded_content,
            "filename": output_name,
            "chunk_size": 5000
        }
    }
    
    # Call the MCP tool directly
    try:
        # This is the built-in MCP client in Cascade
        print("Sending request to the MCP tool...")
        
        # Use requests to make a direct API call
        response = requests.post("http://localhost:8051/api/v1/mcp/tools/invoke", 
                                json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
            print("\nSuccess! Document processed successfully.")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error calling MCP tool: {e}")
    
except Exception as e:
    print(f"Error reading or encoding file: {e}")
    sys.exit(1)
