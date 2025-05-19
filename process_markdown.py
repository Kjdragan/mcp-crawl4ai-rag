#!/usr/bin/env python3
"""
Process local markdown files directly using the local_file_crawler module.
This approach bypasses the MCP API issues by using the underlying functions directly.
"""
import argparse
import json
import os
import sys

# Import the local file crawler
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from local_file_crawler import process_local_markdown_file

def main():
    parser = argparse.ArgumentParser(description="Process local markdown files for the RAG system")
    parser.add_argument("file_path", help="Path to the markdown file to process")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for processing")
    
    args = parser.parse_args()
    
    # Process the file
    print(f"Processing {args.file_path} with chunk size {args.chunk_size}...")
    result = process_local_markdown_file(args.file_path, args.chunk_size)
    
    # Display the result
    print(json.dumps(result, indent=2))
    
    # Return success/failure
    if result.get("success", False):
        print(f"✅ Success! Processed {result.get('chunk_count', 0)} chunks.")
        return 0
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
