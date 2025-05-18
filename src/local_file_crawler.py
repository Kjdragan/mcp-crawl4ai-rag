"""
Helper script to directly process and index local markdown files.
This bypasses some of the issues with the MCP crawler's file URL handling.
"""

import os
import sys
from pathlib import Path
import json

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import necessary functions from crawl4ai_mcp.py
from crawl4ai_mcp import (
    normalize_file_path, 
    process_markdown_file,
    smart_chunk_markdown,
    extract_section_info,
    get_supabase_client
)
from utils import add_documents_to_supabase

def process_local_markdown_file(file_path: str, chunk_size: int = 5000) -> dict:
    """
    Process a local markdown file and add it to Supabase for RAG.
    
    Args:
        file_path: Path to the markdown file
        chunk_size: Maximum size of each content chunk in characters (default: 5000)
        
    Returns:
        Dictionary with processing summary and storage information
    """
    try:
        # Check if the file exists and has .md extension
        if not os.path.exists(file_path):
            return {
                "success": False,
                "path": file_path,
                "error": "File not found"
            }
            
        if not file_path.lower().endswith('.md'):
            return {
                "success": False,
                "path": file_path,
                "error": "Only .md files are supported"
            }
            
        # Normalize the path
        norm_path = normalize_file_path(file_path)
        
        # Read the file content
        content = process_markdown_file(norm_path)
        if not content:
            return {
                "success": False,
                "path": file_path,
                "error": "Failed to read markdown file or file is empty"
            }
            
        # Get absolute path for consistent URL creation
        abs_path = os.path.abspath(norm_path) if '/' in norm_path else os.path.abspath(file_path)
        abs_path = abs_path.replace('\\', '/')
        file_url = f"file://{abs_path}"
        
        # Create file metadata
        filename = os.path.basename(abs_path)
        # Use the filename without extension as the source (for better search)
        source = os.path.splitext(filename)[0]  # Get filename without extension
        
        # Chunk the content
        chunks = smart_chunk_markdown(content, chunk_size=chunk_size)
        if not chunks:
            return {
                "success": False,
                "path": file_path,
                "error": "Failed to chunk markdown content"
            }
            
        # Prepare for Supabase
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            urls.append(file_url)
            chunk_numbers.append(i)
            contents.append(chunk)
            
            # Extract metadata
            meta = extract_section_info(chunk)
            meta["chunk"] = i
            meta["source"] = source
            meta["filename"] = filename
            meta["doc_type"] = "markdown"
            meta["local_file"] = True
            
            metadatas.append(meta)
            
        # Create URL to full document mapping for contextual embedding
        url_to_full_document = {file_url: content}
        
        # Get Supabase client
        supabase_client = get_supabase_client()
        
        # Add documents to Supabase
        add_documents_to_supabase(
            client=supabase_client,
            urls=urls,
            chunk_numbers=chunk_numbers,
            contents=contents,
            metadatas=metadatas,
            url_to_full_document=url_to_full_document,
            batch_size=20
        )
        
        return {
            "success": True,
            "path": file_path,
            "chunk_count": len(chunks),
            "total_characters": len(content),
            "source": source,
            "filename": filename
        }
        
    except Exception as e:
        return {
            "success": False,
            "path": file_path,
            "error": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python local_file_crawler.py <path_to_markdown_file> [chunk_size]")
        sys.exit(1)
        
    file_path = sys.argv[1]
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    
    result = process_local_markdown_file(file_path, chunk_size)
    print(json.dumps(result, indent=2))
