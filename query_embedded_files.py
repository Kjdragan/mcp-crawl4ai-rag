#!/usr/bin/env python3
"""
Query embedded files directly from the Supabase database.
This bypasses the MCP server and directly searches the embedded content.
"""
import argparse
import json
import os
import sys

# Import necessary functions
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import search_documents, get_supabase_client, create_embedding

def query_knowledge_base(query, source=None, match_count=5):
    """
    Query the embedded documents directly in Supabase.
    
    Args:
        query: The search query
        source: Optional source to filter by (e.g., "google_ADK_Agent_Development_Kit")
        match_count: Maximum number of results to return
        
    Returns:
        List of matching documents
    """
    print(f"Querying: '{query}'")
    if source:
        print(f"Filtering by source: {source}")
    
    # Get Supabase client
    supabase = get_supabase_client()
    
    # Search for relevant documents
    results = search_documents(
        client=supabase,
        query=query,
        match_count=match_count,
        source=source
    )
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Query embedded documents in Crawl4AI RAG")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--source", help="Filter by source (document name without extension)")
    parser.add_argument("--match-count", type=int, default=5, help="Maximum number of results")
    
    args = parser.parse_args()
    
    # Perform query
    results = query_knowledge_base(args.query, args.source, args.match_count)
    
    # Process and display results
    print(f"\nFound {len(results)} matches:")
    print("-" * 80)
    
    for i, result in enumerate(results):
        # Extract metadata
        metadata = result.get("metadata", {})
        source = metadata.get("source", "Unknown")
        section = metadata.get("section", "")
        similarity = result.get("similarity", 0)
        
        print(f"Match #{i+1} (Similarity: {similarity:.2f})")
        print(f"Source: {source}")
        if section:
            print(f"Section: {section}")
        print("-" * 40)
        print(result.get("content", "")[:500] + "..." if len(result.get("content", "")) > 500 else result.get("content", ""))
        print("=" * 80)

if __name__ == "__main__":
    main()
