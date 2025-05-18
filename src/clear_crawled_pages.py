"""
Utility script to clear the crawled_pages table in Supabase.
This helps reset the database before re-indexing content.
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from supabase import create_client, Client

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Load environment variables from .env file in project root
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

def get_supabase_client() -> Client:
    """
    Get a Supabase client with the URL and key from environment variables.
    
    Returns:
        Supabase client instance
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")
    
    return create_client(url, key)

def clear_all_crawled_pages():
    """
    Clears all records from the crawled_pages table.
    """
    try:
        # Get Supabase client
        supabase_client = get_supabase_client()
        
        # Delete all records from the crawled_pages table
        # Using 'id > 0' as a condition that should match all records
        result = supabase_client.table('crawled_pages').delete().gt('chunk_number', -1).execute()
        
        print(f"Successfully cleared all records from the crawled_pages table.")
        return True
    except Exception as e:
        print(f"Error clearing crawled_pages table: {e}")
        return False

def clear_crawled_pages_by_source(source):
    """
    Clears records from the crawled_pages table that match a specific source.
    
    Args:
        source: The source to filter by (e.g., a domain name or file identifier)
    """
    try:
        # Get Supabase client
        supabase_client = get_supabase_client()
        
        # Delete records with matching source in metadata
        result = supabase_client.table('crawled_pages').delete().filter('metadata->>source', 'eq', source).execute()
        
        print(f"Successfully cleared records for source '{source}' from the crawled_pages table.")
        return True
    except Exception as e:
        print(f"Error clearing crawled_pages for source '{source}': {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments, clear all records
        clear_all_crawled_pages()
    elif len(sys.argv) == 2:
        # If one argument, clear records for that source
        source = sys.argv[1]
        clear_crawled_pages_by_source(source)
    else:
        print("Usage: python clear_crawled_pages.py [source]")
        print("If source is provided, only records matching that source will be cleared.")
        print("If no source is provided, all records will be cleared.")
