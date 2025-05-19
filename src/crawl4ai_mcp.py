"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
"""
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
import requests
import asyncio
import json
import os
import re
import traceback

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from utils import get_supabase_client, add_documents_to_supabase, search_documents

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Helper functions for local file processing
def normalize_file_path(path: str) -> str:
    """
    Normalize a file path for consistent handling across platforms.

    Args:
        path: File path to normalize

    Returns:
        Normalized file path
    """
    # Convert backslashes to forward slashes
    normalized_path = path.replace('\\', '/')

    # Handle spaces in the path if needed
    if ' ' in normalized_path:
        # On Windows, this isn't strictly necessary as Windows handles spaces in paths well
        # But it might be needed for certain operations
        pass

    return normalized_path

def is_file_path(path: str) -> bool:
    """
    Check if the given string represents a local file path.

    Args:
        path: String to check

    Returns:
        True if it appears to be a file path, False otherwise
    """
    try:
        # Normalize the path first
        norm_path = normalize_file_path(path)

        # Print debug information
        # print(f"Checking if path exists: {path}")
        # print(f"Normalized path: {norm_path}")

        # Check for absolute path patterns
        if os.path.isabs(norm_path):
            # print(f"Path is absolute: {norm_path}")
            if os.path.exists(norm_path):
                # print(f"Path exists: {norm_path}")
                return True
            # else:
                # print(f"Path does not exist: {norm_path}")

        # Try the original path as a fallback
        if os.path.isabs(path):
            # print(f"Original path is absolute: {path}")
            if os.path.exists(path):
                # print(f"Original path exists: {path}")
                return True
            # else:
                # print(f"Original path does not exist: {path}")

        # Simple check for relative paths that exist
        if os.path.exists(norm_path):
            # print(f"Normalized relative path exists: {norm_path}")
            return True
        if os.path.exists(path):
            # print(f"Original relative path exists: {path}")
            return True

        # Additional check for Windows paths with spaces
        if '\\' in path or '/' in path:
            # Looks like a path based on separators
            if path.lower().endswith('.md'):
                # print(f"Path looks like a file path with .md extension: {path}")
                return True

        return False
    except Exception as e:
        # print(f"Error checking if path exists: {e}")
        return False

def process_markdown_file(file_path: str) -> str:
    """
    Process a markdown file and return its content with consistent line endings.

    Args:
        file_path: Path to the markdown file

    Returns:
        str: Processed markdown content with consistent line endings
    """
    # First try with the normalized path
    norm_path = normalize_file_path(file_path)

    # Try different paths and encodings
    paths_to_try = [
        (norm_path, 'utf-8'),
        (file_path, 'utf-8'),
        (norm_path, 'latin-1'),
        (file_path, 'latin-1'),
        (norm_path, 'utf-16'),
        (file_path, 'utf-16'),
        (norm_path, 'cp1252'),
        (file_path, 'cp1252')
    ]

    # print(f"Attempting to read file: {file_path}")
    # print(f"Normalized path: {norm_path}")

    for path_to_try, encoding in paths_to_try:
        try:
            # print(f"Trying to read with path: {path_to_try}, encoding: {encoding}")
            if not os.path.exists(path_to_try):
                # print(f"Path does not exist: {path_to_try}")
                continue

            with open(path_to_try, 'r', encoding=encoding) as f:
                content = f.read()
                if not content.strip():
                    # print(f"File is empty or contains only whitespace: {path_to_try}")
                    continue

                # print(f"Successfully read file with encoding {encoding}: {path_to_try}")
                # Normalize line endings and remove any trailing whitespace
                return '\n'.join(line.rstrip() for line in content.splitlines())
        except FileNotFoundError:
            # print(f"File not found: {path_to_try}")
            continue
        except UnicodeDecodeError:
            # print(f"Unicode decode error with encoding {encoding}: {path_to_try}")
            continue
        except Exception as e:
            print(f"Error reading file {path_to_try}: {e}", file=sys.stderr)
            continue

    print(f"Failed to read file with any method: {file_path}", file=sys.stderr)
    return ""

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.
    
    Args:
        chunk: Markdown chunk
        
    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Client

@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.

    Args:
        server: The FastMCP server instance

    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Supabase client
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )

    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()

    # Initialize Supabase client
    supabase_client = get_supabase_client()

    try:
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client
        )
    finally:
        # Clean up the crawler
        await crawler.__aexit__(None, None, None)

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051")
)

def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.

    Args:
        url: URL to check

    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.

    Args:
        url: URL to check

    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')

def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.

    Args:
        sitemap_url: URL of the sitemap

    Returns:
        List of URLs found in the sitemap
    """
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Supabase.

    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Supabase for later retrieval and querying.

    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl

    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)

        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)

        if result.success and result.markdown:
            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown)

            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []

            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)

                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = urlparse(url).netloc
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)

            # Create url_to_full_document mapping
            url_to_full_document = {url: result.markdown}

            # Add to Supabase
            add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document)

            return json.dumps({
                "success": True,
                "url": url,
                "chunks_stored": len(chunks),
                "content_length": len(result.markdown),
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "error": result.error_message
            }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def smart_crawl_url(ctx: Context, url_or_path: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
    """
    Intelligently process a URL or file path and store content in Supabase.

    This tool automatically detects the input type and applies the appropriate method:
    - For local file paths: Directly reads and processes the file (currently supports .md files)
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth

    All content is chunked and stored in Supabase for later retrieval and querying.

    Args:
        ctx: The MCP server provided context
        url_or_path: URL or local file path to process
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters (default: 5000)

    Returns:
        JSON string with processing summary and storage information
    """
    try:
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        final_chunks_to_store: List[str] = []
        final_metadatas_to_store: List[Dict[str, Any]] = []
        processed_items_info_list: List[str] = []
        processed_items_count = 0
        crawl_type_summary = "unknown"
        url_to_full_document_map: Dict[str, str] = {}

        if is_file_path(url_or_path):
            if url_or_path.lower().endswith('.md'):
                content = process_markdown_file(url_or_path)
                if not content or not content.strip():
                    return json.dumps({
                        "success": False,
                        "url_or_path": url_or_path,
                        "error": "File is empty or could not be read"
                    }, indent=2)

                abs_path = os.path.abspath(url_or_path)
                normalized_abs_path = abs_path.replace('\\', '/')
                file_url = f"file://{normalized_abs_path}"
                filename = os.path.basename(abs_path)
                source_name = os.path.splitext(filename)[0]
                crawl_type_summary = "local_markdown"

                url_to_full_document_map[file_url] = content

                markdown_chunks = smart_chunk_markdown(content, chunk_size=chunk_size)
                for i, chunk_md in enumerate(markdown_chunks):
                    meta = extract_section_info(chunk_md)
                    meta.update({
                        'chunk_index': i,
                        'source': source_name,
                        'filename': filename,
                        'doc_type': 'markdown',
                        'local_file': True,
                        'file_path': normalized_abs_path,
                        'crawl_type': crawl_type_summary,
                        'url': file_url
                    })
                    final_chunks_to_store.append(chunk_md)
                    final_metadatas_to_store.append(meta)
                
                processed_items_info_list.append(file_url)
                processed_items_count = 1
            else:
                return json.dumps({
                    "success": False,
                    "url_or_path": url_or_path,
                    "error": "Only .md files are supported for local file paths"
                }, indent=2)
        else: # It's a URL
            web_crawl_results: List[Dict[str, Any]] = []
            if is_txt(url_or_path):
                web_crawl_results = await crawl_markdown_file(crawler, url_or_path)
                crawl_type_summary = "text_file"
            elif is_sitemap(url_or_path):
                sitemap_urls = parse_sitemap(url_or_path)
                if not sitemap_urls:
                    return json.dumps({"success": False, "url_or_path": url_or_path, "error": "No URLs found in sitemap"}, indent=2)
                web_crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
                crawl_type_summary = "sitemap"
            else:
                web_crawl_results = await crawl_recursive_internal_links(crawler, [url_or_path], max_depth=max_depth, max_concurrent=max_concurrent)
                crawl_type_summary = "webpage"

            if not web_crawl_results:
                return json.dumps({"success": False, "url_or_path": url_or_path, "error": "No content found from URL"}, indent=2)

            for doc_content in web_crawl_results:
                source_url = doc_content['url']
                markdown_content = doc_content['markdown']
                page_metadata = doc_content.get('metadata', {})

                url_to_full_document_map[source_url] = markdown_content

                markdown_chunks = smart_chunk_markdown(markdown_content, chunk_size=chunk_size)
                for i, chunk_md in enumerate(markdown_chunks):
                    meta = extract_section_info(chunk_md)
                    meta.update({
                        'chunk_index': i,
                        'source': urlparse(source_url).netloc,
                        'url': source_url,
                        'doc_type': 'web_markdown', # Content from web, processed into markdown
                        'local_file': False,
                        'crawl_type': crawl_type_summary,
                        'page_title': page_metadata.get('title', page_metadata.get('Title', '')) # Accommodate different casings for title
                    })
                    final_chunks_to_store.append(chunk_md)
                    final_metadatas_to_store.append(meta)
            
            processed_items_info_list = [doc['url'] for doc in web_crawl_results]
            processed_items_count = len(web_crawl_results)

        if not final_chunks_to_store:
            return json.dumps({
                "success": False,
                "url_or_path": url_or_path,
                "error": "No content chunks to process"
            }, indent=2)

        # Prepare arguments for add_documents_to_supabase
        doc_urls_for_supabase = [meta['url'] for meta in final_metadatas_to_store]
        chunk_indices_for_supabase = [meta['chunk_index'] for meta in final_metadatas_to_store]

        add_documents_to_supabase(
            supabase_client, 
            doc_urls_for_supabase, 
            chunk_indices_for_supabase, 
            final_chunks_to_store, 
            final_metadatas_to_store, 
            url_to_full_document_map
        )

        return json.dumps({
            "success": True,
            "url_or_path": url_or_path,
            "crawl_type": crawl_type_summary,
            "pages_crawled": processed_items_count, 
            "chunks_stored": len(final_chunks_to_store),
            "urls_crawled": processed_items_info_list[:5] + (["..."] if len(processed_items_info_list) > 5 else []) 
        }, indent=2)
    except Exception as e:
        traceback.print_exc() # For server-side debugging
        return json.dumps({
            "success": False,
            "url_or_path": url_or_path,
            "error": str(e)
        }, indent=2)

async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.

    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig()

    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{'url': url, 'markdown': result.markdown}]
    else:
        print(f"Failed to crawl {url}: {result.error_message}")
        return []

async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.

    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    results = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
    return [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]

async def crawl_recursive_internal_links(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int = 3, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.

    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
        if not urls_to_crawl:
            break

        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append({'url': result.url, 'markdown': result.markdown})
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    if next_url not in visited:
                        next_level_urls.add(next_url)

        current_urls = next_level_urls

    return results_all

@mcp.tool(name="process_local_markdown")
async def process_local_markdown(ctx: Context, file_path: str, chunk_size: int = 5000) -> str:
    """
    Process a local markdown file and add its content to the RAG system.

    This tool reads a local markdown file, splits it into manageable chunks, 
    and stores these chunks along with their metadata in the Supabase database 
    for later retrieval and querying in a RAG (Retrieval Augmented Generation) setup.

    Args:
        ctx: The MCP server provided context, used to access shared resources like the Supabase client.
        file_path: The absolute or relative path to the local markdown (.md) file to be processed.
        chunk_size: The target maximum size (in characters) for each chunk of content. 
                    The actual chunk size may vary slightly to respect markdown structures (default: 5000).

    Returns:
        A JSON string summarizing the outcome of the processing. This includes:
        - "success": Boolean indicating if the operation was successful.
        - "file_path": The normalized absolute path of the processed file.
        - "chunks_processed": The number of content chunks stored in Supabase.
        - "source_name": The name of the source, derived from the filename.
        - "error": A message describing any error that occurred (if success is false).
    """
    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        if not os.path.exists(file_path):
            return json.dumps({
                "success": False,
                "file_path": file_path,
                "error": "File not found"
            }, indent=2)

        if not file_path.lower().endswith('.md'):
            return json.dumps({
                "success": False,
                "file_path": file_path,
                "error": "Only .md files are supported"
            }, indent=2)

        content = process_markdown_file(file_path)
        if not content or not content.strip():
            return json.dumps({
                "success": False,
                "file_path": file_path,
                "error": "File is empty or could not be read"
            }, indent=2)

        abs_path = os.path.abspath(file_path)
        normalized_abs_path = abs_path.replace('\\', '/')
        file_url = f"file://{normalized_abs_path}"
        filename = os.path.basename(abs_path)
        source_name = os.path.splitext(filename)[0]

        markdown_chunks = smart_chunk_markdown(content, chunk_size=chunk_size)

        final_chunks_to_store: List[str] = []
        final_metadatas_to_store: List[Dict[str, Any]] = []
        url_to_full_document_map: Dict[str, str] = {file_url: content}

        for i, chunk_md in enumerate(markdown_chunks):
            meta = extract_section_info(chunk_md)
            meta.update({
                'chunk_index': i,
                'source': source_name,
                'filename': filename,
                'doc_type': 'markdown',        # Explicitly 'markdown' for local files
                'local_file': True,
                'file_path': normalized_abs_path,
                'crawl_type': 'local_markdown_tool', # To distinguish if needed
                'url': file_url
            })
            final_chunks_to_store.append(chunk_md)
            final_metadatas_to_store.append(meta)

        if not final_chunks_to_store:
            return json.dumps({
                "success": False,
                "file_path": file_path,
                "error": "No content chunks generated from file"
            }, indent=2)

        doc_urls_for_supabase = [meta['url'] for meta in final_metadatas_to_store]
        chunk_indices_for_supabase = [meta['chunk_index'] for meta in final_metadatas_to_store]

        add_documents_to_supabase(
            supabase_client,
            doc_urls_for_supabase,
            chunk_indices_for_supabase,
            final_chunks_to_store,
            final_metadatas_to_store,
            url_to_full_document_map
        )

        return json.dumps({
            "success": True,
            "file_path": normalized_abs_path,
            "chunks_processed": len(final_chunks_to_store),
            "source_name": source_name
        }, indent=2)

    except Exception as e:
        traceback.print_exc() # For server-side debugging
        return json.dumps({
            "success": False,
            "file_path": file_path,
            "error": f"Error processing local markdown file: {str(e)}"
        }, indent=2)

@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources based on unique source metadata values.

    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database. This is useful for discovering what content is available for querying.

    Args:
        ctx: The MCP server provided context

    Returns:
        JSON string with the list of available sources
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Use a direct query with the Supabase client
        # This could be more efficient with a direct Postgres query but
        # I don't want to require users to set a DB_URL environment variable as well
        result = supabase_client.from_('crawled_pages')\
            .select('metadata')\
            .not_.is_('metadata->>source', 'null')\
            .execute()

        # Use a set to efficiently track unique sources
        unique_sources = set()

        # Extract the source values from the result using a set for uniqueness
        if result.data:
            for item in result.data:
                source = item.get('metadata', {}).get('source')
                if source:
                    unique_sources.add(source)

        # Convert set to sorted list for consistent output
        sources = sorted(list(unique_sources))

        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.

    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.

    Use the tool to get source domains if the user is asking to use a specific tool or framework.

    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with the search results
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}

        # Perform the search
        results = search_documents(
            client=supabase_client,
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata
        )

        # Format the results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "url": result.get("url"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity")
            })

        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())
