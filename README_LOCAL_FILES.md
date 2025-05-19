# Processing Local Markdown Files with Crawl4AI RAG

This guide explains how to process local markdown files for inclusion in the Crawl4AI RAG system.

## Overview

The Crawl4AI RAG system can now process local markdown files directly from your machine using the provided utility scripts. This allows you to add your own documentation to the knowledge base without having to host it on a web server.

## Methods

There are two main approaches to processing local files:

### 1. Direct Processing (Recommended)

The `process_markdown.py` script provides direct access to the local file processing functionality:

```bash
# Process a single markdown file
uv run process_markdown.py "path/to/your/file.md"

# Specify custom chunk size (default is 5000)
uv run process_markdown.py "path/to/your/file.md" --chunk-size 3000
```

This method is the most reliable as it directly uses the underlying file processing logic.

### 2. MCP Tool Approach (Via API)

If you need to process files through the MCP API (e.g., from an external application), you can use the `process_local_file.py` script which attempts to communicate with the MCP server:

```bash
# Process a file via the MCP API
uv run process_local_file.py "path/to/your/file.md"
```

Note: This approach is less reliable due to limitations with the MCP server's API endpoints.

## How It Works

1. The script reads the markdown file from your local filesystem
2. The content is processed and divided into semantic chunks
3. Each chunk is processed to extract metadata and embeddings
4. The chunks are stored in the Supabase database associated with the Crawl4AI RAG system
5. The content becomes available for RAG queries via the MCP tools

## Troubleshooting

- If you encounter permission issues, ensure your user has read access to the file
- Large files may take longer to process due to embedding generation
- Files are identified by their absolute path, so processing the same file twice will replace the previous entries

## Example Usage

Process a documentation file and then query it:

```bash
# Add a file to the knowledge base
uv run process_markdown.py "C:\path\to\my_documentation.md"

# Then query it using the standard RAG query tools
# (either through an application using the MCP API or directly)
```

After processing, you can query the content using the standard RAG query tools or the MCP `perform_rag_query` tool.
