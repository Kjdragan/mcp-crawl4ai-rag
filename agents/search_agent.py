"""
Google ADK Search Agent

This agent uses Google ADK to implement a search agent with web search capabilities.
It can be run using ADK web interface.
"""

import os
from typing import Dict, Any, List
from pathlib import Path

# Google ADK imports - correct imports for the latest version
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.tools import google_search
from google.adk.models import VertexAIModel

# Environment variables
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory
project_root = Path(__file__).parent.parent.absolute()

# Load environment variables from the project root .env file
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)
print(f"Loading environment variables from: {dotenv_path}")

# Get environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL_NAME = os.getenv("VERTEX_MODEL_NAME", "gemini-2.5-pro-preview-05-06")


def summarize_search_results(query: str, results: List[Dict[str, Any]]) -> str:
    """
    Summarize search results into a readable format.
    
    Args:
        query: The search query
        results: List of search result dictionaries
        
    Returns:
        Formatted summary of search results
    """
    summary = f"Search results for: '{query}'\n\n"
    
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        url = result.get("url", "No URL")
        snippet = result.get("snippet", "No description available")
        
        summary += f"{i}. {title}\n"
        summary += f"   URL: {url}\n"
        summary += f"   Summary: {snippet}\n\n"
    
    return summary


def get_vertex_model_info() -> str:
    """
    Get information about the Vertex AI model being used.
    
    Returns:
        Information about the Vertex AI model
    """
    return f"Using Vertex AI model: {VERTEX_MODEL_NAME} in {VERTEX_LOCATION} region"


# Create the search agent
def create_search_agent():
    """
    Create and configure the search agent with web search capabilities.
    
    Returns:
        Configured Agent instance
    """
    # Configure the Google Search tool
    search_tool = google_search.GoogleSearch()
    
    # Create a tool to summarize search results
    summarize_tool = FunctionTool(
        name="summarize_search_results",
        description="Summarize search results into a readable format",
        function=summarize_search_results
    )
    
    # Create a tool to get model information
    model_info_tool = FunctionTool(
        name="get_vertex_model_info",
        description="Get information about the Vertex AI model being used",
        function=get_vertex_model_info
    )
    
    # Create the agent with the search tools
    agent = Agent(
        name="search_agent",
        description="An agent that can search the web and summarize results using Gemini 2.5 Pro on Vertex AI",
        model=VertexAIModel(
            project_id=GOOGLE_CLOUD_PROJECT,
            location=VERTEX_LOCATION,
            model_name=VERTEX_MODEL_NAME
        ),
        tools=[search_tool, summarize_tool, model_info_tool],
        instruction="""
        You are a helpful search assistant that can find information on the web.
        
        When a user asks a question:
        1. Use the google_search tool to find relevant information
        2. Use the summarize_search_results tool to format the results nicely
        3. Provide a concise answer based on the search results
        4. Always cite your sources with links
        
        If the search doesn't return useful results, acknowledge this and suggest 
        alternative search terms or approaches.
        """
    )
    
    return agent


# Create the agent instance
search_agent = create_search_agent()
