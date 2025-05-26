#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Search Agent using Google ADK with Gemini 2.5 Pro on Vertex AI
"""

import os
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

# Get the project root directory
project_root = Path(__file__).parent.parent.absolute()

# Load environment variables from the project root .env file
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)
print(f"Loading environment variables from: {dotenv_path}")

# Get environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Model ID to be used by the LlmAgent, read from VERTEX_AI_MODEL_NAME in .env
AGENT_MODEL_ID_FROM_ENV = os.getenv("VERTEX_AI_MODEL_NAME", "gemini-1.5-flash-001") # Default if not set
# GOOGLE_CLOUD_PROJECT_ID and GOOGLE_CLOUD_REGION will be read by ADK from os.environ


# Create the search agent
def create_search_agent() -> LlmAgent:
    """
    Create a search agent using Google ADK with Gemini 2.5 Pro on Vertex AI

    Returns:
        Agent: A Google ADK Agent configured with search capabilities
    """
    # Create the agent with the search tools
    # ADK will use GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION from env
    agent = LlmAgent(
        model=AGENT_MODEL_ID_FROM_ENV, # Pass model name as string
        tools=[
            google_search
        ],
        instruction=(
            "You are a helpful search agent. When asked a question, the Google Search tool will be automatically invoked by the model to find relevant information. "
            "Present the findings in a clear and concise way. Always cite your sources with links if provided by the search tool. "
            "If the user asks about your capabilities, explain that you can search the web using Google Search and provide summarized information with source citations. "
            "For complex queries, the model will break them down into specific search terms to get the most relevant results."
        ),
        name="search_agent",
        description="An agent that can search the web and summarize results using Gemini models with built-in Google Search."
    )

    return agent


# Create the agent instance with the name 'root_agent'
# This is the standard name ADK looks for when discovering agents
root_agent = create_search_agent()

# This is required for ADK to discover the agent when run directly
if __name__ == "__main__":
    print("Search agent initialized and ready to use.")
    print(f"  Model ID (from VERTEX_AI_MODEL_NAME in .env): {AGENT_MODEL_ID_FROM_ENV}")
    # ADK will attempt to use GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION from environment if GOOGLE_GENAI_USE_VERTEXAI is TRUE
    print(f"  Relevant env vars for Vertex: GOOGLE_GENAI_USE_VERTEXAI='{os.getenv('GOOGLE_GENAI_USE_VERTEXAI')}', GOOGLE_CLOUD_PROJECT='{os.getenv('GOOGLE_CLOUD_PROJECT')}', GOOGLE_CLOUD_LOCATION='{os.getenv('GOOGLE_CLOUD_LOCATION')}'")
