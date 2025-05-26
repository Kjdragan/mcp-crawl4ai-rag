"""
Search Agent Package

This package contains a search agent built with Google ADK that uses Gemini 2.5 Pro on Vertex AI.
"""

# Import the module where your root_agent is defined
from . import search_agent

# Create an alias. The ADK framework looks for 'agents.agent.root_agent'.
# By doing this, 'agents.agent' will refer to your 'search_agent.py' module,
# and ADK can find 'root_agent' within it.
agent = search_agent
