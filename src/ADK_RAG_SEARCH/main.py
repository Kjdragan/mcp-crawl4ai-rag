import json
import os
from typing import Dict, List, Optional

from adk import Agent, Config, Message, AgentContext, serve
from openai import OpenAI


class RAGSearchAgent(Agent):
    """
    An ADK agent that searches content using the crawl4ai-rag system
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.api_client = OpenAI()

    async def process_message(self, context: AgentContext, message: Message) -> str:
        """Process user message and return a response"""
        query = message.content
        if not query:
            return "Please provide a search query."

        # Initialize response with context about available sources
        response = [
            f"Searching for: '{query}'\n",
            "Available sources: ai.pydantic.dev, docs.mem0.ai, google.github.io\n"
        ]

        # Perform RAG query
        try:
            # Call the MCP server's RAG query function
            results = await self._perform_rag_query(query)
            
            if not results or not results.get("matches"):
                return f"No results found for '{query}'. Please try a different search term."
            
            response.append("## Search Results\n")
            
            for i, match in enumerate(results.get("matches", [])):
                response.append(f"### {i+1}. {match.get('title', 'Untitled')}\n")
                response.append(f"**Source**: {match.get('source', 'Unknown')}\n")
                response.append(f"**URL**: {match.get('url', 'No URL')}\n")
                response.append("\n**Content**:\n")
                response.append(f"{match.get('content', 'No content available')}\n\n")
            
            return "\n".join(response)
        
        except Exception as e:
            return f"Error performing search: {str(e)}"
    
    async def _perform_rag_query(self, query: str, source: Optional[str] = None, match_count: int = 5) -> Dict:
        """
        Perform a RAG query using an external system
        
        Args:
            query: The search query
            source: Optional source domain to filter results
            match_count: Maximum number of results to return
            
        Returns:
            Dictionary containing search results
        """
        # Here we would normally call an external API, but for demonstration we'll simulate it
        # In a real implementation, you would replace this with actual API calls
        
        # Simulate calling the crawl4ai-rag MCP server
        try:
            # This is a placeholder for the actual API call
            # In a real implementation, you'd replace this with appropriate code
            
            # You could invoke an external API here, like:
            # response = requests.post(
            #     "https://your-rag-api-endpoint",
            #     json={"query": query, "source": source, "match_count": match_count}
            # )
            # return response.json()
            
            # For demonstration, we'll simulate calling the MCP tool:
            
            # Note: In a real implementation, you would replace this with actual API calls
            # This is just a demonstration of how it would work
            
            # Placeholder for returned results
            return {
                "matches": [
                    {
                        "title": "This is a simulated result",
                        "source": "ai.pydantic.dev",
                        "url": "https://ai.pydantic.dev/some-page",
                        "content": "This is simulated content. In a real implementation, this would be the actual content retrieved from the RAG system."
                    }
                ]
            }
            
        except Exception as e:
            print(f"Error in RAG query: {str(e)}")
            return {"matches": []}


def main():
    """Run the agent server"""
    config = Config(name="RAG Search Agent")
    agent = RAGSearchAgent(config)
    serve(agent)


if __name__ == "__main__":
    main()
