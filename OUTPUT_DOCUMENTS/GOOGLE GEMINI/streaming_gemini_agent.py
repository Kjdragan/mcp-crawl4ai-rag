"""
Streaming Gemini Agent Example

This script demonstrates how to create a streaming agent using Google's Agent Development Kit
with the Gemini model.
"""

import asyncio
import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.llms import GeminiLLM
from google.adk.sessions import Session
from google.adk.streaming import StreamingConfig, StreamingHandler

# Load environment variables
load_dotenv()

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Define a custom streaming handler
class CustomStreamingHandler(StreamingHandler):
    """Custom handler for streaming responses."""
    
    async def on_text(self, text: str) -> None:
        """Called when a text chunk is received."""
        print(text, end="", flush=True)
    
    async def on_error(self, error: Exception) -> None:
        """Called when an error occurs during streaming."""
        print(f"\nError during streaming: {error}")
    
    async def on_end(self) -> None:
        """Called when the streaming response is complete."""
        print("\n" + "-" * 50)

async def main():
    # Initialize the Gemini LLM with streaming enabled
    llm = GeminiLLM(
        model="gemini-2.0-flash",
        api_key=GEMINI_API_KEY,
        system_instruction=(
            "You are a helpful assistant that provides detailed and informative responses. "
            "Break down complex topics into easy-to-understand explanations."
        ),
        streaming_config=StreamingConfig(
            enabled=True,
            handler=CustomStreamingHandler()
        )
    )
    
    # Create an agent with the Gemini LLM
    agent = Agent(
        name="StreamingGeminiAgent",
        description="A streaming agent powered by Google Gemini",
        llm=llm
    )
    
    # Create a session for the agent
    session = Session()
    
    print("Streaming Gemini Agent is ready! Type 'exit' to quit.")
    print("Responses will be streamed word by word.")
    print("-" * 50)
    
    # Simple interaction loop
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check if user wants to exit
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        print("Agent: ", end="", flush=True)
        
        # Process the user input with the agent (response will be streamed via handler)
        await agent.process(user_input, session=session)

if __name__ == "__main__":
    asyncio.run(main())
