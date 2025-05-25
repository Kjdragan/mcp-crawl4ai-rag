"""
Simple Gemini Agent Example using Google ADK

This script demonstrates how to create a basic agent using Google's Agent Development Kit
with the Gemini model.
"""

import asyncio
import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.llms import GeminiLLM
from google.adk.sessions import Session

# Load environment variables
load_dotenv()

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

async def main():
    # Initialize the Gemini LLM
    llm = GeminiLLM(
        model="gemini-2.0-flash",
        api_key=GEMINI_API_KEY,
        system_instruction=(
            "You are a helpful assistant that provides clear and concise responses. "
            "Always be polite and professional in your interactions."
        )
    )
    
    # Create an agent with the Gemini LLM
    agent = Agent(
        name="SimpleGeminiAgent",
        description="A simple agent powered by Google Gemini",
        llm=llm
    )
    
    # Create a session for the agent
    session = Session()
    
    print("Simple Gemini Agent is ready! Type 'exit' to quit.")
    print("-" * 50)
    
    # Simple interaction loop
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check if user wants to exit
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        # Process the user input with the agent
        response = await agent.process(user_input, session=session)
        
        # Print the agent's response
        print(f"Agent: {response.text}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
