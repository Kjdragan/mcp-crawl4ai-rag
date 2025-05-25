"""
Gemini Agent with Custom Tools Example

This script demonstrates how to create a Gemini agent with custom tools
using Google's Agent Development Kit.
"""

import asyncio
import os
import json
import datetime
import requests
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.llms import GeminiLLM
from google.adk.sessions import Session
from google.adk.tools import Tool, ToolParameter, ToolSpec

# Load environment variables
load_dotenv()

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Define custom tools
class WeatherTool(Tool):
    """Tool for getting current weather information."""
    
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get the current weather for a specified location",
            parameters=[
                ToolParameter(
                    name="location",
                    description="The city and country, e.g., 'London, UK'",
                    type="string",
                    required=True
                )
            ]
        )
    
    async def execute(self, parameters, context=None):
        location = parameters.get("location", "")
        
        # In a real implementation, you would call a weather API
        # This is a mock implementation
        weather_data = {
            "location": location,
            "temperature": "22°C",
            "condition": "Partly Cloudy",
            "humidity": "65%",
            "wind": "10 km/h",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return json.dumps(weather_data)

class CalendarTool(Tool):
    """Tool for checking calendar events."""
    
    def __init__(self):
        super().__init__(
            name="check_calendar",
            description="Check calendar events for a specific date",
            parameters=[
                ToolParameter(
                    name="date",
                    description="The date to check in YYYY-MM-DD format",
                    type="string",
                    required=True
                )
            ]
        )
    
    async def execute(self, parameters, context=None):
        date = parameters.get("date", "")
        
        # Mock calendar data
        events = [
            {
                "title": "Team Meeting",
                "time": "09:00 - 10:00",
                "location": "Conference Room A"
            },
            {
                "title": "Project Review",
                "time": "14:00 - 15:30",
                "location": "Virtual"
            }
        ]
        
        return json.dumps({
            "date": date,
            "events": events
        })

async def main():
    # Initialize the Gemini LLM
    llm = GeminiLLM(
        model="gemini-2.0-flash",
        api_key=GEMINI_API_KEY,
        system_instruction=(
            "You are a helpful assistant with access to tools. "
            "Use the tools when appropriate to provide accurate information."
        )
    )
    
    # Create tools
    weather_tool = WeatherTool()
    calendar_tool = CalendarTool()
    
    # Create an agent with the Gemini LLM and tools
    agent = Agent(
        name="GeminiToolAgent",
        description="A Gemini agent with weather and calendar tools",
        llm=llm,
        tools=[weather_tool, calendar_tool]
    )
    
    # Create a session for the agent
    session = Session()
    
    print("Gemini Tool Agent is ready! Type 'exit' to quit.")
    print("Try asking about the weather or checking your calendar.")
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
