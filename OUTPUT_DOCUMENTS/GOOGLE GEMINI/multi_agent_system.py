"""
Multi-Agent System Example with Gemini

This script demonstrates how to create a multi-agent system using Google's Agent Development Kit
with Gemini models, where each agent has a specialized role.
"""

import asyncio
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from google.adk.agents import Agent, BaseAgent
from google.adk.llms import GeminiLLM
from google.adk.sessions import Session
from google.adk.tools import Tool, ToolParameter

# Load environment variables
load_dotenv()

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Define a coordinator agent that will manage the specialized agents
class CoordinatorAgent(BaseAgent):
    """Coordinator agent that delegates tasks to specialized agents."""
    
    def __init__(self, specialized_agents: List[Agent]):
        super().__init__(
            name="Coordinator",
            description="Coordinates tasks between specialized agents"
        )
        self.specialized_agents = specialized_agents
        self.agent_map = {agent.name: agent for agent in specialized_agents}
    
    async def process(self, input_text: str, session: Session = None) -> Dict[str, Any]:
        """Process input by determining which agent should handle it."""
        
        # Create a Gemini LLM for the coordinator
        coordinator_llm = GeminiLLM(
            model="gemini-2.0-flash",
            api_key=GEMINI_API_KEY,
            system_instruction=(
                "You are a coordinator that determines which specialized agent should handle a user request. "
                f"Available agents: {', '.join(self.agent_map.keys())}. "
                "Respond with just the name of the most appropriate agent."
            )
        )
        
        # Create a temporary agent to determine which specialized agent to use
        temp_agent = Agent(
            name="AgentSelector",
            description="Selects the appropriate agent",
            llm=coordinator_llm
        )
        
        # Determine which agent should handle this request
        response = await temp_agent.process(
            f"Based on this user request, which agent should handle it? '{input_text}'",
            session=session
        )
        
        selected_agent_name = response.text.strip()
        
        # Default to the first agent if the selection is invalid
        if selected_agent_name not in self.agent_map:
            selected_agent_name = list(self.agent_map.keys())[0]
            
        # Get the selected agent
        selected_agent = self.agent_map[selected_agent_name]
        
        # Process the input with the selected agent
        result = await selected_agent.process(input_text, session=session)
        
        # Add metadata about which agent was used
        result.metadata = {
            "selected_agent": selected_agent_name,
            **result.metadata
        }
        
        return result

async def main():
    # Create specialized agents
    
    # Research Agent
    research_llm = GeminiLLM(
        model="gemini-2.0-flash",
        api_key=GEMINI_API_KEY,
        system_instruction=(
            "You are a research specialist. Provide detailed, well-researched information "
            "on topics, with citations where possible. Focus on accuracy and depth."
        )
    )
    
    research_agent = Agent(
        name="ResearchAgent",
        description="Specializes in providing detailed research and information",
        llm=research_llm
    )
    
    # Creative Agent
    creative_llm = GeminiLLM(
        model="gemini-2.0-flash",
        api_key=GEMINI_API_KEY,
        system_instruction=(
            "You are a creative specialist. Generate imaginative content, stories, "
            "ideas, and creative solutions to problems. Think outside the box."
        )
    )
    
    creative_agent = Agent(
        name="CreativeAgent",
        description="Specializes in creative writing and idea generation",
        llm=creative_llm
    )
    
    # Technical Agent
    technical_llm = GeminiLLM(
        model="gemini-2.0-flash",
        api_key=GEMINI_API_KEY,
        system_instruction=(
            "You are a technical specialist. Provide precise technical explanations, "
            "code examples, and solutions to technical problems. Be concise and accurate."
        )
    )
    
    technical_agent = Agent(
        name="TechnicalAgent",
        description="Specializes in technical content and coding",
        llm=technical_llm
    )
    
    # Create the coordinator agent
    coordinator = CoordinatorAgent(
        specialized_agents=[research_agent, creative_agent, technical_agent]
    )
    
    # Create a session
    session = Session()
    
    print("Multi-Agent System is ready! Type 'exit' to quit.")
    print("Your request will be routed to the most appropriate specialized agent:")
    print("- ResearchAgent: For factual information and research")
    print("- CreativeAgent: For creative content and ideas")
    print("- TechnicalAgent: For technical explanations and code")
    print("-" * 50)
    
    # Simple interaction loop
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check if user wants to exit
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        # Process the user input with the coordinator
        response = await coordinator.process(user_input, session=session)
        
        # Print which agent was used
        agent_used = response.metadata.get("selected_agent", "Unknown")
        print(f"[Using {agent_used}]")
        
        # Print the agent's response
        print(f"Agent: {response.text}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
