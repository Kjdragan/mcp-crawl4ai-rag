"""
Model Comparison: Gemini vs Claude Sonnet 4

This script demonstrates how to use both Google's Gemini and Anthropic's Claude Sonnet 4
models side by side for various tasks, allowing for direct comparison of their capabilities.
"""

import os
import asyncio
import json
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel
import anthropic
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Get API keys from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

# Set up authentication for Vertex AI (assuming you have a service account key)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "path/to/your/service-account-key.json")

# Initialize Vertex AI with your project and region
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION = "us-central1"  # Change to your preferred region

class ModelComparison:
    """Class for comparing Gemini and Claude models."""
    
    def __init__(self):
        """Initialize clients for both models."""
        print("Initializing clients for Gemini and Claude Sonnet 4")
        
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        self.gemini = GenerativeModel("gemini-pro")
        
        # Initialize Anthropic client
        self.claude = Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Common parameters
        self.max_tokens = 1024
        self.temperature = 0.7
    
    async def compare_text_generation(self, prompt):
        """Compare text generation between models."""
        print("\n=== Text Generation Comparison ===")
        print(f"Prompt: {prompt}")
        
        # Generate with Gemini
        print("\n--- Gemini Response ---")
        gemini_response = self.gemini.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens
            }
        )
        print(gemini_response.text)
        
        # Generate with Claude
        print("\n--- Claude Sonnet 4 Response ---")
        claude_response = self.claude.messages.create(
            model="claude-3-sonnet-20240229",  # Use claude-3-5-sonnet-20240620 when available
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        print(claude_response.content[0].text)
        
        return {
            "gemini": gemini_response.text,
            "claude": claude_response.content[0].text
        }
    
    async def compare_structured_output(self, prompt, schema):
        """Compare structured output generation between models."""
        print("\n=== Structured Output Comparison ===")
        print(f"Prompt: {prompt}")
        
        # Generate with Gemini
        print("\n--- Gemini Response ---")
        gemini_response = self.gemini.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,  # Lower temperature for more deterministic output
                "max_output_tokens": self.max_tokens,
                "response_schema": schema
            }
        )
        print(gemini_response.text)
        
        # Generate with Claude
        print("\n--- Claude Sonnet 4 Response ---")
        claude_system = """
        You are an assistant that provides information in structured JSON format.
        Always respond with valid JSON that follows the requested schema.
        Do not include any explanatory text outside the JSON structure.
        """
        
        claude_prompt = f"{prompt}\n\nRespond with JSON that follows this schema: {json.dumps(schema)}"
        
        claude_response = self.claude.messages.create(
            model="claude-3-sonnet-20240229",  # Use claude-3-5-sonnet-20240620 when available
            max_tokens=self.max_tokens,
            temperature=0.2,  # Lower temperature for more deterministic output
            system=claude_system,
            messages=[
                {"role": "user", "content": claude_prompt}
            ]
        )
        
        # Try to parse the Claude response as JSON
        try:
            claude_json = json.loads(claude_response.content[0].text)
            print(json.dumps(claude_json, indent=2))
        except json.JSONDecodeError:
            print("Failed to parse Claude response as JSON")
            print(claude_response.content[0].text)
        
        return {
            "gemini": gemini_response.text,
            "claude": claude_response.content[0].text
        }
    
    async def compare_creative_writing(self, prompt):
        """Compare creative writing capabilities."""
        print("\n=== Creative Writing Comparison ===")
        print(f"Prompt: {prompt}")
        
        # Generate with Gemini
        print("\n--- Gemini Response ---")
        gemini_response = self.gemini.generate_content(
            prompt,
            generation_config={
                "temperature": 0.9,  # Higher temperature for more creative output
                "max_output_tokens": self.max_tokens
            }
        )
        print(gemini_response.text)
        
        # Generate with Claude
        print("\n--- Claude Sonnet 4 Response ---")
        claude_system = "You are a creative writing assistant that specializes in imaginative and engaging content."
        
        claude_response = self.claude.messages.create(
            model="claude-3-sonnet-20240229",  # Use claude-3-5-sonnet-20240620 when available
            max_tokens=self.max_tokens,
            temperature=0.9,  # Higher temperature for more creative output
            system=claude_system,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        print(claude_response.content[0].text)
        
        return {
            "gemini": gemini_response.text,
            "claude": claude_response.content[0].text
        }
    
    async def compare_reasoning(self, prompt):
        """Compare reasoning capabilities."""
        print("\n=== Reasoning Comparison ===")
        print(f"Prompt: {prompt}")
        
        # Generate with Gemini
        print("\n--- Gemini Response ---")
        gemini_response = self.gemini.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,  # Lower temperature for more logical output
                "max_output_tokens": self.max_tokens
            }
        )
        print(gemini_response.text)
        
        # Generate with Claude
        print("\n--- Claude Sonnet 4 Response ---")
        claude_system = "You are an assistant that specializes in logical reasoning and step-by-step problem solving."
        
        claude_response = self.claude.messages.create(
            model="claude-3-sonnet-20240229",  # Use claude-3-5-sonnet-20240620 when available
            max_tokens=self.max_tokens,
            temperature=0.3,  # Lower temperature for more logical output
            system=claude_system,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        print(claude_response.content[0].text)
        
        return {
            "gemini": gemini_response.text,
            "claude": claude_response.content[0].text
        }

async def main():
    """Run the model comparison examples."""
    comparison = ModelComparison()
    
    # Compare text generation
    await comparison.compare_text_generation(
        "Explain the concept of quantum computing to a high school student."
    )
    
    # Compare structured output
    book_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"},
            "year_published": {"type": "integer"},
            "genre": {"type": "string"},
            "summary": {"type": "string"},
            "rating": {"type": "number", "minimum": 1, "maximum": 5}
        },
        "required": ["title", "author", "genre", "summary"]
    }
    
    await comparison.compare_structured_output(
        "Recommend a science fiction book about time travel.",
        book_schema
    )
    
    # Compare creative writing
    await comparison.compare_creative_writing(
        "Write a short story about a robot that discovers it has emotions."
    )
    
    # Compare reasoning
    await comparison.compare_reasoning(
        "A farmer needs to take a fox, a chicken, and a sack of grain across a river. " +
        "The boat is only large enough to carry the farmer and one item. " +
        "If left unattended, the fox will eat the chicken, and the chicken will eat the grain. " +
        "How can the farmer get everything across the river safely?"
    )
    
    print("\nAll comparisons completed!")

if __name__ == "__main__":
    asyncio.run(main())
