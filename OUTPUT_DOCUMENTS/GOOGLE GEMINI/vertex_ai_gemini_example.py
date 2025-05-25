"""
Vertex AI with Gemini Example

This script demonstrates how to use Google Cloud's Vertex AI platform with Gemini models
for various generative AI tasks.
"""

import os
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.preview.generative_models import GenerationConfig

# Load environment variables
load_dotenv()

# Set up authentication (assuming you have a service account key)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "path/to/your/service-account-key.json")

# Initialize Vertex AI with your project and region
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION = "us-central1"  # Change to your preferred region

def initialize_vertex_ai():
    """Initialize Vertex AI with project and location."""
    print(f"Initializing Vertex AI with project: {PROJECT_ID} in {LOCATION}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

def text_generation_example():
    """Example of text generation with Gemini Pro."""
    print("\n=== Text Generation Example ===")
    
    # Load the Gemini Pro model
    model = GenerativeModel("gemini-pro")
    
    # Configure generation parameters
    generation_config = GenerationConfig(
        temperature=0.4,
        top_p=0.8,
        top_k=40,
        max_output_tokens=1024,
    )
    
    # Generate content
    prompt = "Write a short story about a robot that learns to paint."
    print(f"Prompt: {prompt}")
    
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )
    
    print("\nGenerated Story:")
    print(response.text)

def multimodal_example():
    """Example of multimodal input with Gemini Pro Vision."""
    print("\n=== Multimodal Example ===")
    
    # Load the Gemini Pro Vision model
    model = GenerativeModel("gemini-pro-vision")
    
    # You would typically load an image from a file or URL
    # For this example, we'll just describe what would happen
    print("Note: This example requires an actual image file.")
    print("In a real implementation, you would do:")
    print("""
    # Load image
    image = Part.from_uri(
        "gs://your-bucket/your-image.jpg",  # Cloud Storage URI
        mime_type="image/jpeg"
    )
    
    # Or from a local file
    with open("local_image.jpg", "rb") as f:
        image_data = f.read()
    image = Part.from_data(image_data, mime_type="image/jpeg")
    
    # Create a prompt with text and image
    prompt = "Describe what you see in this image in detail."
    
    # Generate content with multimodal input
    response = model.generate_content([prompt, image])
    
    print(response.text)
    """)

def function_calling_example():
    """Example of function calling with Gemini."""
    print("\n=== Function Calling Example ===")
    
    # Load the Gemini Pro model
    model = GenerativeModel("gemini-pro")
    
    # Define a function for weather information
    weather_function = {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g., 'San Francisco, CA, USA'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit"
                }
            },
            "required": ["location"]
        }
    }
    
    # Mock function to handle the weather request
    def get_weather(location, unit="celsius"):
        """Mock function that would normally call a weather API."""
        print(f"Called get_weather with location={location}, unit={unit}")
        # In a real implementation, you would call a weather API here
        return {
            "location": location,
            "temperature": "22" if unit == "celsius" else "72",
            "unit": unit,
            "condition": "Sunny",
            "humidity": "45%"
        }
    
    # Prompt that should trigger function calling
    prompt = "What's the weather like in Chicago right now?"
    print(f"Prompt: {prompt}")
    
    # Generate content with function calling
    response = model.generate_content(
        prompt,
        tools=[weather_function]
    )
    
    # Check if function calling was triggered
    if hasattr(response, "candidates") and response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        print(f"\nFunction call detected: {function_call.name}")
        print(f"Arguments: {function_call.args}")
        
        # Execute the function
        if function_call.name == "get_weather":
            result = get_weather(**function_call.args)
            
            # Send the function result back to the model
            response = model.generate_content(
                [prompt, Part.from_function_response(
                    name=function_call.name,
                    response=result
                )]
            )
            
            print("\nFinal response with function results:")
            print(response.text)
    else:
        print("\nNo function call detected. Response:")
        print(response.text)

def structured_output_example():
    """Example of generating structured output with Gemini."""
    print("\n=== Structured Output Example ===")
    
    # Load the Gemini Pro model
    model = GenerativeModel("gemini-pro")
    
    # Define a schema for a book recommendation
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
    
    # Prompt for book recommendation
    prompt = "Recommend a science fiction book about time travel."
    print(f"Prompt: {prompt}")
    
    # Generate structured output
    response = model.generate_content(
        prompt,
        generation_config={"response_schema": book_schema}
    )
    
    print("\nStructured Book Recommendation:")
    print(response.text)

def main():
    """Run the Vertex AI with Gemini examples."""
    initialize_vertex_ai()
    
    # Run examples
    text_generation_example()
    multimodal_example()
    function_calling_example()
    structured_output_example()
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    main()
