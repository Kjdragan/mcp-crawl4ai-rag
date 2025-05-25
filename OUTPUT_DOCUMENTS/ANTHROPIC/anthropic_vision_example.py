#!/usr/bin/env python3
"""
Example script demonstrating how to use Anthropic's Claude API with vision capabilities.

This example shows how to:
1. Set up the Anthropic client
2. Prepare images for Claude (base64 encoding)
3. Create multimodal messages with text and images
4. Process multimodal responses

Requirements:
- anthropic Python package
- Pillow (PIL) for image processing

Install with: 
uv add anthropic
uv add pillow
"""

import os
import base64
import io
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from anthropic import Anthropic
from PIL import Image


def setup_client() -> Anthropic:
    """
    Set up the Anthropic client with API key from environment variable.
    
    Returns:
        The Anthropic client
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    return Anthropic(api_key=api_key)


def encode_image_from_file(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Encode an image from a file path to base64 for Claude.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with image data in Claude's expected format
    """
    # Convert string path to Path object if needed
    if isinstance(image_path, str):
        image_path = Path(image_path)
    
    # Check if file exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Determine media type based on file extension
    extension = image_path.suffix.lower()
    media_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    
    media_type = media_type_map.get(extension)
    if not media_type:
        raise ValueError(f"Unsupported image format: {extension}")
    
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Return the encoded image in Claude's expected format
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64_image
        }
    }


def encode_image_from_pil(image: Image.Image, format: str = "JPEG") -> Dict[str, Any]:
    """
    Encode a PIL Image object to base64 for Claude.
    
    Args:
        image: PIL Image object
        format: Image format (JPEG, PNG, etc.)
        
    Returns:
        Dictionary with image data in Claude's expected format
    """
    # Convert format to lowercase for media_type
    format_lower = format.lower()
    if format_lower == "jpg":
        format_lower = "jpeg"
    
    media_type = f"image/{format_lower}"
    
    # Convert image to bytes
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Return the encoded image in Claude's expected format
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64_image
        }
    }


def resize_image_if_needed(image_path: Union[str, Path], max_size: int = 5242880) -> Dict[str, Any]:
    """
    Resize an image if it's too large for Claude's API limits.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum size in bytes (default: 5MB)
        
    Returns:
        Dictionary with image data in Claude's expected format
    """
    # Convert string path to Path object if needed
    if isinstance(image_path, str):
        image_path = Path(image_path)
    
    # Check if file exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Check file size
    file_size = image_path.stat().st_size
    
    # If file is small enough, use the direct encoding
    if file_size <= max_size:
        return encode_image_from_file(image_path)
    
    # File is too large, resize it
    print(f"Image is {file_size/1024/1024:.2f}MB, resizing to fit under {max_size/1024/1024:.2f}MB")
    
    # Open the image with PIL
    image = Image.open(image_path)
    
    # Calculate new dimensions while maintaining aspect ratio
    width, height = image.size
    aspect_ratio = width / height
    
    # Start with a quality of 85 and reduce until the image is small enough
    quality = 85
    format = "JPEG"  # JPEG has better compression than PNG
    
    while True:
        # If we've reduced quality too much, resize the image instead
        if quality < 50:
            # Reduce dimensions by 25%
            width = int(width * 0.75)
            height = int(width / aspect_ratio)
            image = image.resize((width, height), Image.LANCZOS)
            quality = 85  # Reset quality
        
        # Convert to bytes to check size
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=quality)
        current_size = len(buffer.getvalue())
        
        # If small enough, break the loop
        if current_size <= max_size:
            break
        
        # Reduce quality for next iteration
        quality -= 10
    
    # Convert to base64
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Return the encoded image in Claude's expected format
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": f"image/{format.lower()}",
            "data": base64_image
        }
    }


def create_multimodal_message(
    client: Anthropic,
    text_prompt: str,
    image_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    system_prompt: Optional[str] = None,
    model: str = "claude-3-opus-20240229",
    max_tokens: int = 1024
) -> Dict[str, Any]:
    """
    Create a multimodal message with text and images using Claude.
    
    Args:
        client: Anthropic client
        text_prompt: Text prompt to send to Claude
        image_data: Image data in Claude's expected format (single image or list)
        system_prompt: Optional system prompt
        model: Claude model to use
        max_tokens: Maximum tokens to generate
        
    Returns:
        The response from Claude
    """
    # Prepare the content array
    content = []
    
    # Add images to content
    if isinstance(image_data, list):
        content.extend(image_data)
    else:
        content.append(image_data)
    
    # Add text prompt to content
    content.append({
        "type": "text",
        "text": text_prompt
    })
    
    # Prepare message parameters
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": content}
        ]
    }
    
    # Add system prompt if provided
    if system_prompt:
        params["system"] = system_prompt
    
    # Create the message
    return client.messages.create(**params)


def extract_text_from_response(response: Dict[str, Any]) -> str:
    """
    Extract text content from Claude's response.
    
    Args:
        response: Response from Claude
        
    Returns:
        The extracted text content
    """
    text_content = ""
    
    # Check if the response has content
    if hasattr(response, "content"):
        # Iterate through content blocks
        for content_block in response.content:
            # Check if the content block is text
            if content_block.type == "text":
                text_content += content_block.text
    
    return text_content


def main():
    """Main function to demonstrate Claude vision capabilities."""
    # Set up the Anthropic client
    client = setup_client()
    
    # Example 1: Single image analysis
    print("Example 1: Single Image Analysis")
    print("-------------------------------")
    
    # Path to an example image
    # Replace with an actual image path on your system
    image_path = "example_image.jpg"
    
    # Check if the image exists, if not, provide instructions
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Please update the script with a valid image path on your system.")
        print("For example: C:\\Users\\username\\Pictures\\example.jpg")
        return
    
    try:
        # Encode the image
        image_data = resize_image_if_needed(image_path)
        
        # Create a multimodal message
        text_prompt = "What's in this image? Please describe it in detail."
        system_prompt = "You are a helpful AI assistant with vision capabilities. Provide detailed and accurate descriptions of images."
        
        response = create_multimodal_message(
            client=client,
            text_prompt=text_prompt,
            image_data=image_data,
            system_prompt=system_prompt
        )
        
        # Extract and print the response
        text_content = extract_text_from_response(response)
        print("\nClaude's response:")
        print(text_content)
        
        # Example 2: Multiple images comparison
        print("\n\nExample 2: Multiple Images Comparison")
        print("------------------------------------")
        
        # Paths to example images for comparison
        # Replace with actual image paths on your system
        image_path_1 = "example_image_1.jpg"
        image_path_2 = "example_image_2.jpg"
        
        # Check if the images exist
        if not Path(image_path_1).exists() or not Path(image_path_2).exists():
            print(f"One or more images not found: {image_path_1}, {image_path_2}")
            print("Please update the script with valid image paths on your system.")
            return
        
        # Encode the images
        image_data_1 = resize_image_if_needed(image_path_1)
        image_data_2 = resize_image_if_needed(image_path_2)
        
        # Create a multimodal message with multiple images
        text_prompt = "Compare these two images and describe the differences between them."
        
        response = create_multimodal_message(
            client=client,
            text_prompt=text_prompt,
            image_data=[image_data_1, image_data_2],
            system_prompt=system_prompt
        )
        
        # Extract and print the response
        text_content = extract_text_from_response(response)
        print("\nClaude's response:")
        print(text_content)
        
        # Example 3: OCR and text extraction
        print("\n\nExample 3: OCR and Text Extraction")
        print("----------------------------------")
        
        # Path to an example image with text
        # Replace with an actual image path on your system
        image_path_text = "example_text_image.jpg"
        
        # Check if the image exists
        if not Path(image_path_text).exists():
            print(f"Image not found: {image_path_text}")
            print("Please update the script with a valid image path on your system.")
            return
        
        # Encode the image
        image_data_text = resize_image_if_needed(image_path_text)
        
        # Create a multimodal message for OCR
        text_prompt = "Extract all text from this image. Format it properly and maintain the original structure."
        
        response = create_multimodal_message(
            client=client,
            text_prompt=text_prompt,
            image_data=image_data_text,
            system_prompt="You are a helpful AI assistant with OCR capabilities. Extract text accurately from images."
        )
        
        # Extract and print the response
        text_content = extract_text_from_response(response)
        print("\nClaude's response:")
        print(text_content)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
