# Anthropic Claude API Scripts - Detailed Documentation

*Last Updated: May 23, 2025*

This document provides detailed documentation for each script in the Anthropic Claude API integration examples.

## Table of Contents

1. [anthropic_mcp_connector_example.py](#anthropic_mcp_connector_examplepy)
2. [anthropic_mcp_example.py](#anthropic_mcp_examplepy)
3. [anthropic_remote_mcp_example.py](#anthropic_remote_mcp_examplepy)
4. [anthropic_message_batch_example.py](#anthropic_message_batch_examplepy)
5. [anthropic_streaming_example.py](#anthropic_streaming_examplepy)
6. [anthropic_tool_use_example.py](#anthropic_tool_use_examplepy)
7. [anthropic_vision_example.py](#anthropic_vision_examplepy)
8. [anthropic_embeddings_example.py](#anthropic_embeddings_examplepy)
9. [claude_vertex_ai_example.py](#claude_vertex_ai_examplepy)

---

## anthropic_mcp_connector_example.py

### Overview

This script demonstrates how to use Anthropic's MCP connector feature to connect to remote MCP servers directly from the Messages API. The MCP connector simplifies integration with external tools by handling the server connection and communication protocol.

### Key Components

#### 1. Client Setup

```python
from anthropic import Anthropic

# Initialize the Anthropic client
client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)
```

#### 2. MCP Server Configuration

```python
# Configure MCP servers
mcp_servers = [
    {
        "server_url": "https://mcp-example-server.com/",
        "name": "example-server",
        "auth": {
            "type": "oauth",
            "token": "YOUR_OAUTH_TOKEN"
        }
    }
]
```

#### 3. Message Creation with MCP Servers

```python
# Create a message with MCP servers
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Hello, Claude!"}],
    mcp_servers=mcp_servers
)
```

### Usage Example

```python
# Complete example of using MCP connector
def use_mcp_connector():
    # Initialize the Anthropic client
    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    # Configure MCP servers
    mcp_servers = [
        {
            "server_url": "https://mcp-example-server.com/",
            "name": "example-server"
        }
    ]
    
    # Create a message with MCP servers
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[{"role": "user", "content": "Use the example-server to find information about Python best practices"}],
        mcp_servers=mcp_servers
    )
    
    # Process the response
    print(response.content)
```

### Implementation Notes

- The MCP connector requires the Anthropic API key to be set as an environment variable.
- Multiple MCP servers can be configured in a single request.
- OAuth authentication is supported for authenticated servers.
- The MCP connector handles the communication protocol between Claude and the MCP server.

---

## anthropic_mcp_example.py

### Overview

This script demonstrates how to set up the Anthropic API client with MCP tools and handle tool calls. It shows how to define tools, make API calls to Claude with tool definitions, and handle tool responses.

### Key Components

#### 1. Tool Definition

```python
# Define MCP tools
def get_mcp_tools():
    return [
        {
            "name": "mcp__crawl4ai-rag__perform_rag_query",
            "description": "Perform a RAG query to find relevant information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    ]
```

#### 2. Message Creation with Tools

```python
# Create a message with tools
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Find information about Python best practices"}],
    tools=get_mcp_tools()
)
```

#### 3. Tool Call Handling

```python
# Handle tool calls
def handle_tool_calls(response):
    if hasattr(response, "content") and len(response.content) > 0:
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_call = content_block.tool_use
                tool_name = tool_call.name
                tool_input = tool_call.input
                
                # Process the tool call
                tool_output = process_tool_call(tool_name, tool_input)
                
                # Continue the conversation with the tool output
                continue_conversation(response.id, tool_call.id, tool_output)
```

### Usage Example

```python
# Complete example of using MCP tools
def use_mcp_tools():
    # Initialize the Anthropic client
    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    # Create a message with tools
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[{"role": "user", "content": "Find information about Python best practices"}],
        tools=get_mcp_tools()
    )
    
    # Handle tool calls
    handle_tool_calls(response)
```

### Implementation Notes

- The tool definition includes a name, description, and input schema.
- The input schema defines the expected input parameters for the tool.
- Tool calls are handled by processing the tool input and continuing the conversation with the tool output.
- The conversation is continued by creating a new message with the tool output.

---

## anthropic_remote_mcp_example.py

### Overview

This script demonstrates how to connect to third-party remote MCP servers like Zapier, Stripe, etc. It shows how to configure multiple remote MCP servers and handle their responses.

### Key Components

#### 1. Remote MCP Server Configuration

```python
# Configure remote MCP servers
def get_remote_mcp_servers():
    return [
        {
            "server_url": "https://mcp.zapier.com/",
            "name": "zapier",
            "auth": {
                "type": "oauth",
                "token": os.environ.get("ZAPIER_OAUTH_TOKEN")
            }
        },
        {
            "server_url": "https://mcp.stripe.com/v1/sse",
            "name": "stripe",
            "auth": {
                "type": "oauth",
                "token": os.environ.get("STRIPE_OAUTH_TOKEN")
            }
        }
    ]
```

#### 2. Message Creation with Remote MCP Servers

```python
# Create a message with remote MCP servers
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Create a Zapier automation that sends an email when a new Stripe payment is received"}],
    mcp_servers=get_remote_mcp_servers()
)
```

#### 3. Handling Remote MCP Server Responses

```python
# Handle remote MCP server responses
def handle_remote_mcp_responses(response):
    if hasattr(response, "content") and len(response.content) > 0:
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_call = content_block.tool_use
                server_name = tool_call.name.split("__")[1]
                
                # Process the tool call based on the server name
                if server_name == "zapier":
                    process_zapier_tool_call(tool_call)
                elif server_name == "stripe":
                    process_stripe_tool_call(tool_call)
```

### Usage Example

```python
# Complete example of using remote MCP servers
def use_remote_mcp_servers():
    # Initialize the Anthropic client
    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    # Create a message with remote MCP servers
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[{"role": "user", "content": "Create a Zapier automation that sends an email when a new Stripe payment is received"}],
        mcp_servers=get_remote_mcp_servers()
    )
    
    # Handle remote MCP server responses
    handle_remote_mcp_responses(response)
```

### Implementation Notes

- Remote MCP servers are configured with a server URL, name, and authentication details.
- OAuth authentication is commonly used for remote MCP servers.
- The server name is extracted from the tool call name to determine which server to process.
- Each remote MCP server may have different processing requirements.

---

## anthropic_message_batch_example.py

### Overview

This script demonstrates how to use Anthropic's Message Batches API for processing multiple requests efficiently. It shows how to create a message batch, poll for completion, and retrieve results.

### Key Components

#### 1. Batch Creation

```python
# Create a message batch
batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": "request-1",
            "params": {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Write a short poem about AI"}
                ]
            }
        },
        {
            "custom_id": "request-2",
            "params": {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Explain quantum computing"}
                ]
            }
        }
    ]
)
```

#### 2. Polling for Batch Completion

```python
# Poll for batch completion
def poll_batch_completion(batch_id, max_attempts=10, delay=2):
    attempts = 0
    while attempts < max_attempts:
        # Get the batch status
        batch_status = client.messages.batches.retrieve(batch_id)
        
        # Check if the batch is complete
        if batch_status.status == "completed":
            return batch_status
        
        # Wait before the next attempt
        time.sleep(delay)
        attempts += 1
    
    raise Exception(f"Batch {batch_id} did not complete within the expected time")
```

#### 3. Retrieving Batch Results

```python
# Retrieve batch results
def retrieve_batch_results(batch_id):
    # Get the batch status
    batch_status = client.messages.batches.retrieve(batch_id)
    
    # Check if the batch is complete
    if batch_status.status != "completed":
        raise Exception(f"Batch {batch_id} is not complete")
    
    # Get the batch results
    results = {}
    for result in batch_status.results:
        custom_id = result.custom_id
        message = result.message
        results[custom_id] = message
    
    return results
```

### Usage Example

```python
# Complete example of using message batches
def use_message_batches():
    # Initialize the Anthropic client
    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    # Create a message batch
    batch = client.messages.batches.create(
        requests=[
            {
                "custom_id": "request-1",
                "params": {
                    "model": "claude-3-7-sonnet-20250219",
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "user", "content": "Write a short poem about AI"}
                    ]
                }
            },
            {
                "custom_id": "request-2",
                "params": {
                    "model": "claude-3-7-sonnet-20250219",
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "user", "content": "Explain quantum computing"}
                    ]
                }
            }
        ]
    )
    
    # Poll for batch completion
    batch_status = poll_batch_completion(batch.id)
    
    # Retrieve batch results
    results = retrieve_batch_results(batch.id)
    
    # Process the results
    for custom_id, message in results.items():
        print(f"Result for {custom_id}:")
        print(message.content)
```

### Implementation Notes

- Message batches allow processing multiple requests efficiently in parallel.
- Each request in a batch can have a custom ID for identification.
- Polling is used to check for batch completion.
- Batch results are retrieved once the batch is complete.
- The Message Batches API is particularly useful for high-throughput applications.

---

## claude_vertex_ai_example.py

### Overview

This script demonstrates how to use Anthropic's Claude models via Google Cloud's Vertex AI platform. It shows how to set up the Anthropic Vertex AI client, create messages, and process responses.

### Key Components

#### 1. Vertex AI Client Setup

```python
# Set up the Anthropic Vertex AI client
def setup_vertex_client(project_id, region="us-east5"):
    # Initialize the Anthropic Vertex client
    client = AnthropicVertex(
        project_id=project_id,
        region=region
    )
    
    return client
```

#### 2. Message Creation

```python
# Create a message using Claude on Vertex AI
def create_message(client, user_message, system_prompt=None, model="claude-opus-4@20250514", max_tokens=1024):
    try:
        # Create the message parameters
        message_params = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        }
        
        # Add system prompt if provided
        if system_prompt:
            message_params["system"] = system_prompt
        
        # Create the message
        response = client.messages.create(**message_params)
        
        return response
    except Exception as e:
        print(f"Error creating message: {e}")
        return {"error": str(e)}
```

#### 3. Response Processing

```python
# Process the response from Claude
def process_response(response):
    if hasattr(response, "error"):
        return f"Error: {response['error']}"
    
    # Extract text content from the response
    content = response.content
    text_content = ""
    
    for item in content:
        if item.get("type") == "text":
            text_content += item.get("text", "")
    
    return text_content
```

### Usage Example

```python
# Complete example of using Claude on Vertex AI
def use_claude_on_vertex_ai():
    # Set your Google Cloud project ID
    project_id = "YOUR_GOOGLE_CLOUD_PROJECT_ID"
    
    # Set the region where Claude is available
    region = "us-east5"
    
    # Set up the Anthropic Vertex client
    client = setup_vertex_client(project_id, region)
    
    # Example user message
    user_message = "What are the key differences between traditional machine learning and deep learning approaches?"
    
    # Example system prompt
    system_prompt = "You are an AI assistant with expertise in machine learning and data science. Provide clear, concise explanations with practical examples when possible."
    
    # Create a message
    response = create_message(
        client=client,
        user_message=user_message,
        system_prompt=system_prompt
    )
    
    # Process the response
    text_content = process_response(response)
    
    # Print the response
    print(text_content)
```

### Implementation Notes

- The Anthropic Vertex AI client requires a Google Cloud project ID and region.
- Google Cloud authentication is used for authentication.
- The client supports system prompts for better control of Claude's behavior.
- Response processing extracts text content from the response.
- Claude on Vertex AI is ideal for users already using Google Cloud.

---

## anthropic_streaming_example.py

### Overview

This script demonstrates how to use Anthropic's Claude API with streaming capabilities. Streaming allows you to receive Claude's response in real-time as it's being generated, rather than waiting for the complete response.

### Key Components

#### 1. Setting Up Streaming

```python
# Create a streaming message
stream = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[{"role": "user", "content": prompt}],
    stream=True
)
```

#### 2. Processing Streaming Events

```python
# Process streaming events
def process_streaming_events(stream):
    full_response = ""
    
    for event in stream:
        event_type = getattr(event, "type", None)
        
        if event_type == "content_block_delta":
            # Extract the text delta and add it to the response
            if event.delta.type == "text_delta":
                text_delta = event.delta.text
                full_response += text_delta
                print(text_delta, end="", flush=True)
    
    return full_response
```

#### 3. Streaming with Tool Use

```python
# Create a streaming message with tool use
stream = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[{"role": "user", "content": prompt}],
    tools=tools,
    stream=True
)
```

### Usage Example

```python
# Complete example of using streaming
def use_streaming():
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Create a streaming message
    prompt = "Explain the concept of streaming APIs in 3-4 sentences."
    
    stream = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    # Process the streaming events
    full_response = process_streaming_events(stream)
    
    print(f"\nFull response: {full_response}")
```

### Implementation Notes

- The `stream=True` parameter enables streaming mode
- Different event types in the stream include: `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, and `message_stop`
- The `content_block_delta` events contain the actual content being streamed
- Streaming works with all Claude features, including tool use

---

## anthropic_tool_use_example.py

### Overview

This script demonstrates how to use Anthropic's Claude API with tool use capabilities. It shows how to define tools, create messages with tools, process tool use responses, and implement a complete tool use loop.

### Key Components

#### 1. Defining Tools

```python
# Define tools for Claude to use
def define_tools():
    return [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    ]
```

#### 2. Creating a Message with Tools

```python
# Create a message with tools
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[{"role": "user", "content": prompt}],
    tools=tools
)
```

#### 3. Processing Tool Use Responses

```python
# Process tool use responses
def process_tool_use_response(response):
    tool_uses = []
    
    for content_block in response.content:
        if content_block.type == "tool_use":
            tool_use = {
                "id": content_block.id,
                "name": content_block.tool_use.name,
                "input": content_block.tool_use.input
            }
            tool_uses.append(tool_use)
    
    return tool_uses
```

#### 4. Continuing the Conversation with Tool Outputs

```python
# Continue the conversation with tool outputs
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=conversation_history,
    tool_outputs=[
        {
            "tool_call_id": tool_use["id"],
            "output": json.dumps(tool_output)
        }
    ]
)
```

### Usage Example

```python
# Complete example of a tool use loop
def run_tool_use_loop(client, initial_prompt, tools):
    # Initialize conversation history
    conversation_history = [
        {"role": "user", "content": initial_prompt}
    ]
    
    # Create the message with tools
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=conversation_history,
        tools=tools
    )
    
    # Add Claude's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response.content})
    
    # Process tool use requests
    tool_uses = process_tool_use_response(response)
    
    # Execute tools and prepare tool outputs
    tool_outputs = []
    for tool_use in tool_uses:
        tool_output = execute_tool(tool_use["name"], tool_use["input"])
        tool_outputs.append({
            "tool_call_id": tool_use["id"],
            "output": json.dumps(tool_output)
        })
    
    # Add an empty user message with tool outputs
    conversation_history.append({"role": "user", "content": ""})
    
    # Continue the conversation with tool outputs
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=conversation_history,
        tool_outputs=tool_outputs
    )
    
    # Add Claude's final response to the conversation history
    conversation_history.append({"role": "assistant", "content": response.content})
    
    return conversation_history
```

### Implementation Notes

- Tools are defined with a name, description, and input schema
- The input schema follows JSON Schema format and defines the expected input parameters
- Tool outputs must be provided as JSON strings
- The tool use loop can continue for multiple iterations if Claude needs to use multiple tools

---

## anthropic_vision_example.py

### Overview

This script demonstrates how to use Anthropic's Claude API with vision capabilities. It shows how to prepare images for Claude, create multimodal messages with text and images, and process multimodal responses.

### Key Components

#### 1. Encoding Images for Claude

```python
# Encode an image from a file path to base64 for Claude
def encode_image_from_file(image_path):
    # Determine media type based on file extension
    extension = Path(image_path).suffix.lower()
    media_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    
    media_type = media_type_map.get(extension)
    
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
```

#### 2. Creating a Multimodal Message

```python
# Create a multimodal message with text and images
def create_multimodal_message(client, text_prompt, image_data):
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
    
    # Create the message
    return client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": content}
        ]
    )
```

#### 3. Resizing Images if Needed

```python
# Resize an image if it's too large for Claude's API limits
def resize_image_if_needed(image_path, max_size=5242880):
    # Check file size
    file_size = Path(image_path).stat().st_size
    
    # If file is small enough, use the direct encoding
    if file_size <= max_size:
        return encode_image_from_file(image_path)
    
    # File is too large, resize it
    image = Image.open(image_path)
    
    # Calculate new dimensions while maintaining aspect ratio
    width, height = image.size
    aspect_ratio = width / height
    
    # Reduce quality until the image is small enough
    quality = 85
    format = "JPEG"
    
    while True:
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=quality)
        current_size = len(buffer.getvalue())
        
        if current_size <= max_size:
            break
        
        # Reduce quality or dimensions
        if quality > 50:
            quality -= 10
        else:
            width = int(width * 0.75)
            height = int(width / aspect_ratio)
            image = image.resize((width, height), Image.LANCZOS)
            quality = 85
```

### Usage Example

```python
# Complete example of using Claude's vision capabilities
def use_vision():
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Encode an image
    image_data = encode_image_from_file("example_image.jpg")
    
    # Create a multimodal message
    text_prompt = "What's in this image? Please describe it in detail."
    
    response = create_multimodal_message(
        client=client,
        text_prompt=text_prompt,
        image_data=image_data
    )
    
    # Extract and print the response
    for content_block in response.content:
        if content_block.type == "text":
            print(content_block.text)
```

### Implementation Notes

- Images must be encoded as base64 strings with the appropriate media type
- Claude has a maximum file size limit (currently 5MB per image)
- Multiple images can be included in a single message
- Claude can analyze images, extract text (OCR), and compare multiple images

---

## anthropic_embeddings_example.py

### Overview

This script demonstrates how to use Anthropic's Claude API for embeddings. It shows how to generate embeddings for text, compare embeddings for semantic similarity, and build a simple RAG system with embeddings.

### Key Components

#### 1. Generating Embeddings

```python
# Generate an embedding for a text
def generate_embedding(client, text, model="claude-3-embedding-20240229"):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    
    return response.embedding
```

#### 2. Generating Batch Embeddings

```python
# Generate embeddings for multiple texts in a single API call
def generate_batch_embeddings(client, texts, model="claude-3-embedding-20240229"):
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    
    return response.embeddings
```

#### 3. Calculating Cosine Similarity

```python
# Calculate cosine similarity between two vectors
def cosine_similarity(v1, v2):
    # Convert to numpy arrays
    v1_array = np.array(v1)
    v2_array = np.array(v2)
    
    # Calculate cosine similarity
    dot_product = np.dot(v1_array, v2_array)
    norm_v1 = np.linalg.norm(v1_array)
    norm_v2 = np.linalg.norm(v2_array)
    
    return dot_product / (norm_v1 * norm_v2)
```

#### 4. Building a Simple RAG System

```python
# Create a simple RAG system using Claude embeddings
def create_simple_rag_system(client, documents, query):
    # Generate embeddings for documents
    document_embeddings = []
    for doc in documents:
        embedding = generate_embedding(client, doc)
        document_embeddings.append((doc, embedding))
    
    # Generate embedding for the query
    query_embedding = generate_embedding(client, query)
    
    # Find most similar documents
    similar_docs = find_most_similar(query_embedding, document_embeddings)
    
    # Create context from similar documents
    context = "\n\n".join([doc for doc, _ in similar_docs])
    
    # Generate response with context
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        system="Answer the question based on the provided context.",
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    
    return response
```

### Usage Example

```python
# Complete example of comparing semantically similar texts
def compare_texts():
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Define semantically similar and dissimilar texts
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "A fast auburn fox leaps above the sleepy canine."  # Similar meaning
    text3 = "Machine learning models require significant computational resources."  # Different meaning
    
    # Generate embeddings
    embedding1 = generate_embedding(client, text1)
    embedding2 = generate_embedding(client, text2)
    embedding3 = generate_embedding(client, text3)
    
    # Calculate similarities
    similarity_1_2 = cosine_similarity(embedding1, embedding2)
    similarity_1_3 = cosine_similarity(embedding1, embedding3)
    
    print(f"Similarity between Text 1 and Text 2: {similarity_1_2:.4f}")
    print(f"Similarity between Text 1 and Text 3: {similarity_1_3:.4f}")
```

### Implementation Notes

- Claude embeddings capture semantic meaning, not just lexical similarity
- Embeddings can be used for semantic search, clustering, and classification
- The embedding model (`claude-3-embedding-20240229`) is different from the chat models
- Batch embedding is more efficient for processing multiple texts
- Embeddings are the foundation for building RAG (Retrieval Augmented Generation) systems

---

## Additional Resources

- [Anthropic API Documentation](https://docs.anthropic.com/en/api/overview)
- [MCP Connector Documentation](https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector)
- [Remote MCP Servers](https://docs.anthropic.com/en/docs/agents-and-tools/remote-mcp-servers)
- [Message Batches Documentation](https://docs.anthropic.com/en/api/messages-batch-examples)
- [Streaming Documentation](https://docs.anthropic.com/en/docs/build-with-claude/streaming)
- [Tool Use Documentation](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use)
- [Vision Documentation](https://docs.anthropic.com/en/docs/build-with-claude/vision)
- [Embeddings Documentation](https://docs.anthropic.com/en/docs/build-with-claude/embeddings)
- [Claude on Vertex AI](https://docs.anthropic.com/en/api/claude-on-vertex-ai)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
