# Google ADK Search Agent with Gemini 2.5 Pro on Vertex AI

A web search agent built with Google's Agent Development Kit (ADK) that uses Gemini 2.5 Pro on Vertex AI to search the web and summarize results.

## Features

- Web search capabilities using ADK's built-in search tools
- Powered by Gemini 2.5 Pro on Vertex AI
- Result summarization and formatting
- Interactive command-line interface
- Can be run using ADK web interface
- Deployable to Vertex AI

## Setup

### Prerequisites

- Python 3.9+
- Google API key for Gemini model access
- Google Cloud project with Vertex AI API enabled
- Google Cloud service account with Vertex AI permissions

### Installation

1. Install dependencies using uv:

```bash
uv add -r requirements.txt
```

2. Set up your Google Cloud and Gemini credentials:

Ensure the following environment variables are set in your `.env` file:

```
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_CLOUD_PROJECT=your_gcp_project_id_here
GOOGLE_APPLICATION_CREDENTIALS=path_to_your_service_account_json_file
VERTEX_LOCATION=us-central1
VERTEX_MODEL_NAME=gemini-2-5-pro
```

## Usage

### Running from Command Line

Run the agent directly:

```bash
uv run python -m search_agent.agent
```

### Running with ADK Web

1. Start the ADK web interface:

```bash
uv run adk web
```

2. Navigate to http://localhost:8080 in your browser
3. Select the search_agent from the list of available agents
4. Start interacting with the agent through the web interface

### Deploying to Vertex AI

Deploy the agent to Vertex AI:

```bash
uv run python -m search_agent.agent --deploy
```

This will:
1. Create a new Vertex AI endpoint with the agent
2. Configure the deployment with Gemini 2.5 Pro
3. Output the deployment details including the endpoint URL

## Example Queries

- "What are the latest developments in quantum computing?"
- "Find information about climate change solutions"
- "Search for recipes with avocado and chicken"

## Customization

You can modify the agent's behavior by editing the instructions in `agent.py`. You can also add additional tools or change the search providers as needed.

## License

This project is licensed under the terms of the MIT license.
