# Google ADK Search Agent with Gemini 2.5 Pro

A powerful web search agent built with Google's Agent Development Kit (ADK) that leverages Gemini 2.5 Pro on Vertex AI to search the web and provide intelligent, summarized results.

## Features

- Web search capabilities using ADK's built-in search tools
- Powered by Gemini 2.5 Pro on Vertex AI
- Result summarization and formatting
- Interactive command-line interface
- ADK web interface integration
- Vertex AI deployment support

## Setup

### Prerequisites

- Python 3.9+
- Google Cloud project with Vertex AI API enabled
- Google API key for Gemini access
- Google Cloud service account with Vertex AI permissions

### Installation

1. Clone this repository or create a new directory for the project

2. Install dependencies using uv:

```bash
uv add -r requirements.txt
```

3. Set up your Google Cloud and Gemini credentials in your `.env` file:

```
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_CLOUD_PROJECT=your_gcp_project_id_here
GOOGLE_APPLICATION_CREDENTIALS=path_to_your_service_account_json_file
VERTEX_LOCATION=us-central1
VERTEX_MODEL_NAME=gemini-2.5-pro-preview-05-06
```

## Usage

### Running from Command Line

Run the agent directly:

```bash
uv run python -m agent.py
```

### Running with ADK Web Interface

1. Start the ADK web interface:

```bash
uv run adk web --project-path .
```

2. Navigate to http://localhost:8000 in your browser
3. Select the search_agent from the list of available agents
4. Start interacting with the agent through the web interface

### Deploying to Vertex AI

Deploy the agent to Vertex AI:

```bash
uv run python -m agent.py --deploy
```

This will:
1. Create a new Vertex AI endpoint with the agent
2. Configure the deployment with Gemini 2.5 Pro
3. Output the deployment details including the endpoint URL

## Example Queries

- "What are the latest developments in quantum computing?"
- "Find information about climate change solutions"
- "Search for recipes with avocado and chicken"

## Project Structure

```
search_agent_project/
├── __init__.py          # Package initialization
├── agent.py             # Main agent implementation
├── requirements.txt     # Project dependencies
└── README.md            # This documentation
```

## Lessons Learned During Development

### 1. SDK Migration and Deprecation

- **Use `google-genai` instead of `google-generativeai`**: The `google-generativeai` package is deprecated. Google has unified its GenAI SDKs into a single package called `google-genai`. This new SDK supports all Google's generative AI models (Gemini, Veo, Imagen, etc.).

- **Version Compatibility**: Ensure you're using compatible versions of `google-adk` and `google-genai`. The Google AI ecosystem is evolving rapidly, and version mismatches can cause subtle integration issues.

### 2. ADK Project Structure

- **Proper Project Structure**: ADK requires a specific project structure to discover agents. Ensure your agent is properly exported in the `__init__.py` file.

- **Agent Discovery**: When running the ADK web interface, use the `--project-path` flag to specify where your agent is located: `adk web --project-path .`

### 3. Vertex AI Integration

- **Authentication Methods**: There are two ways to authenticate with Gemini models:
  - Direct API key for simple use cases
  - Vertex AI for production deployments (requires GCP project and service account)

- **Environment Variables**: Store configuration in environment variables or a `.env` file rather than hardcoding values.

### 4. Dependency Management

- **Regular Updates**: Google's AI libraries are evolving rapidly. Regularly update dependencies to access new features and bug fixes.

- **Specific Versions**: Pin specific versions in production to avoid unexpected changes.

- **Use uv**: Package management with uv provides faster and more reliable dependency resolution than pip.

## Maintenance

To keep your dependencies up to date:

```bash
uv add --upgrade-package google-adk --upgrade-package google-genai --upgrade-package google-cloud-aiplatform
```

Or to upgrade all packages:

```bash
uv pip install --upgrade -r requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google ADK team for creating a powerful agent development framework
- Google Gemini team for the advanced language model capabilities
