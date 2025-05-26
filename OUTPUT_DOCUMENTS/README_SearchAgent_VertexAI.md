# Configuring ADK Search Agent with Vertex AI: Troubleshooting and Lessons Learned

This document outlines the steps taken and lessons learned while configuring a Google Agent Development Kit (ADK) Python agent (`search_agent.py`) to use Google Vertex AI for its language model capabilities, specifically leveraging a Gemini model.

## Project Goal

The primary objective was to enable the `search_agent` to utilize a specific Gemini model hosted on Vertex AI, configured via environment variables, allowing it to perform Google searches and process results.

## Initial Problem

The agent initially failed to connect to or utilize the specified Vertex AI model, primarily manifesting as a `ValueError: No model found for search_agent` or similar errors indicating that the ADK framework could not identify or configure the language model correctly, despite attempts to pass model configurations directly in the Python code.

## Troubleshooting Steps & Lessons Learned

The path to a working configuration involved several iterations and uncovered key aspects of how ADK interacts with Vertex AI and environment variables:

1.  **`ModelConfig` vs. Direct Model String:**
    *   **Initial Approach:** Explicitly creating a `ModelConfig` object within `search_agent.py`, populating it with `model_name`, `use_vertex=True`, `project`, and `location` read from environment variables.
    *   **Observation:** While logically sound, this didn't immediately resolve the issue, suggesting ADK might have overriding or preceding checks based on environment variables.
    *   **Final Approach:** Simplifying the `LlmAgent` initialization to pass only the `model` name as a string (e.g., `model="gemini-2.5-flash-preview-05-20"`). This relies more heavily on ADK's internal mechanisms to detect Vertex AI usage via specific environment variables.
    *   **Lesson:** For Vertex AI with ADK, if the standard environment variables are correctly set, ADK can often infer the necessary configuration, and a simpler agent initialization (passing just the model string) can be more robust.

2.  **Correct Environment Variable Naming (The Crucial `_AI_`):**
    *   **Initial `.env` setup:** Used `VERTEX_MODEL_NAME` to specify the Gemini model ID.
    *   **Problem:** ADK's internal `envs.py` module, when `GOOGLE_GENAI_USE_VERTEXAI=TRUE` is set, specifically looks for `VERTEX_AI_MODEL_NAME` (note the `_AI_`).
    *   **Correction:** Renamed `VERTEX_MODEL_NAME` to `VERTEX_AI_MODEL_NAME` in the `.env` file.
    *   **Lesson:** Adherence to ADK's expected environment variable names is critical. A subtle difference like `VERTEX_MODEL_NAME` vs. `VERTEX_AI_MODEL_NAME` can prevent the model from being recognized.

3.  **Ensuring `GOOGLE_GENAI_USE_VERTEXAI=TRUE`:**
    *   **Observation:** This variable is the primary switch for ADK to attempt Vertex AI integration. If missing or `FALSE`, ADK will likely try to use other Google AI services (like the Generative Language API via `GOOGLE_API_KEY`) or fail if no other model is configured.
    *   **Verification:** Ensured this line was present and correctly set to `TRUE` in the `.env` file.
    *   **Lesson:** This is a foundational setting for using Vertex AI with ADK.

4.  **Google Cloud Authentication (`gcloud auth application-default login`):**
    *   **Problem:** Even with correct code and `.env` settings, the agent failed with a `Reauthentication is needed. Please run 'gcloud auth application-default login' to reauthenticate.` error.
    *   **Correction:** Executed `gcloud auth application-default login` in the terminal. This command authenticates the local environment to Google Cloud, creating credentials that Application Default Credentials (ADC) can use.
    *   **Lesson:** Python client libraries for Google Cloud (including those used by ADK for Vertex AI) often rely on ADC. Running `gcloud auth application-default login` is essential for local development to provide these credentials.

5.  **Other Standard Vertex AI Environment Variables:**
    *   `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID.
    *   `GOOGLE_CLOUD_LOCATION`: The region for your Vertex AI resources (e.g., `us-central1`).
    *   **Lesson:** These must be correctly set in the `.env` file for ADK to target the correct Vertex AI endpoint and project.

## Final Working Configuration Summary

**`c:\Users\kevin\repos\TOOLS\mcp-crawl4ai-rag\.env` (Relevant Lines):**
```env
# Google Cloud & Gemini API Configuration
GOOGLE_CLOUD_PROJECT=kev-agents-in-cloud

# Vertex AI Configuration
GOOGLE_CLOUD_LOCATION=us-central1
VERTEX_AI_MODEL_NAME=gemini-2.5-flash-preview-05-20
GOOGLE_GENAI_USE_VERTEXAI=TRUE

# Other variables like HOST, PORT, API keys for other services...
```

**`c:\Users\kevin\repos\TOOLS\mcp-crawl4ai-rag\agents\search_agent.py` (Key Snippets):**
```python
import os
from pathlib import Path
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

# Load .env file from the project root
project_root = Path(__file__).resolve().parents[1] # Assuming agents/ is one level down
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path=dotenv_path)
print(f"Loading environment variables from: {dotenv_path}")

# Model ID to be used by the LlmAgent, read from VERTEX_AI_MODEL_NAME in .env
AGENT_MODEL_ID_FROM_ENV = os.getenv("VERTEX_AI_MODEL_NAME", "gemini-1.5-flash-001")

def create_search_agent() -> LlmAgent:
    agent = LlmAgent(
        model=AGENT_MODEL_ID_FROM_ENV, # Pass model name as string
        tools=[
            google_search
        ],
        instruction=(
            "You are a helpful search agent. When asked a question, the Google Search tool will be "
            "automatically invoked by the model to find relevant information. Present the findings in a clear "
            "and concise way. Always cite your sources with links if provided by the search tool. "
            "If the user asks about your capabilities, explain that you can search the web using Google Search "
            "and provide summarized information with source citations. For complex queries, the model will break "
            "them down into specific search terms to get the most relevant results."
        )
    )
    return agent

if __name__ == "__main__":
    print("Search agent initialized and ready to use.")
    print(f"  Model ID (from VERTEX_AI_MODEL_NAME in .env): {AGENT_MODEL_ID_FROM_ENV}")
    print(f"  Relevant env vars for Vertex: GOOGLE_GENAI_USE_VERTEXAI='{os.getenv('GOOGLE_GENAI_USE_VERTEXAI')}', GOOGLE_CLOUD_PROJECT='{os.getenv('GOOGLE_CLOUD_PROJECT')}', GOOGLE_CLOUD_LOCATION='{os.getenv('GOOGLE_CLOUD_LOCATION')}'")
```

By ensuring these environment variables were correctly named and set, and that Google Cloud authentication was established via `gcloud auth application-default login`, the ADK agent was successfully able to use the specified Gemini model on Vertex AI.
