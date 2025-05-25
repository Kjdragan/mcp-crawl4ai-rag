# Google ADK with Gemini Integration

This repository contains sample scripts and documentation for using the Google Agent Development Kit (ADK) with Gemini models.

## Overview

The Google Agent Development Kit (ADK) is an open-source, code-first toolkit for building, evaluating, and deploying sophisticated AI agents. This repository demonstrates how to integrate Gemini models with ADK to create powerful AI agents.

## Contents

- `simple_gemini_agent.py` - A basic agent using Gemini with ADK
- `gemini_with_tools.py` - An agent with custom tools integration
- `streaming_gemini_agent.py` - A streaming agent example
- `multi_agent_system.py` - A multi-agent system using Gemini
- `evaluation_example.py` - How to evaluate Gemini agents
- `requirements.txt` - Required dependencies

## Getting Started

1. Install the required dependencies:
   ```
   uv add -r requirements.txt
   ```

2. Set up your API keys in the `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. Run the simple agent example:
   ```
   uv run simple_gemini_agent.py
   ```

## Documentation

For more detailed information on using ADK with Gemini, see the following resources:
- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [ADK Samples Repository](https://github.com/google/adk-samples)
