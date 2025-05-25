"""
Gemini Agent Evaluation Example

This script demonstrates how to evaluate Gemini agents using Google's Agent Development Kit
evaluation framework.
"""

import asyncio
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel
from google.adk.agents import Agent
from google.adk.llms import GeminiLLM
from google.adk.sessions import Session
from google.adk.evaluation import (
    EvaluationDataset,
    EvaluationExample,
    EvaluationMetric,
    EvaluationResult,
    Evaluator
)

# Load environment variables
load_dotenv()

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Define evaluation metrics
class RelevanceMetric(EvaluationMetric):
    """Metric to evaluate the relevance of agent responses."""
    
    name = "relevance"
    description = "Evaluates how relevant the response is to the query"
    
    async def evaluate(self, example: EvaluationExample, response: Dict[str, Any]) -> float:
        # Create an evaluator LLM
        evaluator_llm = GeminiLLM(
            model="gemini-2.0-flash",
            api_key=GEMINI_API_KEY,
            system_instruction=(
                "You are an objective evaluator of AI assistant responses. "
                "Rate the relevance of the response to the query on a scale of 0.0 to 1.0, "
                "where 0.0 is completely irrelevant and 1.0 is perfectly relevant."
            )
        )
        
        # Create a temporary agent for evaluation
        eval_agent = Agent(
            name="RelevanceEvaluator",
            description="Evaluates response relevance",
            llm=evaluator_llm
        )
        
        # Prepare the evaluation prompt
        eval_prompt = (
            f"Query: {example.input}\n\n"
            f"Response: {response.get('text', '')}\n\n"
            "Rate the relevance of this response on a scale of 0.0 to 1.0. "
            "Provide only the numerical score."
        )
        
        # Get the evaluation score
        eval_response = await eval_agent.process(eval_prompt)
        
        try:
            # Extract the numerical score
            score = float(eval_response.text.strip())
            # Ensure the score is in the valid range
            score = max(0.0, min(1.0, score))
            return score
        except ValueError:
            # Default score if parsing fails
            return 0.5

class AccuracyMetric(EvaluationMetric):
    """Metric to evaluate the factual accuracy of agent responses."""
    
    name = "accuracy"
    description = "Evaluates the factual accuracy of the response"
    
    async def evaluate(self, example: EvaluationExample, response: Dict[str, Any]) -> float:
        # Create an evaluator LLM
        evaluator_llm = GeminiLLM(
            model="gemini-2.0-flash",
            api_key=GEMINI_API_KEY,
            system_instruction=(
                "You are an objective evaluator of AI assistant responses. "
                "Rate the factual accuracy of the response on a scale of 0.0 to 1.0, "
                "where 0.0 is completely inaccurate and 1.0 is perfectly accurate."
            )
        )
        
        # Create a temporary agent for evaluation
        eval_agent = Agent(
            name="AccuracyEvaluator",
            description="Evaluates response accuracy",
            llm=evaluator_llm
        )
        
        # Prepare the evaluation prompt
        eval_prompt = (
            f"Query: {example.input}\n\n"
            f"Response: {response.get('text', '')}\n\n"
            f"Expected information (reference): {example.expected_output}\n\n"
            "Rate the factual accuracy of this response on a scale of 0.0 to 1.0. "
            "Provide only the numerical score."
        )
        
        # Get the evaluation score
        eval_response = await eval_agent.process(eval_prompt)
        
        try:
            # Extract the numerical score
            score = float(eval_response.text.strip())
            # Ensure the score is in the valid range
            score = max(0.0, min(1.0, score))
            return score
        except ValueError:
            # Default score if parsing fails
            return 0.5

async def main():
    # Create a test dataset
    dataset = EvaluationDataset(
        name="GeminiAgentTestDataset",
        description="Test dataset for evaluating Gemini agents",
        examples=[
            EvaluationExample(
                input="What is the capital of France?",
                expected_output="The capital of France is Paris."
            ),
            EvaluationExample(
                input="Explain how photosynthesis works.",
                expected_output="Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. It converts carbon dioxide and water into glucose and oxygen using energy from sunlight."
            ),
            EvaluationExample(
                input="Write a short poem about technology.",
                expected_output="A creative poem about technology that captures its impact on modern life."
            ),
            EvaluationExample(
                input="What are the main features of Python?",
                expected_output="Python features include: easy to learn syntax, interpreted language, dynamically typed, object-oriented, high-level, portable, extensive libraries, and strong community support."
            )
        ]
    )
    
    # Create the agent to evaluate
    llm = GeminiLLM(
        model="gemini-2.0-flash",
        api_key=GEMINI_API_KEY,
        system_instruction=(
            "You are a helpful assistant that provides clear, concise, and accurate information. "
            "Answer questions directly and provide relevant details."
        )
    )
    
    agent = Agent(
        name="GeminiAgentUnderTest",
        description="A Gemini agent being evaluated",
        llm=llm
    )
    
    # Create the evaluator with metrics
    evaluator = Evaluator(
        metrics=[RelevanceMetric(), AccuracyMetric()]
    )
    
    # Run the evaluation
    print("Starting evaluation of Gemini agent...")
    results = await evaluator.evaluate(agent, dataset)
    
    # Print the results
    print("\nEvaluation Results:")
    print("-" * 50)
    
    # Print overall scores
    print("Overall Scores:")
    for metric_name, score in results.overall_scores.items():
        print(f"- {metric_name}: {score:.2f}")
    
    # Print individual example results
    print("\nIndividual Example Results:")
    for i, example_result in enumerate(results.example_results):
        print(f"\nExample {i+1}: {example_result.example.input}")
        print(f"Response: {example_result.response.get('text', '')[:100]}...")
        print("Scores:")
        for metric_name, score in example_result.scores.items():
            print(f"- {metric_name}: {score:.2f}")
    
    # Save results to file
    with open("evaluation_results.json", "w") as f:
        # Convert results to a serializable format
        serializable_results = {
            "overall_scores": results.overall_scores,
            "example_results": [
                {
                    "example": {
                        "input": er.example.input,
                        "expected_output": er.example.expected_output
                    },
                    "response": er.response,
                    "scores": er.scores
                }
                for er in results.example_results
            ]
        }
        json.dump(serializable_results, f, indent=2)
    
    print("\nEvaluation results saved to evaluation_results.json")

if __name__ == "__main__":
    asyncio.run(main())
