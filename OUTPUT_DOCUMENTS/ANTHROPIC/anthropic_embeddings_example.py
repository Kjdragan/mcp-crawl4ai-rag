#!/usr/bin/env python3
"""
Example script demonstrating how to use Anthropic's Claude API for embeddings.

This example shows how to:
1. Set up the Anthropic client
2. Generate embeddings for text
3. Compare embeddings for semantic similarity
4. Build a simple RAG system with embeddings

Requirements:
- anthropic Python package
- numpy for vector operations

Install with: 
uv add anthropic
uv add numpy
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

from anthropic import Anthropic


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


def generate_embedding(
    client: Anthropic,
    text: str,
    model: str = "claude-3-embedding-20240229"
) -> List[float]:
    """
    Generate an embedding for a text using Claude.
    
    Args:
        client: Anthropic client
        text: Text to generate embedding for
        model: Embedding model to use
        
    Returns:
        The embedding vector
    """
    try:
        # Create the embedding
        response = client.embeddings.create(
            model=model,
            input=text
        )
        
        # Return the embedding vector
        return response.embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise


def generate_batch_embeddings(
    client: Anthropic,
    texts: List[str],
    model: str = "claude-3-embedding-20240229"
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in a single API call.
    
    Args:
        client: Anthropic client
        texts: List of texts to generate embeddings for
        model: Embedding model to use
        
    Returns:
        List of embedding vectors
    """
    try:
        # Create the embeddings
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        
        # Return the embedding vectors
        return response.embeddings
    except Exception as e:
        print(f"Error generating batch embeddings: {e}")
        raise


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Cosine similarity (between -1 and 1)
    """
    # Convert to numpy arrays
    v1_array = np.array(v1)
    v2_array = np.array(v2)
    
    # Calculate cosine similarity
    dot_product = np.dot(v1_array, v2_array)
    norm_v1 = np.linalg.norm(v1_array)
    norm_v2 = np.linalg.norm(v2_array)
    
    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    return dot_product / (norm_v1 * norm_v2)


def find_most_similar(
    query_embedding: List[float],
    document_embeddings: List[Tuple[str, List[float]]],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Find the most similar documents to a query embedding.
    
    Args:
        query_embedding: Query embedding vector
        document_embeddings: List of tuples (document_text, embedding_vector)
        top_k: Number of top results to return
        
    Returns:
        List of tuples (document_text, similarity_score)
    """
    # Calculate similarities
    similarities = []
    for doc_text, doc_embedding in document_embeddings:
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((doc_text, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k results
    return similarities[:top_k]


def create_simple_rag_system(
    client: Anthropic,
    documents: List[str],
    query: str,
    model: str = "claude-3-opus-20240229",
    embedding_model: str = "claude-3-embedding-20240229",
    max_tokens: int = 1024
) -> Dict[str, Any]:
    """
    Create a simple RAG system using Claude embeddings.
    
    Args:
        client: Anthropic client
        documents: List of documents to search
        query: User query
        model: Claude model to use for generation
        embedding_model: Claude model to use for embeddings
        max_tokens: Maximum tokens to generate
        
    Returns:
        The response from Claude
    """
    # Generate embeddings for documents
    print(f"Generating embeddings for {len(documents)} documents...")
    document_embeddings = []
    
    # Process in batches of 10 to avoid rate limits
    batch_size = 10
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_embeddings = generate_batch_embeddings(client, batch, model=embedding_model)
        
        for j, embedding in enumerate(batch_embeddings):
            document_embeddings.append((batch[j], embedding))
    
    # Generate embedding for the query
    print("Generating embedding for query...")
    query_embedding = generate_embedding(client, query, model=embedding_model)
    
    # Find most similar documents
    print("Finding most similar documents...")
    similar_docs = find_most_similar(query_embedding, document_embeddings)
    
    # Create context from similar documents
    context = "\n\n".join([doc for doc, _ in similar_docs])
    
    # Create prompt with context
    system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context. 
If the context doesn't contain the information needed to answer the question, say so."""
    
    user_prompt = f"""Context:
{context}

Question: {query}

Please answer the question based on the context provided."""
    
    # Generate response
    print("Generating response...")
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return {
        "query": query,
        "similar_documents": similar_docs,
        "response": response
    }


def main():
    """Main function to demonstrate Claude embeddings."""
    # Set up the Anthropic client
    client = setup_client()
    
    print("Example 1: Basic Embeddings")
    print("-------------------------")
    
    # Generate embeddings for a single text
    text = "Embeddings are vector representations of text that capture semantic meaning."
    
    try:
        embedding = generate_embedding(client, text)
        
        # Print the first 10 dimensions of the embedding
        print(f"Generated embedding with {len(embedding)} dimensions")
        print(f"First 10 dimensions: {embedding[:10]}")
        
        # Example 2: Comparing Semantically Similar Texts
        print("\nExample 2: Comparing Semantically Similar Texts")
        print("-------------------------------------------")
        
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
        
        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"Text 3: {text3}")
        print(f"Similarity between Text 1 and Text 2: {similarity_1_2:.4f}")
        print(f"Similarity between Text 1 and Text 3: {similarity_1_3:.4f}")
        
        # Example 3: Batch Embeddings
        print("\nExample 3: Batch Embeddings")
        print("------------------------")
        
        # Define a batch of texts
        batch_texts = [
            "Artificial intelligence is transforming industries.",
            "Machine learning algorithms improve with more data.",
            "Neural networks are inspired by the human brain.",
            "Deep learning has revolutionized computer vision.",
            "Natural language processing helps computers understand text."
        ]
        
        # Generate batch embeddings
        batch_embeddings = generate_batch_embeddings(client, batch_texts)
        
        print(f"Generated {len(batch_embeddings)} embeddings in a single API call")
        print(f"Each embedding has {len(batch_embeddings[0])} dimensions")
        
        # Example 4: Simple RAG System
        print("\nExample 4: Simple RAG System")
        print("-------------------------")
        
        # Define a set of documents
        documents = [
            "Python is a high-level, interpreted programming language known for its readability and simplicity.",
            "Python was created by Guido van Rossum and first released in 1991.",
            "Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming.",
            "JavaScript is a scripting language that enables interactive web pages and is an essential part of web applications.",
            "JavaScript was created by Brendan Eich in 1995 while he was working at Netscape Communications Corporation.",
            "Unlike Python, JavaScript is primarily used for client-side web development, though it can also be used on the server side with Node.js.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "Deep learning is a subset of machine learning that uses neural networks with many layers (hence 'deep') to analyze various factors of data.",
            "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language."
        ]
        
        # Define a query
        query = "When was Python created and by whom?"
        
        # Create a simple RAG system
        rag_result = create_simple_rag_system(client, documents, query)
        
        # Print the results
        print(f"\nQuery: {rag_result['query']}")
        print("\nTop similar documents:")
        for i, (doc, similarity) in enumerate(rag_result['similar_documents']):
            print(f"{i+1}. Similarity: {similarity:.4f}")
            print(f"   {doc}")
        
        print("\nResponse:")
        for content_block in rag_result['response'].content:
            if content_block.type == "text":
                print(content_block.text)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
