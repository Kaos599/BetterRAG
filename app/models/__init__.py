from typing import Dict, Any, Union, Optional, List
from app.models.azure_openai import AzureOpenAIConnector
from app.models.gemini import GeminiConnector
import numpy as np
import random
import time


class MockModelConnector:
    """
    Mock model connector that provides fake embeddings and text generation.
    Used for testing without real API keys.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize mock model connector.
        
        Args:
            config: Configuration dictionary (not used, but kept for interface compatibility)
        """
        self.embedding_dim = 1536  # Standard OpenAI embedding dimension
        print("Using MockModelConnector for testing without real API calls")
        
    def initialize(self) -> bool:
        """
        Mock initialization method.
        
        Returns:
            bool: Always returns True for mock connector
        """
        return True
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate fake embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (random but deterministic based on text content)
        """
        # Create deterministic but random-looking embeddings
        embeddings = []
        for text in texts:
            # Use text hash as random seed for deterministic results
            seed = hash(text) % 10000
            np.random.seed(seed)
            
            # Generate a random embedding vector
            embedding = np.random.randn(self.embedding_dim).astype(float)
            # Normalize to unit length
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())
        
        # Simulate API latency
        time.sleep(0.05)
        return embeddings
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get a single embedding for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0]
    
    def generate_text(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate fake text response.
        
        Args:
            prompt: The query or prompt
            context: Optional context to include
            
        Returns:
            Generated text (a mock response)
        """
        # Simulate API latency
        time.sleep(0.2)
        
        # Create a deterministic but varied response based on the prompt and context
        if context:
            # Extract some content from the context to make the response look relevant
            words = context.split()
            if len(words) > 20:
                keywords = [words[i] for i in range(0, len(words), len(words) // 10) if i < len(words)]
                response = f"This is a mock response for testing. Based on the context, I can tell you about {', '.join(keywords[:5])}."
            else:
                response = f"This is a mock response for testing. The context is too short to extract meaningful information."
        else:
            response = f"This is a mock response for testing without context. Your query was about '{prompt[:30]}...'."
            
        return response
        
    def evaluate_batch(self, queries: List[str], contexts: Optional[List[str]] = None, max_workers: int = 4) -> List[str]:
        """
        Evaluate multiple queries in parallel (mock implementation).
        
        Args:
            queries: List of query texts to evaluate
            contexts: Optional list of contexts for each query
            max_workers: Maximum number of parallel workers (not used in mock)
            
        Returns:
            List of generated responses
        """
        # Simulate parallel processing but with sequential execution for simplicity
        results = []
        
        for i, query in enumerate(queries):
            context = contexts[i] if contexts and i < len(contexts) else None
            result = self.generate_text(query, context)
            results.append(result)
            
        return results


def is_dummy_api_key(api_key: str) -> bool:
    """
    Check if the API key is a dummy key for testing.
    
    Args:
        api_key: API key to check
        
    Returns:
        bool: True if the key is a dummy key, False otherwise
    """
    dummy_patterns = ["dummy", "test", "fake", "DUMMY", "TEST", "FAKE", "1234", "0000"]
    return any(pattern in api_key for pattern in dummy_patterns)


def get_model_connector(config: Dict[str, Any], model_name: Optional[str] = None) -> Union[AzureOpenAIConnector, GeminiConnector, MockModelConnector]:
    """
    Factory function to get the appropriate model connector based on configuration.
    
    Args:
        config: Configuration dictionary
        model_name: Optional model name to override the one in config
        
    Returns:
        A model connector instance
        
    Raises:
        ValueError: If the requested model is not supported
    """
    # Get the model configuration
    model_config = config.get("model", {})
    provider = model_name or model_config.get("provider", "azure_openai")
    
    # Check if testing mode (dummy API keys) is active
    if provider == "azure_openai":
        azure_config = model_config.get("azure_openai", {})
        if not azure_config:
            raise ValueError("Missing Azure OpenAI configuration")
            
        api_key = azure_config.get("api_key", "")
        if is_dummy_api_key(api_key):
            print("Detected dummy API key for Azure OpenAI, using mock connector")
            connector = MockModelConnector(azure_config)
        else:
            connector = AzureOpenAIConnector(azure_config)
            
    elif provider == "gemini":
        gemini_config = model_config.get("gemini", {})
        if not gemini_config:
            raise ValueError("Missing Google Gemini configuration")
            
        api_key = gemini_config.get("api_key", "")
        if is_dummy_api_key(api_key):
            print("Detected dummy API key for Gemini, using mock connector")
            connector = MockModelConnector(gemini_config)
        else:
            connector = GeminiConnector(gemini_config)
    else:
        raise ValueError(f"Unsupported model provider: {provider}")
    
    # Enable caching if specified in config
    if config.get('general', {}).get('enable_model_caching', False):
        connector = ModelCachingWrapper(connector)
    
    return connector


def is_valid_api_config(config: Dict[str, Any]) -> bool:
    """
    Check if the API configuration is valid for the active model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    model_config = config.get("model", {})
    provider = model_config.get("provider", "azure_openai")
    
    if provider == "azure_openai":
        azure_config = model_config.get("azure_openai", {})
        # For mock mode with dummy keys, we don't need all fields to be valid
        api_key = azure_config.get("api_key", "")
        if is_dummy_api_key(api_key):
            return True
            
        return (azure_config.get("api_key") and 
                azure_config.get("api_base") and 
                azure_config.get("api_version") and
                azure_config.get("embedding_deployment") and
                azure_config.get("completion_deployment"))
    
    elif provider == "gemini":
        gemini_config = model_config.get("gemini", {})
        # For mock mode with dummy keys, we don't need all fields to be valid
        api_key = gemini_config.get("api_key", "")
        if is_dummy_api_key(api_key):
            return True
            
        return (bool(gemini_config.get("api_key")) and
                bool(gemini_config.get("embedding_model")) and
                bool(gemini_config.get("completion_model")))
    
    return False 


class ModelCachingWrapper:
    """
    Wrapper class that adds caching to any model connector.
    """
    
    def __init__(self, model_connector):
        """
        Initialize with a base model connector.
        
        Args:
            model_connector: Base model connector to wrap
        """
        self.connector = model_connector
        self.embedding_cache = {}
        self.generation_cache = {}
        self.evaluation_cache = {}
    
    def generate_embeddings(self, texts):
        """
        Generate embeddings with caching.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        # Create a hashable key for the texts
        if isinstance(texts, list):
            # For a list of texts, create a tuple of the texts for hashing
            cache_key = tuple(texts)
        else:
            # For a single text input
            cache_key = texts
            
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        embeddings = self.connector.generate_embeddings(texts)
        self.embedding_cache[cache_key] = embeddings
        return embeddings
    
    def generate_text(self, query, context):
        """
        Generate text with caching.
        
        Args:
            query: Query text
            context: Context text
            
        Returns:
            Generated text
        """
        cache_key = f"{query}::{context}"
        if cache_key in self.generation_cache:
            return self.generation_cache[cache_key]
            
        response = self.connector.generate_text(query, context)
        self.generation_cache[cache_key] = response
        return response
        
    def get_embedding(self, text):
        """
        Get a single embedding with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if hasattr(self.connector, 'get_embedding'):
            cache_key = text
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
                
            embedding = self.connector.get_embedding(text)
            self.embedding_cache[cache_key] = embedding
            return embedding
        else:
            # Fall back to generate_embeddings if get_embedding is not available
            embeddings = self.generate_embeddings([text])
            return embeddings[0] if embeddings else None 

    def evaluate_batch(self, queries, contexts=None, max_workers=4):
        """
        Evaluate multiple queries in parallel with caching.
        
        Args:
            queries: List of query texts to evaluate
            contexts: Optional list of contexts for each query
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of evaluation results
        """
        import concurrent.futures
        
        # Prepare the evaluation tasks
        def process_query(idx):
            query = queries[idx]
            context = contexts[idx] if contexts else None
            
            # Check cache first
            cache_key = f"{query}::{context}" if context else query
            if cache_key in self.evaluation_cache:
                return self.evaluation_cache[cache_key]
            
            # Generate the result
            if context:
                result = self.connector.generate_text(query, context)
            else:
                result = self.connector.generate_text(query, "")
                
            # Cache the result
            self.evaluation_cache[cache_key] = result
            return result
        
        # Run evaluations in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(process_query, i): i for i in range(len(queries))}
            for future in concurrent.futures.as_completed(future_to_idx):
                results.append(future.result())
        
        return results 