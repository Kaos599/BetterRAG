from typing import Dict, Any, Union, Optional
from app.models.azure_openai import AzureOpenAIConnector
from app.models.gemini import GeminiConnector


def get_model_connector(config: Dict[str, Any], model_name: Optional[str] = None) -> Union[AzureOpenAIConnector, GeminiConnector]:
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
    
    if provider == "azure_openai":
        azure_config = model_config.get("azure_openai", {})
        if not azure_config:
            raise ValueError("Missing Azure OpenAI configuration")
        connector = AzureOpenAIConnector(azure_config)
    elif provider == "gemini":
        gemini_config = model_config.get("gemini", {})
        if not gemini_config:
            raise ValueError("Missing Google Gemini configuration")
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
        return (azure_config.get("api_key") and 
                azure_config.get("api_base") and 
                azure_config.get("api_version") and
                azure_config.get("embedding_deployment") and
                azure_config.get("completion_deployment"))
    
    elif provider == "gemini":
        gemini_config = model_config.get("gemini", {})
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