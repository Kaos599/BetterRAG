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
        return AzureOpenAIConnector(azure_config)
    
    elif provider == "gemini":
        gemini_config = model_config.get("gemini", {})
        if not gemini_config:
            raise ValueError("Missing Google Gemini configuration")
        return GeminiConnector(gemini_config)
        
    else:
        raise ValueError(f"Unsupported model provider: {provider}")
        
    # Never reached
    raise ValueError("Failed to create model connector")


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