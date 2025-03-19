"""
Configuration utilities for BetterRAG.

This module provides functions for loading and validating the configuration file.
"""

import os
import yaml
from typing import Dict, Any, Optional
import re

def _resolve_env_vars(value: str) -> str:
    """
    Resolve environment variables in a string value.
    
    Args:
        value: The string value that may contain environment variable references.
        
    Returns:
        The string with environment variables resolved.
    """
    if not isinstance(value, str):
        return value
        
    # Find all ${ENV_VAR} patterns
    pattern = r'\${([^}]+)}'
    matches = re.findall(pattern, value)
    
    # Replace each environment variable with its value
    result = value
    for match in matches:
        env_value = os.environ.get(match)
        if env_value is not None:
            result = result.replace(f'${{{match}}}', env_value)
            
    return result

def _process_config_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process configuration values, resolving environment variables.
    
    Args:
        config: The configuration dictionary to process.
        
    Returns:
        The processed configuration dictionary.
    """
    processed_config = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            processed_config[key] = _process_config_values(value)
        elif isinstance(value, list):
            processed_config[key] = [
                _process_config_values(item) if isinstance(item, dict) else 
                _resolve_env_vars(item) if isinstance(item, str) else item
                for item in value
            ]
        elif isinstance(value, str):
            processed_config[key] = _resolve_env_vars(value)
        else:
            processed_config[key] = value
            
    return processed_config

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file. Defaults to "config.yaml".
        
    Returns:
        The loaded configuration as a dictionary.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration file can't be parsed.
    """
    # Check if config file exists
    if not os.path.exists(config_path):
        template_path = "config.template.yaml"
        
        # Check if template exists but config doesn't
        if os.path.exists(template_path):
            raise FileNotFoundError(
                f"Configuration file '{config_path}' not found. "
                f"Please copy '{template_path}' to '{config_path}' and update it with your settings."
            )
        else:
            raise FileNotFoundError(
                f"Configuration file '{config_path}' not found. "
                f"Please create a configuration file based on the documentation."
            )
    
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            
        # Process environment variables
        config = _process_config_values(config)
        
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file: {str(e)}")

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate the configuration.
    
    Args:
        config: The configuration dictionary to validate.
        
    Returns:
        True if the configuration is valid.
        
    Raises:
        ValueError: If the configuration is invalid.
    """
    # Check for required top-level keys
    required_keys = ['model', 'database', 'chunkers', 'evaluation']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")
    
    # Validate model configuration
    if 'provider' not in config['model']:
        raise ValueError("Missing 'provider' in model configuration")
        
    provider = config['model']['provider']
    if provider not in ['azure_openai', 'gemini']:
        raise ValueError(f"Invalid model provider: {provider}. Must be one of: azure_openai, gemini")
        
    # Validate provider-specific configuration
    if provider == 'azure_openai':
        if 'azure_openai' not in config['model']:
            raise ValueError("Missing 'azure_openai' section in model configuration")
            
        required_azure_keys = ['api_key', 'api_base', 'api_version', 'embedding_deployment', 'completion_deployment']
        for key in required_azure_keys:
            if key not in config['model']['azure_openai']:
                raise ValueError(f"Missing required Azure OpenAI configuration: {key}")
    
    elif provider == 'gemini':
        if 'gemini' not in config['model']:
            raise ValueError("Missing 'gemini' section in model configuration")
            
        required_gemini_keys = ['api_key', 'embedding_model', 'completion_model']
        for key in required_gemini_keys:
            if key not in config['model']['gemini']:
                raise ValueError(f"Missing required Gemini configuration: {key}")
    
    # Validate database configuration
    if 'type' not in config['database']:
        raise ValueError("Missing 'type' in database configuration")
        
    db_type = config['database']['type']
    if db_type not in ['mongodb', 'chroma']:
        raise ValueError(f"Invalid database type: {db_type}. Must be one of: mongodb, chroma")
    
    # Validate chunking strategies
    if not config['chunkers']:
        raise ValueError("No chunking strategies defined")
        
    # Check if at least one chunking strategy is enabled
    if not any(strategy.get('enabled', False) for name, strategy in config['chunkers'].items() 
              if isinstance(strategy, dict)):
        raise ValueError("No chunking strategies are enabled")
    
    return True

def get_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load and validate the configuration.
    
    Args:
        config_path: Path to the configuration file. Defaults to "config.yaml".
        
    Returns:
        The validated configuration dictionary.
    """
    config = load_config(config_path)
    validate_config(config)
    return config 