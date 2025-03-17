import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    """Class to load and parse configuration from YAML file."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ConfigLoader with path to config file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dict containing configuration parameters
        
        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the config file is invalid YAML
        """
        try:
            with open(self.config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
                return config
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {self.config_path}")
            raise
        except yaml.YAMLError as e:
            print(f"Error parsing the configuration file: {e}")
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the entire configuration.
        
        Returns:
            Dict containing all configuration parameters
        """
        return self.config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific section of the configuration.
        
        Args:
            section: Section name to retrieve
            
        Returns:
            Dict containing the requested section parameters
            
        Raises:
            KeyError: If the requested section doesn't exist
        """
        if section in self.config:
            return self.config[section]
        else:
            print(f"Error: Section '{section}' not found in configuration")
            raise KeyError(f"Section '{section}' not found in configuration")
    
    def validate_required_paths(self):
        """
        Validate that all required paths in the configuration exist,
        and create directories if they don't.
        """
        # Validate source documents path
        source_path = self.config.get('document', {}).get('source_path', '')
        if source_path and not os.path.exists(source_path):
            os.makedirs(source_path)
            print(f"Created directory: {source_path}")
        
        # Validate output directory
        output_dir = self.config.get('visualization', {}).get('output_directory', '')
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
    
    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """
        Check if a chunking strategy is enabled in the configuration.
        
        Args:
            strategy_name: Name of the strategy to check
            
        Returns:
            bool: True if strategy is enabled, False otherwise
        """
        try:
            chunking_config = self.get_section('chunking_strategies')
            strategy_config = chunking_config.get(strategy_name, {})
            return strategy_config.get('enabled', False)
        except KeyError:
            return False
    
    def get_active_model(self) -> str:
        """
        Get the currently active model from configuration.
        
        Returns:
            str: Name of the active model ("azure_openai" or "gemini")
        """
        try:
            model_config = self.get_section('model')
            return model_config.get('provider', 'azure_openai')
        except KeyError:
            return 'azure_openai'  # Default to Azure OpenAI if not specified 