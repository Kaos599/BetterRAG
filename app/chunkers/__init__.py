from typing import Dict, Any, List
from app.chunkers.base_chunker import BaseChunker
from app.chunkers.fixed_size import FixedSizeChunker
from app.chunkers.recursive import RecursiveChunker
from app.chunkers.semantic import SemanticChunker


def get_chunker(strategy_name: str, config: Dict[str, Any]) -> BaseChunker:
    """
    Factory function to get the appropriate chunker based on strategy name.
    
    Args:
        strategy_name: Name of the chunking strategy
        config: Configuration parameters for the chunker
        
    Returns:
        An instance of the requested chunker
        
    Raises:
        ValueError: If the requested strategy is not supported
    """
    chunkers = {
        "fixed_size": FixedSizeChunker,
        "fixed_size_no_overlap": lambda cfg: FixedSizeChunker({**cfg, "chunk_overlap": 0}),
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker
    }
    
    if strategy_name not in chunkers:
        raise ValueError(f"Unsupported chunking strategy: {strategy_name}")
    
    chunker_class = chunkers[strategy_name]
    
    if strategy_name == "fixed_size_no_overlap":
        return chunker_class(config)
    else:
        return chunker_class(config)


def get_all_enabled_chunkers(config: Dict[str, Any]) -> List[BaseChunker]:
    """
    Get all enabled chunkers based on the configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of chunker instances
    """
    chunking_config = config.get("chunking_strategies", {})
    enabled_chunkers = []
    
    for strategy_name, strategy_config in chunking_config.items():
        if isinstance(strategy_config, dict) and strategy_config.get("enabled", False):
            try:
                chunker = get_chunker(strategy_name, strategy_config)
                enabled_chunkers.append(chunker)
            except ValueError as e:
                print(f"Warning: {e}")
    
    if not enabled_chunkers:
        print("Warning: No chunking strategies are enabled in the configuration.")
        print("Using default fixed-size chunker.")
        default_config = {"chunk_size": 512, "chunk_overlap": 128}
        enabled_chunkers.append(FixedSizeChunker(default_config))
    
    return enabled_chunkers 