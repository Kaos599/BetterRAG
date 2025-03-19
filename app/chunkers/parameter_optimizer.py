from typing import Dict, List, Any, Tuple
import itertools
import copy
import logging
import random
from app.chunkers import get_chunker
from app.chunkers.fixed_size import FixedSizeChunker
from app.chunkers.recursive import RecursiveChunker
from app.chunkers.semantic import SemanticChunker

logger = logging.getLogger(__name__)

class ChunkingParameterOptimizer:
    """
    Class for generating and evaluating multiple chunking configurations
    with different parameters to find optimal settings.
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        """
        Initialize the parameter optimizer with a base configuration.
        
        Args:
            base_config: Base configuration containing chunking strategies
        """
        self.base_config = base_config
        self.param_config = base_config.get("parameter_optimization", {})
        # Get max configurations limit if specified
        self.max_configs_per_strategy = self.param_config.get("max_configs_per_strategy", None)
        logger.info(f"Initializing parameter optimizer" + 
                   (f" (max {self.max_configs_per_strategy} configs per strategy)" 
                    if self.max_configs_per_strategy else ""))
        self.parameter_sets = self._generate_parameter_sets()
        
    def _generate_parameter_sets(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate parameter sets for different chunking strategies.
        
        Returns:
            Dictionary mapping strategy types to lists of parameter configurations
        """
        parameter_sets = {
            "fixed_size": [],
            "recursive": [],
            "semantic": []
        }
        
        # Get parameter ranges from config (or use defaults)
        chunk_sizes = self.param_config.get("chunk_sizes", [256, 512, 768, 1024])
        overlap_percentages = self.param_config.get("overlap_percentages", [0.12, 0.15, 0.18, 0.20, 0.25, 0.30])
        similarity_thresholds = self.param_config.get("similarity_thresholds", [0.65, 0.75, 0.85, 0.95])
        semantic_methods = self.param_config.get("semantic_methods", ["standard", "interquartile", "percentile"])
        
        # Generate parameters for fixed size chunkers
        fixed_size_params = []
        for size in chunk_sizes:
            for overlap_pct in overlap_percentages:
                overlap = int(size * overlap_pct)
                fixed_size_params.append({
                    "enabled": True,
                    "chunk_size": size,
                    "chunk_overlap": overlap,
                    "config_name": f"fixed_size_{size}_{int(overlap_pct*100)}pct"
                })
        
        # Sample if max_configs is set
        if self.max_configs_per_strategy and len(fixed_size_params) > self.max_configs_per_strategy:
            parameter_sets["fixed_size"] = random.sample(fixed_size_params, self.max_configs_per_strategy)
        else:
            parameter_sets["fixed_size"] = fixed_size_params
        
        # Recursive chunker parameters
        # Same chunk sizes and overlaps as fixed size
        # But with different separator combinations
        separator_sets = [
            ["\n\n", "\n", ". ", " ", ""],  # Standard
            ["\n\n\n", "\n\n", "\n", ". ", " ", ""],  # More paragraph emphasis
            ["\n", ". ", "; ", ", ", " ", ""],  # More sentence emphasis
            ["## ", "# ", "\n\n", "\n", ". ", " ", ""]  # Markdown-aware
        ]
        
        recursive_params = []
        for size in chunk_sizes:
            for overlap_pct in overlap_percentages:
                for i, separators in enumerate(separator_sets):
                    overlap = int(size * overlap_pct)
                    recursive_params.append({
                        "enabled": True,
                        "chunk_size": size,
                        "chunk_overlap": overlap,
                        "separators": separators,
                        "config_name": f"recursive_{size}_{int(overlap_pct*100)}pct_sep{i+1}"
                    })
        
        # Sample if max_configs is set
        if self.max_configs_per_strategy and len(recursive_params) > self.max_configs_per_strategy:
            parameter_sets["recursive"] = random.sample(recursive_params, self.max_configs_per_strategy)
        else:
            parameter_sets["recursive"] = recursive_params
        
        # Semantic chunker parameters
        # Get min/max chunk size combinations from config or use defaults
        min_chunk_sizes = self.param_config.get("min_chunk_sizes", [50, 100, 150])
        max_chunk_sizes = self.param_config.get("max_chunk_sizes", [600, 800, 1000, 1200])
        
        semantic_params = []
        for thresh in similarity_thresholds:
            for min_size in min_chunk_sizes:
                for max_size in max_chunk_sizes:
                    for method in semantic_methods:
                        # Skip invalid combinations
                        if min_size >= max_size:
                            continue
                            
                        semantic_params.append({
                            "enabled": True,
                            "similarity_threshold": thresh,
                            "min_chunk_size": min_size,
                            "max_chunk_size": max_size,
                            "semantic_method": method,
                            "config_name": f"semantic_{method}_{thresh}_{min_size}_{max_size}"
                        })
        
        # Sample if max_configs is set
        if self.max_configs_per_strategy and len(semantic_params) > self.max_configs_per_strategy:
            parameter_sets["semantic"] = random.sample(semantic_params, self.max_configs_per_strategy)
        else:
            parameter_sets["semantic"] = semantic_params
        
        # Log parameter space summary
        total_configs = sum(len(configs) for configs in parameter_sets.values())
        logger.info(f"Parameter optimization summary:")
        logger.info(f"- Fixed-size chunkers: {len(parameter_sets['fixed_size'])}")
        logger.info(f"- Recursive chunkers: {len(parameter_sets['recursive'])}")
        logger.info(f"- Semantic chunkers: {len(parameter_sets['semantic'])}")
        logger.info(f"- Total configurations: {total_configs}")
        
        return parameter_sets
    
    def generate_chunker_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate complete chunker configurations for all parameter sets.
        
        Returns:
            Dictionary mapping configuration names to chunker configs
        """
        chunker_configs = {}
        
        # Process fixed size chunkers
        for params in self.parameter_sets["fixed_size"]:
            try:
                params_copy = copy.deepcopy(params)
                if "config_name" in params_copy:
                    config_name = params_copy.pop("config_name")
                else:
                    # Create a default name if missing
                    size = params_copy.get("chunk_size", "unknown")
                    overlap = params_copy.get("chunk_overlap", "unknown")
                    config_name = f"fixed_size_{size}_{overlap}"
                chunker_configs[config_name] = params_copy
            except Exception as e:
                logger.error(f"Error processing fixed size parameter set: {e}")
                logger.error(f"Parameter set: {params}")
        
        # Process recursive chunkers
        for params in self.parameter_sets["recursive"]:
            try:
                params_copy = copy.deepcopy(params)
                if "config_name" in params_copy:
                    config_name = params_copy.pop("config_name")
                else:
                    # Create a default name if missing
                    size = params_copy.get("chunk_size", "unknown")
                    overlap = params_copy.get("chunk_overlap", "unknown")
                    config_name = f"recursive_{size}_{overlap}"
                chunker_configs[config_name] = params_copy
            except Exception as e:
                logger.error(f"Error processing recursive parameter set: {e}")
                logger.error(f"Parameter set: {params}")
        
        # Process semantic chunkers
        for params in self.parameter_sets["semantic"]:
            try:
                params_copy = copy.deepcopy(params)
                if "config_name" in params_copy:
                    config_name = params_copy.pop("config_name")
                else:
                    # Create a default name if missing
                    thresh = params_copy.get("similarity_threshold", "unknown")
                    min_size = params_copy.get("min_chunk_size", "unknown")
                    max_size = params_copy.get("max_chunk_size", "unknown")
                    config_name = f"semantic_{thresh}_{min_size}_{max_size}"
                chunker_configs[config_name] = params_copy
            except Exception as e:
                logger.error(f"Error processing semantic parameter set: {e}")
                logger.error(f"Parameter set: {params}")
        
        return chunker_configs
    
    def get_all_chunkers(self) -> List[Tuple[str, Any]]:
        """
        Get all chunker instances with their parameter configurations.
        
        Returns:
            List of tuples with (config_name, chunker_instance)
        """
        chunker_configs = self.generate_chunker_configs()
        chunkers = []
        
        logger.info(f"Initializing {len(chunker_configs)} chunker instances...")
        
        # Create all chunker instances
        for config_name, config in chunker_configs.items():
            if "separators" in config:
                # This is a recursive chunker
                chunker = RecursiveChunker(config)
                chunkers.append((config_name, chunker))
            elif "similarity_threshold" in config:
                # This is a semantic chunker
                chunker = SemanticChunker(config)
                chunkers.append((config_name, chunker))
            else:
                # This is a fixed size chunker
                chunker = FixedSizeChunker(config)
                chunkers.append((config_name, chunker))
        
        return chunkers 