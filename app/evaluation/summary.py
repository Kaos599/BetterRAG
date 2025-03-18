from typing import Dict, List, Any, Optional
import numpy as np
from app.models import get_model_connector


class EvaluationSummarizer:
    """
    Generates insightful summaries and comparisons between different retrieval methods.
    """
    
    def __init__(self, config):
        """
        Initialize the summarizer with configuration.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.model = get_model_connector(config)
    
    def generate_comparative_summary(self, evaluation_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a summary that compares different methods based on evaluation metrics.
        
        Args:
            evaluation_results: Dictionary mapping method names to their metrics
                Format: {
                    "method_name": {
                        "precision": float,
                        "recall": float,
                        "f1_score": float,
                        "latency": float,
                        ...
                    }
                }
        
        Returns:
            A natural language summary of the comparison with insights
        """
        # Extract key metrics for comparison
        methods = list(evaluation_results.keys())
        if not methods:
            return "No evaluation results to summarize."
        
        # Find the best method for each metric
        metric_comparison = {}
        for metric in evaluation_results[methods[0]].keys():
            # For metrics where higher is better (like precision, recall)
            is_higher_better = metric not in ["latency", "time", "error_rate"]
            
            if is_higher_better:
                best_method = max(methods, key=lambda m: evaluation_results[m].get(metric, 0))
                best_value = evaluation_results[best_method].get(metric, 0)
            else:
                best_method = min(methods, key=lambda m: evaluation_results[m].get(metric, float('inf')))
                best_value = evaluation_results[best_method].get(metric, 0)
                
            metric_comparison[metric] = {
                "best_method": best_method,
                "best_value": best_value,
                "all_values": {m: evaluation_results[m].get(metric, 0) for m in methods}
            }
        
        # Generate prompt for summary
        prompt = self._create_comparison_prompt(methods, metric_comparison)
        
        # Get summary from LLM
        summary = self.model.generate_text(
            query="Generate an insightful comparison of RAG methods",
            context=prompt
        )
        
        return summary
    
    def _create_comparison_prompt(self, methods: List[str], metric_comparison: Dict[str, Any]) -> str:
        """
        Create a detailed prompt for the LLM to generate insights.
        """
        prompt = f"Compare the following {len(methods)} retrieval methods: {', '.join(methods)}.\n\n"
        prompt += "Performance metrics:\n"
        
        for metric, data in metric_comparison.items():
            prompt += f"\n{metric.upper()}:\n"
            prompt += f"- Best method: {data['best_method']} ({data['best_value']:.4f})\n"
            
            # Sort values for comparison
            is_higher_better = metric not in ["latency", "time", "error_rate"]
            sorted_values = sorted(
                data['all_values'].items(),
                key=lambda x: x[1],
                reverse=is_higher_better
            )
            
            # Add comparisons
            for method, value in sorted_values:
                delta = abs(value - data['best_value'])
                if method != data['best_method']:
                    if is_higher_better:
                        prompt += f"- {method}: {value:.4f} ({delta:.4f} lower than best)\n"
                    else:
                        prompt += f"- {method}: {value:.4f} ({delta:.4f} higher than best)\n"
        
        prompt += "\nProvide a concise summary that includes:\n"
        prompt += "1. Which method performs best overall and why\n"
        prompt += "2. Trade-offs between different methods (e.g., one might be faster but less accurate)\n"
        prompt += "3. Specific scenarios where each method would be most appropriate\n"
        prompt += "4. Actionable recommendations for users\n"
        
        return prompt
    
    def get_metric_distribution(self, evaluation_results):
        """
        Calculate statistical distributions of metrics across methods.
        """
        distributions = {}
        
        for metric in evaluation_results[list(evaluation_results.keys())[0]].keys():
            values = [results.get(metric, 0) for results in evaluation_results.values()]
            distributions[metric] = {
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": min(values),
                "max": max(values)
            }
            
        return distributions 