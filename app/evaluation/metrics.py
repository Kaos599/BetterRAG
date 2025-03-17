from typing import Dict, List, Any, Optional
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ChunkingEvaluator:
    """
    Class for evaluating chunking strategies using multiple metrics.
    """
    
    def __init__(self, model_connector, db_connector, config: Dict[str, Any]):
        """
        Initialize the evaluator with connectors and configuration.
        
        Args:
            model_connector: Model connector for embeddings and text generation
            db_connector: Database connector for retrieving chunks
            config: Evaluation configuration
        """
        self.model = model_connector
        self.db = db_connector
        self.config = config
        self.top_k = config.get("top_k", 5)
        
        # Store ground truth if available
        self.ground_truth = None
        ground_truth_path = config.get("ground_truth_path")
        if ground_truth_path:
            try:
                import json
                with open(ground_truth_path, 'r') as f:
                    self.ground_truth = json.load(f)
            except Exception as e:
                print(f"Error loading ground truth: {e}")
                self.ground_truth = None
    
    def evaluate_query(self, query: str, strategy: str) -> Dict[str, Any]:
        """
        Evaluate a query against a specific chunking strategy.
        
        Args:
            query: The query to evaluate
            strategy: The chunking strategy to evaluate against
            
        Returns:
            Dict with evaluation metrics
        """
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.model.generate_embeddings([query])
        if not query_embedding:
            return {"error": "Failed to generate query embedding"}
        
        query_embedding = query_embedding[0]
        
        # Find relevant chunks
        similar_chunks = self.db.find_similar_chunks(
            query_embedding=query_embedding,
            strategy=strategy,
            top_k=self.top_k
        )
        
        retrieval_time = time.time() - start_time
        
        if not similar_chunks:
            return {
                "error": "No chunks retrieved",
                "latency": retrieval_time,
                "context_precision": 0,
                "token_efficiency": 0,
                "answer_relevance": 0,
                "combined_score": 0
            }
        
        # Prepare context from retrieved chunks
        context = "\n\n".join([chunk.get("text", "") for chunk in similar_chunks])
        context_tokens = self._estimate_tokens(context)
        
        # Generate answer
        generation_start = time.time()
        answer = self.model.generate_text(query, context)
        generation_time = time.time() - generation_start
        
        total_time = retrieval_time + generation_time
        
        # Calculate metrics
        context_precision = self._calculate_context_precision(similar_chunks, query)
        token_efficiency = self._calculate_token_efficiency(similar_chunks, query)
        
        # Calculate answer relevance if ground truth is available
        answer_relevance = 0
        if self.ground_truth and query in self.ground_truth:
            ground_truth_answer = self.ground_truth[query]
            answer_relevance = self._calculate_answer_relevance(answer, ground_truth_answer)
        
        # Calculate combined score
        combined_score = (0.4 * context_precision) + (0.3 * token_efficiency) + (0.3 * answer_relevance)
        
        result = {
            "context_precision": context_precision,
            "token_efficiency": token_efficiency,
            "answer_relevance": answer_relevance,
            "latency": total_time,
            "combined_score": combined_score,
            "answer": answer,
            "chunks_retrieved": len(similar_chunks),
            "context_tokens": context_tokens
        }
        
        return result
    
    def evaluate_all_strategies(self, query: str) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate a query against all available chunking strategies.
        
        Args:
            query: The query to evaluate
            
        Returns:
            Dict mapping strategy names to evaluation results
        """
        results = {}
        
        # Get all strategies from the database
        strategies = self.db.get_all_strategies()
        
        for strategy in strategies:
            results[strategy] = self.evaluate_query(query, strategy)
        
        return results
    
    def evaluate_strategies(self) -> tuple:
        """
        Evaluate all chunking strategies using test queries from configuration.
        
        Returns:
            Tuple containing:
            - Evaluation results dictionary
            - Aggregated metrics dictionary
            - Name of the best performing strategy
        """
        # Load test queries from the test queries file
        import json
        import os
        
        test_queries_file = self.config['general']['test_queries_file']
        if not os.path.exists(test_queries_file):
            raise FileNotFoundError(f"Test queries file not found: {test_queries_file}")
        
        with open(test_queries_file, 'r') as f:
            queries = json.load(f)
        
        # Evaluate all queries against all strategies
        strategies = self.db.get_all_strategies()
        evaluation_results = self.evaluate_all_queries(queries, strategies)
        
        # Calculate aggregate metrics
        aggregated_metrics = self.calculate_aggregate_metrics(evaluation_results)
        
        # Determine the best strategy
        best_strategy = self.get_best_strategy(aggregated_metrics)
        
        return evaluation_results, aggregated_metrics, best_strategy
    
    def evaluate_all_queries(self, queries: List[Dict[str, Any]], strategies: List[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Evaluate all queries against all specified strategies.
        
        Args:
            queries: List of query dictionaries to evaluate
            strategies: Optional list of strategies to evaluate against
                        (if None, uses all available strategies)
            
        Returns:
            Dict mapping query IDs to strategies to evaluation results
        """
        results = {}
        
        # Get all strategies from the database if none specified
        if not strategies:
            strategies = self.db.get_all_strategies()
        
        # Initialize results structure
        for i, query_dict in enumerate(queries):
            query_id = f"query_{i}"  # Create a unique ID for each query
            query_text = query_dict["query"]
            
            results[query_id] = {
                "query_text": query_text,
                "expected_keywords": query_dict.get("expected_keywords", []),
                "results": {}
            }
        
        # Use parallel processing if enabled in config
        use_parallel = self.config.get('general', {}).get('parallel_evaluation', False)
        
        if use_parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import multiprocessing
            
            # Determine number of workers - default to CPU count
            max_workers = self.config.get('general', {}).get('max_workers', multiprocessing.cpu_count())
            
            # Process each strategy-query combination in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for strategy in strategies:
                    for i, query_dict in enumerate(queries):
                        query_id = f"query_{i}"
                        query_text = query_dict["query"]
                        future = executor.submit(self.evaluate_query, query_text, strategy)
                        futures[(strategy, query_id)] = future
                
                # Collect results as they complete
                for (strategy, query_id), future in as_completed(futures):
                    results[query_id]["results"][strategy] = future.result()
        else:
            # Sequential processing
            for i, query_dict in enumerate(queries):
                query_id = f"query_{i}"
                query_text = query_dict["query"]
                
                for strategy in strategies:
                    results[query_id]["results"][strategy] = self.evaluate_query(query_text, strategy)
        
        return results
    
    def calculate_aggregate_metrics(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate aggregate metrics across all queries for each strategy.
        
        Args:
            evaluation_results: Results from evaluate_all_queries
            
        Returns:
            Dict mapping strategies to aggregated metrics
        """
        aggregated = {}
        
        # Get all strategies from the results
        strategies = set()
        for query_id, query_data in evaluation_results.items():
            for strategy in query_data["results"].keys():
                strategies.add(strategy)
        
        # Initialize aggregated metrics for each strategy
        for strategy in strategies:
            aggregated[strategy] = {
                "context_precision": [],
                "token_efficiency": [],
                "answer_relevance": [],
                "latency": [],
                "combined_score": []
            }
        
        # Collect metrics for each strategy across all queries
        for query_id, query_data in evaluation_results.items():
            for strategy, metrics in query_data["results"].items():
                for metric_name in ["context_precision", "token_efficiency", "answer_relevance", "latency", "combined_score"]:
                    if metric_name in metrics:
                        aggregated[strategy][metric_name].append(metrics[metric_name])
        
        # Calculate averages
        for strategy in strategies:
            for metric_name in ["context_precision", "token_efficiency", "answer_relevance", "latency", "combined_score"]:
                values = aggregated[strategy][metric_name]
                if values:
                    aggregated[strategy][metric_name] = sum(values) / len(values)
                else:
                    aggregated[strategy][metric_name] = 0.0
        
        return aggregated
    
    def get_best_strategy(self, aggregated_metrics: Dict[str, Dict[str, Any]]) -> str:
        """
        Determine the best strategy based on aggregated metrics.
        
        Args:
            aggregated_metrics: Aggregated metrics from calculate_aggregate_metrics
            
        Returns:
            Name of the best performing strategy
        """
        if not aggregated_metrics:
            return None
        
        # Define weights for different metrics
        weights = {
            "context_precision": 0.4,
            "token_efficiency": 0.3,
            "answer_relevance": 0.3,
            # Latency is considered but with lower weight
            "latency": -0.1  # Negative because lower latency is better
        }
        
        # Calculate weighted scores
        scores = {}
        for strategy, metrics in aggregated_metrics.items():
            score = 0
            for metric, weight in weights.items():
                if metric in metrics:
                    # For latency, lower is better, so we use 1/latency
                    if metric == "latency" and metrics[metric] > 0:
                        score += weight * (1 / metrics[metric])
                    else:
                        score += weight * metrics[metric]
            
            scores[strategy] = score
        
        # Find the strategy with the highest score
        if not scores:
            return None
            
        best_strategy = max(scores.items(), key=lambda x: x[1])[0]
        return best_strategy
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple estimation: average of 4 characters per token
        return len(text) // 4
    
    def _calculate_context_precision(self, chunks: List[Dict[str, Any]], query: str) -> float:
        """
        Calculate how precise the retrieved chunks are for the query.
        
        Args:
            chunks: Retrieved chunks
            query: The query
            
        Returns:
            Precision score (0-1)
        """
        if not chunks:
            return 0
        
        # Use the similarity scores as a proxy for precision
        similarities = [chunk.get("similarity", 0) for chunk in chunks]
        return np.mean(similarities) if similarities else 0
    
    def _calculate_token_efficiency(self, chunks: List[Dict[str, Any]], query: str) -> float:
        """
        Calculate token efficiency: how many tokens are relevant vs. total.
        
        Args:
            chunks: Retrieved chunks
            query: The query
            
        Returns:
            Efficiency score (0-1)
        """
        if not chunks:
            return 0
        
        # Simple formula: average similarity * average chunk length ratio
        similarities = [chunk.get("similarity", 0) for chunk in chunks]
        avg_similarity = np.mean(similarities) if similarities else 0
        
        # Get the average effective length ratio (prioritize shorter chunks)
        chunk_texts = [chunk.get("text", "") for chunk in chunks]
        if not chunk_texts:
            return 0
        
        avg_chunk_length = np.mean([len(text) for text in chunk_texts])
        max_chunk_length = max([len(text) for text in chunk_texts])
        
        length_ratio = 1 - (avg_chunk_length / max_chunk_length) if max_chunk_length > 0 else 0
        
        return avg_similarity * (0.5 + 0.5 * length_ratio)
    
    def _calculate_answer_relevance(self, answer: str, ground_truth: str) -> float:
        """
        Calculate relevance of the generated answer to the ground truth.
        
        Args:
            answer: Generated answer
            ground_truth: Ground truth answer
            
        Returns:
            Relevance score (0-1)
        """
        if not answer or not ground_truth:
            return 0
        
        # Generate embeddings for answer and ground truth
        embeddings = self.model.generate_embeddings([answer, ground_truth])
        if not embeddings or len(embeddings) < 2:
            return 0
        
        # Calculate cosine similarity
        answer_embedding = np.array(embeddings[0]).reshape(1, -1)
        truth_embedding = np.array(embeddings[1]).reshape(1, -1)
        
        similarity = cosine_similarity(answer_embedding, truth_embedding)[0][0]
        
        return max(0, min(1, similarity))  # Ensure it's in range [0, 1] 