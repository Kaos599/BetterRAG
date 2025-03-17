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
                "strategy": strategy,
                "query": query,
                "retrieved_chunks": 0,
                "retrieval_time": retrieval_time,
                "error": "No chunks retrieved"
            }
        
        # Prepare context from retrieved chunks
        context = "\n\n".join([chunk.get("text", "") for chunk in similar_chunks])
        context_tokens = self._estimate_tokens(context)
        
        # Generate answer
        generation_start = time.time()
        answer = self.model.generate_text(query, context)
        generation_time = time.time() - generation_start
        
        # Calculate metrics
        result = {
            "strategy": strategy,
            "query": query,
            "retrieved_chunks": len(similar_chunks),
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": retrieval_time + generation_time,
            "context_tokens": context_tokens,
            "context_precision": self._calculate_context_precision(similar_chunks, query),
            "token_efficiency": self._calculate_token_efficiency(similar_chunks, query),
            "chunk_similarities": [chunk.get("similarity", 0) for chunk in similar_chunks],
            "answer": answer,
        }
        
        # Add chunk details for analysis
        result["chunks"] = [
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", "")[:100] + "...",  # Truncate for readability
                "similarity": chunk.get("similarity", 0),
                "chunk_index": chunk.get("chunk_index", 0),
                "strategy": chunk.get("strategy", "")
            }
            for chunk in similar_chunks
        ]
        
        # Calculate additional metrics if ground truth is available
        if self.ground_truth and query in self.ground_truth:
            ground_truth_answer = self.ground_truth[query]
            result["answer_relevance"] = self._calculate_answer_relevance(answer, ground_truth_answer)
        
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
    
    def evaluate_all_queries(self, queries: List[str], strategies: List[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Evaluate all queries against all specified strategies.
        
        Args:
            queries: List of queries to evaluate
            strategies: Optional list of strategies to evaluate against
                        (if None, uses all available strategies)
            
        Returns:
            Dict mapping queries to strategies to evaluation results
        """
        results = {}
        
        # Get all strategies from the database if none specified
        if not strategies:
            strategies = self.db.get_all_strategies()
        
        for query in queries:
            results[query] = {}
            for strategy in strategies:
                results[query][strategy] = self.evaluate_query(query, strategy)
        
        return results
    
    def calculate_aggregate_metrics(self, evaluation_results: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate aggregate metrics across all queries for each strategy.
        
        Args:
            evaluation_results: Results from evaluate_all_queries
            
        Returns:
            Dict mapping strategy names to aggregated metrics
        """
        aggregated = {}
        
        # Get all strategies
        strategies = set()
        for query_results in evaluation_results.values():
            strategies.update(query_results.keys())
        
        # Initialize aggregated metrics for each strategy
        for strategy in strategies:
            aggregated[strategy] = {
                "strategy": strategy,
                "queries_evaluated": 0,
                "avg_retrieval_time": 0,
                "avg_generation_time": 0,
                "avg_total_time": 0,
                "avg_context_tokens": 0,
                "avg_context_precision": 0,
                "avg_token_efficiency": 0,
                "avg_chunk_similarities": 0,
                "avg_answer_relevance": 0,
                "has_answer_relevance": False
            }
        
        # Aggregate metrics
        for query, query_results in evaluation_results.items():
            for strategy, result in query_results.items():
                if "error" in result:
                    continue
                
                agg = aggregated[strategy]
                agg["queries_evaluated"] += 1
                agg["avg_retrieval_time"] += result["retrieval_time"]
                agg["avg_generation_time"] += result["generation_time"]
                agg["avg_total_time"] += result["total_time"]
                agg["avg_context_tokens"] += result["context_tokens"]
                agg["avg_context_precision"] += result["context_precision"]
                agg["avg_token_efficiency"] += result["token_efficiency"]
                agg["avg_chunk_similarities"] += np.mean(result["chunk_similarities"]) if result["chunk_similarities"] else 0
                
                if "answer_relevance" in result:
                    agg["avg_answer_relevance"] += result["answer_relevance"]
                    agg["has_answer_relevance"] = True
        
        # Calculate averages
        for strategy, agg in aggregated.items():
            queries_count = agg["queries_evaluated"]
            if queries_count > 0:
                agg["avg_retrieval_time"] /= queries_count
                agg["avg_generation_time"] /= queries_count
                agg["avg_total_time"] /= queries_count
                agg["avg_context_tokens"] /= queries_count
                agg["avg_context_precision"] /= queries_count
                agg["avg_token_efficiency"] /= queries_count
                agg["avg_chunk_similarities"] /= queries_count
                
                if agg["has_answer_relevance"]:
                    agg["avg_answer_relevance"] /= queries_count
        
        return aggregated
    
    def get_best_strategy(self, aggregated_metrics: Dict[str, Dict[str, Any]]) -> str:
        """
        Determine the best strategy based on aggregated metrics.
        
        Args:
            aggregated_metrics: Aggregated metrics from calculate_aggregate_metrics
            
        Returns:
            Name of the best strategy
        """
        if not aggregated_metrics:
            return None
        
        # Define weights for different metrics (customize as needed)
        weights = {
            "avg_context_precision": 0.4,
            "avg_token_efficiency": 0.3,
            "avg_chunk_similarities": 0.2,
            "avg_total_time": -0.1  # Negative weight because lower is better
        }
        
        # Calculate weighted scores
        scores = {}
        for strategy, metrics in aggregated_metrics.items():
            score = 0
            for metric, weight in weights.items():
                if metric in metrics:
                    # Normalize time metric (lower is better)
                    if metric == "avg_total_time":
                        max_time = max(m.get("avg_total_time", 0) for m in aggregated_metrics.values())
                        if max_time > 0:
                            normalized_value = 1 - (metrics[metric] / max_time)
                            score += normalized_value * abs(weight)
                    else:
                        score += metrics[metric] * weight
            
            scores[strategy] = score
        
        # Return the strategy with the highest score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return None
    
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