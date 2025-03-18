from app.evaluation.metrics import ChunkingEvaluator
from app.evaluation.summary import EvaluationSummarizer

def get_evaluator(model_connector, db_connector, config):
    """
    Get an evaluator instance.
    
    Args:
        model_connector: Model connector for embeddings and text generation
        db_connector: Database connector for retrieving chunks
        config: Evaluation configuration
        
    Returns:
        ChunkingEvaluator instance
    """
    return ChunkingEvaluator(model_connector, db_connector, config)

__all__ = ['EvaluationSummarizer'] 