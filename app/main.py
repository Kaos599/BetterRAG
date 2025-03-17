"""
Main entry point for the BetterRAG application.
"""

import argparse
import os
import sys
import logging
from typing import Dict, List, Any, Tuple

from app.utils.config import get_config
from app.utils.helpers import find_documents, load_document_from_file, create_directory_if_not_exists
from app.chunkers import get_chunker
from app.models import get_model_connector
from app.db import get_db_connector
from app.evaluation import get_evaluator
from app.visualization import get_dashboard

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="BetterRAG: Evaluate text chunking strategies")
    
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    
    parser.add_argument("--dashboard-only", action="store_true",
                       help="Run only the visualization dashboard")
    
    parser.add_argument("--reset-db", action="store_true",
                       help="Reset the database before processing")
    
    return parser.parse_args()

def process_documents(config: Dict[str, Any], model_connector: Any, db_connector: Any, chunkers: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Process documents using each chunking strategy.
    
    Args:
        config: Application configuration.
        model_connector: Model connector instance.
        db_connector: Database connector instance.
        chunkers: Dictionary of chunker instances.
        
    Returns:
        Dictionary mapping strategy names to lists of chunk IDs.
    """
    logger.info("Processing documents...")
    
    # Find documents
    source_path = config['general']['data_source']
    file_types = config.get('document', {}).get('file_types', [".txt"])
    file_paths = find_documents(source_path, file_types)
    
    if not file_paths:
        logger.error(f"No documents found in {source_path}")
        sys.exit(1)
    
    # Process documents with each chunking strategy
    chunk_ids_by_strategy = {}
    
    for strategy_name, chunker in chunkers.items():
        logger.info(f"Processing documents with {strategy_name} strategy...")
        chunk_ids = []
        
        for file_path in file_paths:
            doc_id, content = load_document_from_file(file_path)
            chunks = chunker.chunk_text(content)
            
            # Generate embeddings and store in database
            for chunk in chunks:
                embedding = model_connector.get_embedding(chunk)
                chunk_id = db_connector.insert_chunk(
                    chunk, embedding, doc_id, strategy_name
                )
                chunk_ids.append(chunk_id)
            
            logger.info(f"Processed {len(chunks)} chunks for document {doc_id}")
        
        chunk_ids_by_strategy[strategy_name] = chunk_ids
    
    return chunk_ids_by_strategy

def evaluate_strategies(config: Dict[str, Any], model_connector: Any, db_connector: Any) -> Tuple[Dict, Dict, str]:
    """
    Evaluate chunking strategies and return results.
    
    Args:
        config: Application configuration.
        model_connector: Model connector instance.
        db_connector: Database connector instance.
    
    Returns:
        Tuple containing:
        - Evaluation results dictionary
        - Aggregated metrics dictionary
        - Name of the best performing strategy
    """
    logger.info("Evaluating chunking strategies...")
    
    # Create evaluator
    evaluator = get_evaluator(model_connector, db_connector, config)
    
    # Load test queries
    test_queries_file = config['general']['test_queries_file']
    if not os.path.exists(test_queries_file):
        logger.error(f"Test queries file not found: {test_queries_file}")
        sys.exit(1)
    
    # Run evaluation
    evaluation_results, aggregated_metrics, best_strategy = evaluator.evaluate_strategies()
    
    logger.info(f"Evaluation complete. Best strategy: {best_strategy}")
    
    return evaluation_results, aggregated_metrics, best_strategy

def run_visualization(config: Dict[str, Any], evaluation_results: Dict, 
                     aggregated_metrics: Dict, best_strategy: str) -> None:
    """
    Generate visualizations and run the dashboard.
    
    Args:
        config: Application configuration.
        evaluation_results: Evaluation results dictionary.
        aggregated_metrics: Aggregated metrics dictionary.
        best_strategy: Name of the best performing strategy.
    """
    logger.info("Generating visualizations...")
    
    # Create output directory
    output_dir = config['general']['output_directory']
    create_directory_if_not_exists(output_dir)
    
    # Get dashboard
    dashboard = get_dashboard(config)
    
    # Save results
    dashboard.save_results(evaluation_results, aggregated_metrics, best_strategy)
    
    # Generate charts
    dashboard.generate_charts(evaluation_results, aggregated_metrics, best_strategy)
    
    # Run dashboard
    logger.info(f"Starting dashboard on port {config['visualization']['dashboard']['port']}...")
    dashboard.run_dashboard(evaluation_results, aggregated_metrics, best_strategy)

def main():
    """Main entry point for the application."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Set custom config path if provided
        if args.config != "config.yaml":
            os.environ["BETTERRAG_CONFIG_PATH"] = args.config
        
        # Load configuration
        config = get_config()
        
        # Check if we're running in dashboard-only mode
        if args.dashboard_only:
            # TODO: Load previous results and run dashboard
            logger.info("Running in dashboard-only mode...")
            # This would need to load previously saved results
            return
        
        # Get model connector
        model_connector = get_model_connector(config)
        
        # Get database connector
        db_connector = get_db_connector(config)
        
        # Reset database if requested
        if args.reset_db or config['general'].get('db_reset', False):
            logger.info("Resetting database...")
            db_connector.reset_database()
        
        # Get chunkers
        chunkers = {}
        for strategy_name, strategy_config in config['chunking_strategies'].items():
            if strategy_config.get('enabled', False):
                chunkers[strategy_name] = get_chunker(strategy_name, strategy_config)
        
        # Process documents
        process_documents(config, model_connector, db_connector, chunkers)
        
        # Evaluate strategies
        evaluation_results, aggregated_metrics, best_strategy = evaluate_strategies(
            config, model_connector, db_connector
        )
        
        # Run visualization
        run_visualization(config, evaluation_results, aggregated_metrics, best_strategy)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 