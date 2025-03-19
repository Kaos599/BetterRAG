"""
Main entry point for the BetterRAG application.
"""

import argparse
import os
import sys
import logging
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

from app.utils.config import get_config
from app.utils.helpers import find_documents, load_document_from_file, create_directory_if_not_exists
from app.chunkers import get_chunker, get_all_enabled_chunkers
from app.chunkers.parameter_optimizer import ChunkingParameterOptimizer
from app.models import get_model_connector
from app.db import get_db_connector
from app.evaluation import get_evaluator
from app.visualization import get_dashboard, get_parameter_dashboard
from app.visualization.enhanced_dashboard import ParameterOptimizationDashboard

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce verbosity of httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)
# Also suppress urllib3 logs which might be used by some libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
# Suppress OpenAI library logs
logging.getLogger("openai").setLevel(logging.WARNING)
# Suppress SentenceTransformer logs except for errors
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

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
    
    parser.add_argument("--parameter-optimization", action="store_true",
                       help="Run parameter optimization with varied chunking configurations")
    
    parser.add_argument("--optimization-dashboard", action="store_true",
                       help="Run the parameter optimization dashboard")
    
    parser.add_argument("--skip-parameter-optimization", action="store_true",
                       help="Skip parameter optimization even if enabled in config")
    
    return parser.parse_args()

def process_with_parameter_optimization(config: Dict[str, Any], model_connector, db_connector) -> Tuple[Dict, Dict, str]:
    """
    Process documents and evaluate with automated parameter optimization.
    
    Args:
        config: Configuration dictionary
        model_connector: Model connector instance
        db_connector: Database connector instance
        
    Returns:
        Tuple containing evaluation results, aggregated metrics, and best strategy
    """
    logger.info("Starting parameter optimization...")
    
    # Initialize parameter optimizer
    parameter_optimizer = ChunkingParameterOptimizer(config)
    
    # Get all chunker configurations
    chunker_configs = parameter_optimizer.generate_chunker_configs()
    logger.info(f"Generated {len(chunker_configs)} chunking configurations for evaluation")
    
    # Get all chunker instances
    chunkers = parameter_optimizer.get_all_chunkers()
    
    # Process documents with all chunkers
    documents_path = config["general"]["data_source"]
    logger.info(f"Processing documents from {documents_path}")
    
    # Find all documents
    document_files = find_documents(documents_path)
    
    if not document_files:
        logger.error(f"No documents found in {documents_path}")
        return {}, {}, ""
    
    logger.info(f"Found {len(document_files)} documents")
    
    # Process documents for each chunker
    for i, (config_name, chunker) in enumerate(tqdm(chunkers, desc="Processing chunkers", unit="chunker")):
        logger.info(f"[{i+1}/{len(chunkers)}] Processing with chunker: {config_name}")
        
        # Process all documents with this chunker
        for j, doc_path in enumerate(document_files):
            doc_name = os.path.basename(doc_path)
            logger.info(f"  Processing document {j+1}/{len(document_files)}: {doc_name}")
            document_text = load_document_from_file(doc_path)
            if document_text:
                document_id = os.path.basename(doc_path)
                chunks = chunker.process_document(document_text, document_id)
                
                # Store chunks in database using the correct method
                for chunk in tqdm(chunks, desc=f"Storing chunks for {doc_name}", leave=False):
                    # Handle both object chunks and dictionary chunks
                    if isinstance(chunk, dict):
                        chunk_text = chunk.get("text", "")
                        chunk_doc_id = chunk.get("document_id", document_id)
                    else:
                        chunk_text = chunk.text
                        chunk_doc_id = chunk.document_id
                        
                    db_connector.insert_chunk(
                        text=chunk_text,
                        embedding=model_connector.get_embedding(chunk_text),
                        document_id=chunk_doc_id,
                        strategy=config_name
                    )
                
                logger.info(f"    Generated {len(chunks)} chunks")
        
        # Log progress percentage
        progress = (i + 1) / len(chunkers) * 100
        logger.info(f"Parameter optimization progress: {progress:.1f}% ({i+1}/{len(chunkers)} chunkers processed)")
    
    # Initialize evaluator
    logger.info("Parameter optimization complete. Starting evaluation...")
    evaluator = get_evaluator(model_connector, db_connector, config)
    
    # Evaluate all strategies
    logger.info("Evaluating all parameter variations...")
    evaluation_results, aggregated_metrics, best_strategy = evaluator.evaluate_strategies()
    
    # Save and visualize parameter optimization results
    output_dir = config["general"]["output_directory"]
    create_directory_if_not_exists(output_dir)
    
    # Use parameter optimization dashboard for visualization
    param_dashboard = get_parameter_dashboard(config)
    
    # Save the parameter optimization results
    param_dashboard.save_results(evaluation_results, aggregated_metrics, best_strategy)
    
    logger.info(f"Evaluation complete. Best strategy: {best_strategy}")
    
    return evaluation_results, aggregated_metrics, best_strategy

def main() -> None:
    """
    Main entry point for the application.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = get_config(args.config)
    
    # If reset_db is specified in arguments, override configuration
    if args.reset_db:
        config["general"]["db_reset"] = True
    
    # Run visualization dashboard only
    if args.dashboard_only:
        logger.info("Running visualization dashboard only...")
        dashboard = get_dashboard(config)
        dashboard.run_dashboard()
        return
    
    # Run parameter optimization dashboard only
    if args.optimization_dashboard:
        logger.info("Running parameter optimization dashboard only...")
        dashboard = get_parameter_dashboard(config)
        dashboard.run_dashboard()
        return
    
    # Backward compatibility: Run parameter optimization only
    if args.parameter_optimization:
        logger.info("Running parameter optimization only mode...")
        # Create model and database connectors
        model_connector = get_model_connector(config)
        db_connector = get_db_connector(config["database"])
        
        # Reset database if configured
        if config["general"].get("db_reset", False):
            logger.info("Resetting database...")
            db_connector.reset_database()
            
        process_with_parameter_optimization(config, model_connector, db_connector)
        return
        
    # Standard run - process documents and evaluate strategies
    logger.info("Starting BetterRAG evaluation...")
    
    # Create model and database connectors
    model_connector = get_model_connector(config)
    db_connector = get_db_connector(config["database"])
    
    # Reset database if configured
    if config["general"].get("db_reset", False):
        logger.info("Resetting database...")
        db_connector.reset_database()
    
    # Check if parameter optimization is enabled in config
    enable_parameter_optimization = config.get("general", {}).get("enable_parameter_optimization", False)
    
    # Skip parameter optimization if explicitly requested
    if args.skip_parameter_optimization:
        enable_parameter_optimization = False
    
    # Parameter optimization is now part of the main workflow if enabled
    param_opt_results = None
    if enable_parameter_optimization:
        logger.info("Running parameter optimization as part of main workflow...")
        param_opt_results = process_with_parameter_optimization(config, model_connector, db_connector)
    
    # Continue with standard chunking evaluation
    logger.info("Running standard chunking evaluation...")
    
    # Get all enabled chunkers from config
    chunkers = get_all_enabled_chunkers(config)
    logger.info(f"Enabled chunking strategies: {len(chunkers)}")
    
    # Process documents
    documents_path = config["general"]["data_source"]
    logger.info(f"Processing documents from {documents_path}")
    
    # Find all documents
    document_files = find_documents(documents_path)
    
    if not document_files:
        logger.error(f"No documents found in {documents_path}")
        return
    
    logger.info(f"Found {len(document_files)} documents")
    
    # Process all documents for each chunker
    for chunker in chunkers:
        logger.info(f"Processing with chunker: {chunker.name}")
        
        for doc_path in document_files:
            document_text = load_document_from_file(doc_path)
            if document_text:
                document_id = os.path.basename(doc_path)
                chunks = chunker.process_document(document_text, document_id)
                
                # Store chunks in database
                for chunk in chunks:
                    # Handle both object chunks and dictionary chunks
                    if isinstance(chunk, dict):
                        chunk_text = chunk.get("text", "")
                        chunk_doc_id = chunk.get("document_id", document_id)
                    else:
                        chunk_text = chunk.text
                        chunk_doc_id = chunk.document_id
                        
                    db_connector.insert_chunk(
                        text=chunk_text,
                        embedding=model_connector.get_embedding(chunk_text),
                        document_id=chunk_doc_id,
                        strategy=chunker.name
                    )
                
                logger.info(f"  Generated {len(chunks)} chunks for {document_id}")
    
    # Evaluate strategies
    logger.info("Evaluating chunking strategies...")
    evaluator = get_evaluator(model_connector, db_connector, config)
    evaluation_results, aggregated_metrics, best_strategy = evaluator.evaluate_strategies()
    
    # Save results
    output_dir = config["general"]["output_directory"]
    create_directory_if_not_exists(output_dir)
    
    # Get visualization dashboard
    dashboard = get_dashboard(config)
    
    # Save and visualize results
    dashboard.save_results(evaluation_results, aggregated_metrics, best_strategy)
    dashboard.generate_matplotlib_charts()
    dashboard.generate_plotly_charts()
    
    # If parameter optimization was run, also provide its results to the dashboard
    if param_opt_results:
        param_eval_results, param_aggregated, param_best = param_opt_results
        dashboard.add_parameter_optimization_results(param_eval_results, param_aggregated, param_best)
    
    # Generate insights
    dashboard.generate_insights_summary()
    
    # Generate final report
    dashboard.generate_final_report()
    
    logger.info(f"Evaluation completed. Best strategy: {best_strategy}")
    logger.info(f"Results saved to {output_dir}")
    
    # Run the dashboard
    if config.get("visualization", {}).get("run_dashboard", False):
        logger.info("Starting visualization dashboard...")
        dashboard.run_dashboard()

if __name__ == "__main__":
    main() 