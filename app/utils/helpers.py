import os
import glob
from typing import List, Dict, Any, Tuple
import json


def load_document_from_file(file_path: str) -> str:
    """
    Load document text from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Document text
    """
    try:
        # Try UTF-8 first
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            # Fall back to latin-1 (which will always work but might not be correct)
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading document from {file_path}: {e}")
            return None
    except Exception as e:
        print(f"Error loading document from {file_path}: {e}")
        return None


def find_documents(source_path: str, file_types: List[str] = None) -> List[str]:
    """
    Find documents in the source path matching the given file types.
    
    Args:
        source_path: Path to search for documents
        file_types: List of file extensions to include (e.g., [".txt", ".md"])
        
    Returns:
        List of file paths
    """
    if not file_types:
        file_types = [".txt"]
    
    # Ensure directory exists
    if not os.path.exists(source_path):
        os.makedirs(source_path, exist_ok=True)
        print(f"Created directory: {source_path}")
        return []
    
    # Find files matching the specified types
    documents = []
    for file_type in file_types:
        pattern = os.path.join(source_path, f"*{file_type}")
        documents.extend(glob.glob(pattern))
    
    return documents


def save_json(data: Any, file_path: str) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False


def load_json(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return None


def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"Created directory: {directory_path}")


def verify_config_paths(config: Dict[str, Any]) -> None:
    """
    Verify and create necessary paths specified in the configuration.
    
    Args:
        config: Configuration dictionary
    """
    # Check source documents path
    source_path = config.get("document", {}).get("source_path", "./data/source_documents/")
    create_directory_if_not_exists(source_path)
    
    # Check output directory
    output_dir = config.get("visualization", {}).get("output_directory", "./results/")
    create_directory_if_not_exists(output_dir)
    
    # Check for ground truth file
    ground_truth_path = config.get("evaluation", {}).get("ground_truth_path")
    if ground_truth_path:
        ground_truth_dir = os.path.dirname(ground_truth_path)
        create_directory_if_not_exists(ground_truth_dir) 