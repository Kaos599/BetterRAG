from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseChunker(ABC):
    """
    Abstract base class for document chunking strategies.
    All chunking strategies should inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the chunker with configuration parameters.
        
        Args:
            config: Dictionary containing configuration for this chunker
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """
        Split the input text into chunks based on the strategy.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        pass
    
    @abstractmethod
    def get_chunk_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the chunking strategy and its parameters.
        
        Returns:
            Dictionary with metadata including strategy name and parameters
        """
        pass
    
    def process_document(self, document_text: str, document_id: str = None) -> List[Dict[str, Any]]:
        """
        Process a document by chunking it and adding metadata.
        
        Args:
            document_text: Text content of the document
            document_id: Optional ID for the document
            
        Returns:
            List of dictionaries, each containing a chunk and its metadata
        """
        chunks = self.chunk_text(document_text)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": f"{document_id}_{i}" if document_id else f"chunk_{i}",
                "text": chunk,
                "strategy": self.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "metadata": self.get_chunk_metadata()
            }
            processed_chunks.append(chunk_data)
        
        return processed_chunks
    
    def get_statistics(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Calculate and return statistics about the generated chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Dictionary with statistics like min/max/avg chunk size
        """
        if not chunks:
            return {
                "count": 0,
                "min_length": 0,
                "max_length": 0,
                "avg_length": 0,
                "total_length": 0
            }
        
        chunk_lengths = [len(chunk) for chunk in chunks]
        
        return {
            "count": len(chunks),
            "min_length": min(chunk_lengths),
            "max_length": max(chunk_lengths),
            "avg_length": sum(chunk_lengths) / len(chunks),
            "total_length": sum(chunk_lengths)
        } 