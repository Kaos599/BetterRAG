from typing import List, Dict, Any
from app.chunkers.base_chunker import BaseChunker


class FixedSizeChunker(BaseChunker):
    """
    Chunker that splits text into fixed-size chunks with optional overlap.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the fixed-size chunker with configuration parameters.
        
        Args:
            config: Dictionary containing configuration for this chunker
        """
        super().__init__(config)
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 0)
        
        # Validate configuration
        if self.chunk_overlap >= self.chunk_size:
            print(f"Warning: Overlap ({self.chunk_overlap}) is greater than or equal to chunk size ({self.chunk_size}). Setting overlap to half of chunk size.")
            self.chunk_overlap = self.chunk_size // 2
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split the input text into fixed-size chunks with specified overlap.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # If text is shorter than chunk_size, return it as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position of the current chunk
            end = min(start + self.chunk_size, len(text))
            
            # Extract the chunk
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position for the next chunk, accounting for overlap
            start = start + self.chunk_size - self.chunk_overlap
            
            # If the next chunk would be too small, merge it with the current one
            if len(text) - start < self.chunk_size / 2:
                if start < len(text):
                    chunks[-1] = chunks[-1] + text[start:]
                break
        
        return chunks
    
    def get_chunk_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the fixed-size chunking strategy.
        
        Returns:
            Dictionary with metadata including strategy parameters
        """
        return {
            "strategy_name": "fixed_size",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        } 