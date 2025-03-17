from typing import List, Dict, Any
from app.chunkers.base_chunker import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    Chunker that recursively splits text based on a hierarchy of separators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the recursive chunker with configuration parameters.
        
        Args:
            config: Dictionary containing configuration for this chunker
        """
        super().__init__(config)
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.min_chunk_size = config.get("min_chunk_size", 100)
        self.separators = config.get("separators", ["\n\n", "\n", ". ", "! ", "? ", ", ", " "])
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Recursively split the input text using a hierarchy of separators.
        
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
        
        # Try to split using each separator in order
        return self._split_text_recursive(text, 0)
    
    def _split_text_recursive(self, text: str, separator_idx: int) -> List[str]:
        """
        Recursively split text using the separator at the given index.
        If chunks are still too large, use the next separator.
        
        Args:
            text: Text to split
            separator_idx: Index of current separator to use
            
        Returns:
            List of chunks
        """
        # Base case: if we've tried all separators or text is small enough
        if separator_idx >= len(self.separators) or len(text) <= self.chunk_size:
            # If text is still too large, use fixed-size chunking as fallback
            if len(text) > self.chunk_size:
                return self._split_by_size(text)
            return [text]
        
        # Get current separator
        separator = self.separators[separator_idx]
        
        # Split text by current separator
        splits = text.split(separator)
        
        # If splitting didn't work (only one chunk), try next separator
        if len(splits) == 1:
            return self._split_text_recursive(text, separator_idx + 1)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for split in splits:
            # Skip empty splits
            if not split.strip():
                continue
                
            # Add separator back unless it's a space
            if separator != " ":
                if current_chunk:  # Don't add separator to the start of a chunk
                    split_with_sep = separator + split
                else:
                    split_with_sep = split
            else:
                if current_chunk:  # Add space between words
                    split_with_sep = " " + split
                else:
                    split_with_sep = split
            
            # Check if adding this split would exceed chunk_size
            split_size = len(split_with_sep)
            
            if current_size + split_size > self.chunk_size and current_chunk:
                # Current chunk is full, process it and start a new one
                chunks.append(separator.join(current_chunk) if separator != " " else " ".join(current_chunk))
                
                # Start new chunk, possibly with overlap from previous chunk
                if self.chunk_overlap > 0 and current_chunk:
                    # Calculate how many elements to include from the end of the previous chunk
                    overlap_size = 0
                    overlap_elements = []
                    
                    # Add elements from the end until we reach overlap size
                    for element in reversed(current_chunk):
                        if overlap_size + len(element) > self.chunk_overlap:
                            break
                        overlap_size += len(element) + len(separator)
                        overlap_elements.insert(0, element)
                    
                    current_chunk = overlap_elements
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
            
            # Add the current split to the chunk
            current_chunk.append(split)
            current_size += split_size
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(separator.join(current_chunk) if separator != " " else " ".join(current_chunk))
        
        # Process any chunks that are still too large with the next separator
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                final_chunks.extend(self._split_text_recursive(chunk, separator_idx + 1))
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _split_by_size(self, text: str) -> List[str]:
        """
        Split text by size as a fallback method when all separators fail.
        
        Args:
            text: Text to split
            
        Returns:
            List of chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to find a space near the end to break on
            if end < len(text) and text[end] != ' ' and ' ' in text[max(start, end - 50):end]:
                # Find the last space within the last 50 characters
                end = text.rindex(' ', max(start, end - 50), end)
            
            chunks.append(text[start:end])
            start = end
            
            # Skip any spaces at the start of the next chunk
            while start < len(text) and text[start] == ' ':
                start += 1
        
        return chunks
    
    def get_chunk_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the recursive chunking strategy.
        
        Returns:
            Dictionary with metadata including strategy parameters
        """
        return {
            "strategy_name": "recursive",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "separators": self.separators
        } 