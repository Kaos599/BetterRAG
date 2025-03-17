from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from app.chunkers.base_chunker import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Chunker that splits text based on semantic similarity between sentences.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the semantic chunker with configuration parameters.
        
        Args:
            config: Dictionary containing configuration for this chunker
        """
        super().__init__(config)
        self.similarity_threshold = config.get("similarity_threshold", 0.8)
        self.min_chunk_size = config.get("min_chunk_size", 50)
        self.max_chunk_size = config.get("max_chunk_size", 1000)
        
        # Initialize sentence transformer model
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Error loading sentence transformer model: {e}")
            print("Using fallback chunking method")
            self.model = None
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using basic punctuation rules.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        # Basic sentence splitting (can be improved with NLP libraries)
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        sentences = []
        current_sentence = ""
        
        i = 0
        while i < len(text):
            current_sentence += text[i]
            
            # Check if current position matches any sentence ending
            for ending in sentence_endings:
                end_len = len(ending)
                if i + end_len <= len(text) and text[i:i+end_len] == ending:
                    # Found a sentence ending, add to list and reset current sentence
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
                    i += end_len - 1  # -1 because the loop will increment i
                    break
            else:
                # No ending found, move to next character
                i += 1
        
        # Add any remaining text as a sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize the vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        # Calculate cosine similarity
        return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split the input text based on semantic similarity between sentences.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        if not text or not self.model:
            # If model failed to load or text is empty, use a simple fallback
            return [text] if text else []
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # Get embeddings for all sentences
        try:
            sentence_embeddings = self.model.encode(sentences)
        except Exception as e:
            print(f"Error encoding sentences: {e}")
            return [text]  # Return original text as fallback
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = sentence_embeddings[0]
        
        for i in range(1, len(sentences)):
            next_sentence = sentences[i]
            next_embedding = sentence_embeddings[i]
            
            # Calculate similarity with current chunk embedding
            similarity = self._calculate_similarity(current_embedding, next_embedding)
            
            # Check if the next sentence is semantically similar to the current chunk
            # or if current chunk is too small
            current_chunk_size = sum(len(s) for s in current_chunk)
            
            if (similarity >= self.similarity_threshold and 
                current_chunk_size + len(next_sentence) <= self.max_chunk_size) or \
               current_chunk_size < self.min_chunk_size:
                # Add to current chunk
                current_chunk.append(next_sentence)
                
                # Update chunk embedding as average of all sentences in chunk
                current_chunk_idx = len(current_chunk) - 1
                current_embedding = (current_embedding * current_chunk_idx + next_embedding) / (current_chunk_idx + 1)
            else:
                # Start a new chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [next_sentence]
                current_embedding = next_embedding
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def get_chunk_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the semantic chunking strategy.
        
        Returns:
            Dictionary with metadata including strategy parameters
        """
        return {
            "strategy_name": "semantic",
            "similarity_threshold": self.similarity_threshold,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "model": "all-MiniLM-L6-v2" if self.model else "None"
        } 