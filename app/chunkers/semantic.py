from typing import List, Dict, Any
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from app.chunkers.base_chunker import BaseChunker

logger = logging.getLogger(__name__)

# Initialize sentence transformer model once at module level
_SENTENCE_TRANSFORMER = None
_MODEL_LOAD_ATTEMPTED = False

def get_sentence_transformer():
    """Get or initialize the sentence transformer model."""
    global _SENTENCE_TRANSFORMER, _MODEL_LOAD_ATTEMPTED
    
    # Only attempt to load the model once to avoid repeated log messages
    if _SENTENCE_TRANSFORMER is None and not _MODEL_LOAD_ATTEMPTED:
        try:
            _MODEL_LOAD_ATTEMPTED = True
            logger.info("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
            _SENTENCE_TRANSFORMER = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("SentenceTransformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {e}")
            _SENTENCE_TRANSFORMER = None
    return _SENTENCE_TRANSFORMER


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
        self.semantic_method = config.get("semantic_method", "standard")
        
        # For percentile method
        self.percentile_threshold = config.get("percentile_threshold", 75)
        
        # Use the shared model instance
        self.model = get_sentence_transformer()
    
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
    
    def _get_dynamic_threshold(self, similarities: List[float]) -> float:
        """
        Calculate a dynamic threshold based on the semantic method.
        
        Args:
            similarities: List of similarity scores
            
        Returns:
            Dynamic threshold value
        """
        if not similarities:
            return self.similarity_threshold
            
        if self.semantic_method == "interquartile":
            # Calculate threshold based on interquartile range
            q1 = np.percentile(similarities, 25)
            q3 = np.percentile(similarities, 75)
            iqr = q3 - q1
            threshold = q1 - (1.5 * iqr)
            # Ensure threshold is not too low
            return max(threshold, 0.5)
            
        elif self.semantic_method == "percentile":
            # Calculate threshold based on percentile
            return np.percentile(similarities, self.percentile_threshold)
            
        else:  # "standard" method
            return self.similarity_threshold
    
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
            logger.error(f"Error encoding sentences: {e}")
            return [text]  # Return original text as fallback
        
        # Pre-calculate all pairwise similarities for dynamic methods
        if self.semantic_method in ["interquartile", "percentile"]:
            all_similarities = []
            for i in range(len(sentence_embeddings) - 1):
                for j in range(i + 1, len(sentence_embeddings)):
                    similarity = self._calculate_similarity(sentence_embeddings[i], sentence_embeddings[j])
                    all_similarities.append(similarity)
            
            # Get dynamic threshold
            dynamic_threshold = self._get_dynamic_threshold(all_similarities)
        else:
            dynamic_threshold = self.similarity_threshold
        
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
            
            if (similarity >= dynamic_threshold and 
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
        metadata = {
            "strategy_name": "semantic",
            "semantic_method": self.semantic_method,
            "similarity_threshold": self.similarity_threshold,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "model": "all-MiniLM-L6-v2" if self.model else "None"
        }
        
        # Add method-specific parameters
        if self.semantic_method == "percentile":
            metadata["percentile_threshold"] = self.percentile_threshold
            
        return metadata 