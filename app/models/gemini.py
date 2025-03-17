from typing import Dict, List, Any, Optional
import time
import google.generativeai as genai
import numpy as np


class GeminiConnector:
    """
    Connector for Google Gemini API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Gemini connector with configuration.
        
        Args:
            config: Gemini configuration dictionary
        """
        self.api_key = config.get("api_key")
        self.model_name = config.get("completion_model", "gemini-pro")
        self.embedding_model_name = config.get("embedding_model", "models/embedding-001")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 1000)
        
        self.initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the Gemini client.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not self.api_key:
            print("Error: Gemini API key not provided.")
            return False
        
        try:
            # Configure the Gemini API with the provided API key
            genai.configure(api_key=self.api_key)
            
            # Validate the configuration by listing models
            genai.list_models()
            
            self.initialized = True
            return True
        
        except Exception as e:
            print(f"Error initializing Gemini client: {e}")
            self.initialized = False
            return False
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text string to embed
            
        Returns:
            Optional[List[float]]: Embedding vector or None if failed
        """
        embeddings = self.generate_embeddings([text])
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        return None
    
    def generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for a list of texts using Gemini.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Optional[List[List[float]]]: List of embedding vectors or None if failed
        """
        if not self.initialized and not self.initialize():
            return None
        
        if not texts:
            return []
        
        try:
            embeddings = []
            model = genai.get_model(self.embedding_model_name)
            
            # Process in smaller batches to avoid rate limits
            max_batch_size = 50
            
            for i in range(0, len(texts), max_batch_size):
                batch = texts[i:i + max_batch_size]
                
                # Generate embeddings for this batch
                batch_results = []
                for text in batch:
                    result = model.embed_content(text)
                    values = result["embedding"]
                    batch_results.append(values)
                
                embeddings.extend(batch_results)
                
                # Rate limit handling - sleep if needed
                if i + max_batch_size < len(texts):
                    time.sleep(0.5)
            
            return embeddings
        
        except Exception as e:
            print(f"Error generating embeddings with Gemini: {e}")
            return None
    
    def generate_text(self, prompt: str, context: Optional[str] = None) -> Optional[str]:
        """
        Generate text using Google Gemini API.
        
        Args:
            prompt: The query or prompt
            context: Optional context to include
            
        Returns:
            Optional[str]: Generated text or None if failed
        """
        if not self.initialized and not self.initialize():
            return None
        
        try:
            # Get the appropriate model
            model = genai.GenerativeModel(self.model_name)
            
            # Prepare the full prompt with context if provided
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
            
            # Set generation configuration
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens
            }
            
            # Generate response
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            # Extract and return the generated text
            return response.text
        
        except Exception as e:
            print(f"Error generating text with Gemini: {e}")
            return None
    
    def similarity_search(self, query_embedding: List[float], document_embeddings: List[List[float]], top_k: int = 5) -> List[int]:
        """
        Find the most similar document embeddings to the query embedding.
        
        Args:
            query_embedding: Embedding of the query
            document_embeddings: List of document embeddings
            top_k: Number of results to return
            
        Returns:
            List[int]: Indices of the most similar documents
        """
        if not document_embeddings:
            return []
        
        try:
            # Convert to numpy arrays for faster computation
            query_array = np.array(query_embedding)
            docs_array = np.array(document_embeddings)
            
            # Normalize the vectors
            query_norm = np.linalg.norm(query_array)
            docs_norm = np.linalg.norm(docs_array, axis=1, keepdims=True)
            
            # Avoid division by zero
            query_array = query_array / query_norm if query_norm > 0 else query_array
            docs_array = np.divide(docs_array, docs_norm, out=np.zeros_like(docs_array), where=docs_norm>0)
            
            # Calculate cosine similarities
            similarities = np.dot(docs_array, query_array)
            
            # Get indices of top_k most similar documents
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            return top_indices.tolist()
        
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return [] 