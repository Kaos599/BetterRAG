from typing import Dict, List, Any, Optional
import time
import openai
from openai import AzureOpenAI


class AzureOpenAIConnector:
    """
    Connector for Azure OpenAI API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Azure OpenAI connector with configuration.
        
        Args:
            config: Azure OpenAI configuration dictionary
        """
        self.api_key = config.get("api_key")
        self.endpoint = config.get("api_base")
        self.deployment_name = config.get("completion_deployment")
        self.embedding_deployment = config.get("embedding_deployment")
        self.api_version = config.get("api_version", "2023-05-15")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 1000)
        
        self.client = None
        self.initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the Azure OpenAI client.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not self.api_key or not self.endpoint or not self.deployment_name or not self.embedding_deployment:
            print("Error: Azure OpenAI API key, endpoint, completion deployment, or embedding deployment not provided.")
            return False
        
        try:
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint
            )
            self.initialized = True
            return True
        
        except Exception as e:
            print(f"Error initializing Azure OpenAI client: {e}")
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
        Generate embeddings for a list of texts.
        
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
            # Handle large batches by processing in chunks
            max_batch_size = 100
            embeddings = []
            total_texts = len(texts)
            
            for i in range(0, total_texts, max_batch_size):
                batch = texts[i:i + max_batch_size]
                batch_end = min(i + max_batch_size, total_texts)
                
                if total_texts > max_batch_size:
                    print(f"Embedding batch {i//max_batch_size + 1}: processing texts {i+1}-{batch_end} of {total_texts}")
                
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.embedding_deployment
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Rate limit handling - sleep if needed
                if i + max_batch_size < total_texts:
                    time.sleep(0.5)
            
            return embeddings
        
        except Exception as e:
            print(f"Error generating embeddings with Azure OpenAI: {e}")
            return None
    
    def generate_text(self, prompt: str, context: Optional[str] = None) -> Optional[str]:
        """
        Generate text using Azure OpenAI API.
        
        Args:
            prompt: The query or prompt
            context: Optional context to include
            
        Returns:
            Optional[str]: Generated text or None if failed
        """
        if not self.initialized and not self.initialize():
            return None
        
        try:
            # Prepare messages
            messages = []
            
            # Add system message with context if provided
            if context:
                messages.append({
                    "role": "system",
                    "content": f"You are a helpful assistant. Use the following context to answer the question:\n\n{context}"
                })
            else:
                messages.append({
                    "role": "system",
                    "content": "You are a helpful assistant."
                })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Generate response - remove temperature parameter
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_completion_tokens=self.max_tokens
            )
            
            # Extract and return the generated text
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error generating text with Azure OpenAI: {e}")
            return None 