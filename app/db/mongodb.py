import pymongo
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
import json


class MongoDBConnector:
    """
    Connector class for MongoDB database operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MongoDB connector with configuration.
        
        Args:
            config: MongoDB configuration dictionary
        """
        self.connection_string = config.get("connection_string", "mongodb://localhost:27017/")
        self.database_name = config.get("database_name", "rag_evaluation")
        self.collection_name = config.get("collection_name", "document_chunks")
        self.client = None
        self.db = None
        self.collection = None
        self.connected = False
    
    def connect(self) -> bool:
        """
        Connect to MongoDB database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.client = pymongo.MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            
            # Check if connection is valid
            self.client.server_info()
            
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            # Create index for similarity search if it doesn't exist
            if "vector_index" not in self.collection.index_information():
                self.collection.create_index([("embedding", pymongo.ASCENDING)], name="vector_index")
            
            # Create index for strategy name
            if "strategy_index" not in self.collection.index_information():
                self.collection.create_index([("strategy", pymongo.ASCENDING)], name="strategy_index")
            
            self.connected = True
            return True
        
        except pymongo.errors.ServerSelectionTimeoutError as e:
            print(f"Error connecting to MongoDB: {e}")
            self.connected = False
            return False
        except Exception as e:
            print(f"Unexpected error connecting to MongoDB: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """
        Disconnect from MongoDB database.
        """
        if self.client:
            self.client.close()
            self.connected = False
    
    def insert_chunk(self, text: str, embedding: List[float], document_id: str, strategy: str) -> Optional[str]:
        """
        Insert a document chunk with its embedding into MongoDB.
        
        Args:
            text: The text content of the chunk
            embedding: Vector embedding for the chunk
            document_id: ID of the source document
            strategy: Name of the chunking strategy used
            
        Returns:
            Optional[str]: ID of the inserted document or None if failed
        """
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            # Create document with metadata and embedding
            document = {
                "text": text,
                "document_id": document_id,
                "strategy": strategy,
                "embedding": embedding,
                "created_at": datetime.now()
            }
            
            # MongoDB cannot store numpy arrays directly
            if isinstance(embedding, np.ndarray):
                document["embedding"] = embedding.tolist()
            
            # Insert document
            result = self.collection.insert_one(document)
            return str(result.inserted_id)
        
        except Exception as e:
            print(f"Error inserting chunk into MongoDB: {e}")
            return None
    
    def find_similar_chunks(self, 
                           query_embedding: List[float], 
                           strategy: str = None, 
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar chunks to the query embedding.
        
        Args:
            query_embedding: Embedding vector for the query
            strategy: Optional strategy name filter
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing chunk data
        """
        if not self.connected:
            if not self.connect():
                return []
        
        try:
            # Convert numpy array to list if needed
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Build the aggregation pipeline
            pipeline = [
                # Calculate vector similarity
                {
                    "$addFields": {
                        "similarity": {
                            "$reduce": {
                                "input": {"$zip": {"inputs": ["$embedding", query_embedding]}},
                                "initialValue": 0,
                                "in": {"$add": ["$$value", {"$multiply": [{"$arrayElemAt": ["$$this", 0]}, {"$arrayElemAt": ["$$this", 1]}]}]}
                            }
                        }
                    }
                }
            ]
            
            # Add strategy filter if specified
            if strategy:
                pipeline.append({"$match": {"strategy": strategy}})
            
            # Sort by similarity (descending) and limit results
            pipeline.extend([
                {"$sort": {"similarity": -1}},
                {"$limit": top_k},
                {"$project": {"embedding": 0}}  # Exclude the embedding from results
            ])
            
            # Execute the aggregation
            results = list(self.collection.aggregate(pipeline))
            return results
        
        except Exception as e:
            print(f"Error finding similar chunks in MongoDB: {e}")
            return []
    
    def clear_collection(self, strategy: str = None) -> bool:
        """
        Clear all documents in the collection or only those with a specific strategy.
        
        Args:
            strategy: Optional strategy name to filter documents to delete
            
        Returns:
            bool: True if operation successful, False otherwise
        """
        if not self.connected:
            if not self.connect():
                return False
        
        try:
            if strategy:
                self.collection.delete_many({"strategy": strategy})
            else:
                self.collection.delete_many({})
            return True
        
        except Exception as e:
            print(f"Error clearing collection in MongoDB: {e}")
            return False
    
    def get_chunk_count(self, strategy: str = None) -> int:
        """
        Get the count of chunks in the collection.
        
        Args:
            strategy: Optional strategy name to filter the count
            
        Returns:
            int: Number of chunks
        """
        if not self.connected:
            if not self.connect():
                return 0
        
        try:
            if strategy:
                return self.collection.count_documents({"strategy": strategy})
            else:
                return self.collection.count_documents({})
        
        except Exception as e:
            print(f"Error getting chunk count from MongoDB: {e}")
            return 0
    
    def get_all_strategies(self) -> List[str]:
        """
        Get a list of all unique strategy names in the collection.
        
        Returns:
            List[str]: List of strategy names
        """
        if not self.connected:
            if not self.connect():
                return []
        
        try:
            return self.collection.distinct("strategy")
        
        except Exception as e:
            print(f"Error getting strategies from MongoDB: {e}")
            return [] 