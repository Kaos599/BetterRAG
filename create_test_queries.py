import json
import os

# Define test queries
queries = [
    {
        "query": "What are the key features of chunking strategies?",
        "expected_keywords": ["chunking", "strategies", "features"]
    },
    {
        "query": "How does chunk size affect retrieval quality?",
        "expected_keywords": ["chunk", "size", "retrieval", "quality"]
    },
    {
        "query": "What makes semantic chunking different from fixed-size chunking?",
        "expected_keywords": ["semantic", "fixed-size", "different"]
    },
    {
        "query": "What are the advantages of recursive chunking?",
        "expected_keywords": ["recursive", "advantages"]
    },
    {
        "query": "How can I optimize chunking for my RAG system?",
        "expected_keywords": ["optimize", "RAG", "system"]
    }
]

# Save to file
output_path = "./test_queries.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(queries, f, indent=4)

print(f"Created test queries file at {output_path}")

# Also create data directory version
data_dir = "./data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Created directory {data_dir}")

data_path = os.path.join(data_dir, "test_queries.json")
with open(data_path, "w", encoding="utf-8") as f:
    json.dump(queries, f, indent=4)

print(f"Created test queries file at {data_path}") 