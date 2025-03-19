from app.models import get_model_connector, is_valid_api_config

# Create a test config with dummy API keys
test_config = {
    'model': {
        'provider': 'azure_openai',
        'azure_openai': {
            'api_key': 'dummy_key',
            'api_base': 'https://example.com'
        }
    },
    'general': {
        'test_queries_file': './test_queries.json',
        'enable_model_caching': True
    }
}

# Test if config is valid
print('Config validation test:')
is_valid = is_valid_api_config(test_config)
print(f'Config valid: {is_valid}')

# Test model creation
print('\nModel creation test:')
model = get_model_connector(test_config)
print('Model created successfully')

# Test embedding generation
print('\nEmbedding generation test:')
test_text = 'This is a test text for embedding generation'
embeddings = model.generate_embeddings([test_text])
print(f'Generated embedding dimension: {len(embeddings[0])}')

# Test text generation
print('\nText generation test:')
test_prompt = 'What is chunking in RAG systems?'
test_context = 'Chunking is the process of dividing documents into smaller pieces for retrieval.'
response = model.generate_text(test_prompt, test_context)
print(f'Generated response: {response}')

print('\nAll tests completed successfully!') 