# Mock Implementation for Testing

## Issues Fixed

1. **Authentication Error with Azure OpenAI API**:
   - Implemented a `MockModelConnector` class that generates deterministic but random-looking embeddings
   - Added logic to detect dummy API keys using the `is_dummy_api_key` function
   - Modified `get_model_connector` to use the mock connector when dummy API keys are detected
   - Updated the `is_valid_api_config` function to accept configs with dummy API keys

2. **JSON Decoding Error with test_queries.json**:
   - Enhanced the file handling in the `evaluate_strategies` method to:
     - Look for the test_queries.json file in multiple locations
     - Create a default test_queries.json file if none is found
     - Handle JSON decoding errors gracefully
     - Use queries from the configuration if available

3. **Added Support for Parallel Evaluation**:
   - Implemented `evaluate_batch` method in the `MockModelConnector` class
   - This ensures compatibility with the parallel processing in `ChunkingEvaluator`

## Implementation Details

### MockModelConnector Class

- Provides fake but deterministic embeddings based on text hash
- Generates mock text responses that vary based on the input prompt and context
- Includes all required methods:
  - `generate_embeddings`
  - `get_embedding`
  - `generate_text`
  - `evaluate_batch`

### Configuration Changes

- Updated config.yaml to use `chunkers` instead of `chunking_strategies`
- Set dummy API keys in the configuration for testing

## Testing

- Created a test script (`test_mock_connector.py`) to verify the implementation
- The mock connector allows testing without real API keys or authentication
- Generated embeddings have the correct dimensions (1536) and are deterministic 