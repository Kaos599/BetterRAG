# BetterRAG: Text Chunking Evaluation for RAG Pipelines

BetterRAG is a comprehensive application for evaluating different text chunking strategies in Retrieval-Augmented Generation (RAG) pipelines. This tool helps you determine the most effective chunking approach for your specific use case without requiring any code modifications.

## Features

- **Multiple Chunking Strategies**: Fixed-size, recursive (hierarchical), and semantic chunking with configurable parameters
- **Model Integration**: Support for Azure OpenAI and Google Gemini
- **MongoDB Integration**: Stores and retrieves document embeddings
- **Comprehensive Evaluation**: Measures precision, recall, token efficiency, and more
- **Visualization Dashboard**: Interactive charts and graphs to compare chunking strategies
- **User-Friendly Configuration**: All settings defined in a single configuration file

## Setup Instructions

### Prerequisites

- Python 3.8+
- MongoDB installed and running locally (or accessible remotely)
- API keys for Azure OpenAI and/or Google Gemini

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/betterrag.git
   cd betterrag
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your configuration:
   ```
   cp config.template.yaml config.yaml
   ```

4. Edit the `config.yaml` file to add your API keys and configure your preferences.

5. Create a data directory with your source documents:
   ```
   mkdir -p data/documents
   ```

6. Add your documents to the `data/documents` directory.

### Configuration

BetterRAG uses two configuration files:

1. **`config.template.yaml`**: A template with placeholder values that is tracked in version control.
2. **`config.yaml`**: Your personal configuration file that contains actual API keys and settings. This file is ignored by Git to keep your credentials safe.

All application settings are managed through the `config.yaml` file. You can configure:

- **API Keys**: Add your Azure OpenAI or Google Gemini API keys
- **Database Settings**: Configure MongoDB connection
- **Chunking Strategies**: Enable/disable strategies and set parameters
- **Evaluation Settings**: Set metrics, weights, and retrieval parameters
- **Visualization Settings**: Configure dashboard and output formats

See [config_setup.md](config_setup.md) for detailed configuration instructions.

## Usage

1. After configuring `config.yaml`, run the application:
   ```
   python app/main.py
   ```

2. The application will:
   - Process your source documents using each enabled chunking strategy
   - Generate embeddings and store them in MongoDB
   - Run your test queries against each chunking strategy
   - Calculate evaluation metrics
   - Generate a visualization dashboard
   - Provide a final recommendation on the best chunking strategy

3. View the results:
   - Check the terminal for the summary results
   - Open the dashboard (default: http://127.0.0.1:8050/) for interactive visualizations
   - Examine the output files in the results directory

## Testing the Application

### Prepare Test Data

1. Ensure you have test documents in the `data/documents/` directory.

2. Verify that your test queries are configured in `data/test_queries.json` or update the path in your config.

### Configure Your Environment

1. Make sure your configuration file (`config.yaml`) is properly set up.

2. Set environment variables if using them for API keys:
   - Windows (PowerShell):
     ```
     $env:AZURE_OPENAI_API_KEY="your-api-key"
     $env:AZURE_OPENAI_ENDPOINT="your-endpoint"
     $env:MONGODB_CONNECTION_STRING="your-connection-string"
     ```
   - Linux/Mac:
     ```
     export AZURE_OPENAI_API_KEY=your-api-key
     export AZURE_OPENAI_ENDPOINT=your-endpoint
     export MONGODB_CONNECTION_STRING=your-connection-string
     ```

### Run the Evaluation

1. Process documents and evaluate chunking strategies:
   ```
   python -m app.main
   ```

2. For dashboard-only mode (if data is already processed):
   ```
   python -m app.main --dashboard-only
   ```

3. To reset the database before processing:
   ```
   python -m app.main --reset-db
   ```

4. To use a different configuration file:
   ```
   python -m app.main --config my_custom_config.yaml
   ```

### View Results

1. Check the generated charts and reports in the `results/` directory
2. Access the interactive dashboard at http://localhost:8050 (or configured port)
3. Review the evaluation_results.json for detailed metrics

## Customization

### Adding New Chunking Strategies

1. Create a new chunker implementation in `app/chunkers/`
2. Register it in `app/chunkers/__init__.py`
3. Add configuration parameters in `config.yaml`

### Custom Metrics

Extend the `ChunkingEvaluator` class in `app/evaluation/metrics.py` to add new metrics.

## Troubleshooting

- **MongoDB Connection Issues**: Ensure MongoDB is running and the connection string is correct
- **API Key Errors**: Verify that your API keys are correctly entered in `config.yaml`
- **Missing Results**: Check that your source documents exist in the correct directory
- **Dashboard Not Loading**: Ensure the port specified in the configuration is available
- **Configuration Issues**: Validate your YAML syntax and ensure all required fields are present

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This tool was built to help researchers and developers optimize their RAG pipelines by providing quantitative comparisons between different chunking strategies. 