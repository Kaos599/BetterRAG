# BetterRAG Configuration Setup

This guide explains how to set up your configuration file for BetterRAG.

## Configuration Files

BetterRAG uses two configuration files:

1. **`config.template.yaml`**: A template with placeholder values that is tracked in version control.
2. **`config.yaml`**: Your personal configuration file that contains actual API keys and settings. This file is ignored by Git to keep your credentials safe.

## Setup Instructions

### Initial Setup

1. Copy the template file to create your personal configuration:

   ```bash
   # For Linux/Mac
   cp config.template.yaml config.yaml
   
   # For Windows (Command Prompt)
   copy config.template.yaml config.yaml
   
   # For Windows (PowerShell)
   Copy-Item config.template.yaml config.yaml
   ```

2. Edit the `config.yaml` file to add your personal settings:
   - Add your API keys
   - Configure database connection
   - Adjust chunking parameters as needed
   - Customize evaluation metrics

### API Keys Configuration

You can add API keys in two ways:

1. **Direct value**: Add your API key directly in the config file
   ```yaml
   azure_openai:
     api_key: "your-actual-api-key-here"
   ```

2. **Environment variable**: Reference an environment variable (recommended for security)
   ```yaml
   azure_openai:
     api_key: "${AZURE_OPENAI_API_KEY}"
   ```
   
   Then set the environment variable in your terminal:
   ```bash
   # For Linux/Mac
   export AZURE_OPENAI_API_KEY="your-api-key"
   
   # For Windows (Command Prompt)
   set AZURE_OPENAI_API_KEY=your-api-key
   
   # For Windows (PowerShell)
   $env:AZURE_OPENAI_API_KEY="your-api-key"
   ```

### Database Configuration

Configure your MongoDB connection:

```yaml
database:
  provider: "mongodb"
  mongodb:
    connection_string: "mongodb://localhost:27017/"  # Default local MongoDB
    database_name: "betterrag"
    collection_name: "chunks"
```

For a remote MongoDB instance, update the connection string with your credentials:

```yaml
connection_string: "mongodb+srv://username:password@your-cluster.mongodb.net/"
```

### Adjusting Chunking Strategies

You can enable/disable chunking strategies and adjust their parameters:

```yaml
chunking_strategies:
  fixed_size:
    enabled: true  # Set to false to disable this strategy
    chunk_size: 500  # Adjust as needed
    chunk_overlap: 50  # Adjust as needed
```

## Running the Application

After setting up your configuration, run the application:

```bash
python -m app.main
```

## Security Considerations

1. **Never commit** your `config.yaml` file to version control.
2. **Use environment variables** for sensitive information whenever possible.
3. If you modify the structure of the configuration, update the template file.

## Troubleshooting

If the application cannot find your configuration:

1. Verify that `config.yaml` exists in the root directory of the project.
2. Check for syntax errors in your YAML file.
3. Ensure all required fields are properly filled. 