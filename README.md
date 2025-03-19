<!-- BetterRAG Logo Banner -->
<div align="center">
  <img src="https://github.com/user-attachments/assets/2d68ab1c-2962-4429-ad87-f91f00a08160" alt="BetterRAG Logo" width="700px">
  <h1>BetterRAG</h1>
  <p><strong>🚀 Supercharge your RAG pipeline with optimized text chunking</strong></p>
  
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
  [![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
  [![Dashboard](https://img.shields.io/badge/Dash-Interactive-blue?logo=plotly&logoColor=white)](https://dash.plotly.com/)
</div>

## ✨ Overview

**BetterRAG** helps you find the optimal text chunking strategy for your Retrieval-Augmented Generation pipeline through rigorous, data-driven evaluation. Stop guessing which chunking method works best—measure it!

<div align="center">
  <table>
    <tr>
      <td align="center">📊 <b>Compare Strategies</b></td>
      <td align="center">⚙️ <b>Zero-Code Configuration</b></td>
      <td align="center">📈 <b>Interactive Dashboard</b></td>
    </tr>
  </table>
</div>

## 🔎 Why BetterRAG?

Text chunking can make or break your RAG system's performance. Different strategies yield dramatically different results, but the optimal approach depends on your specific documents and use case. BetterRAG provides:

- **Quantitative comparison** between chunking strategies
- **Visualized metrics** to understand performance differences
- **Clear recommendations** based on real data
- **No coding required** to evaluate and improve your pipeline

## 🛠️ Features

<table>
  <tr>
    <td width="50%">
      <h3>🧩 Multiple Chunking Strategies</h3>
      <ul>
        <li><b>Fixed-size chunking</b>: Simple token-based splitting</li>
        <li><b>Recursive chunking</b>: Follows document hierarchy</li>
        <li><b>Semantic chunking</b>: Preserves meaning and context</li>
      </ul>
    </td>
    <td width="50%">
      <h3>🤖 LLM Integration</h3>
      <ul>
        <li>Azure OpenAI compatibility</li>
        <li>Google Gemini support</li>
        <li>Extensible for other models</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>
      <h3>📊 Comprehensive Metrics</h3>
      <ul>
        <li>Context precision</li>
        <li>Token efficiency</li>
        <li>Answer relevance</li>
        <li>Latency measurement</li>
      </ul>
    </td>
    <td>
      <h3>💾 Persistent Storage</h3>
      <ul>
        <li>MongoDB integration</li>
        <li>Reuse embeddings across evaluations</li>
        <li>Cache results for faster iteration</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>
      <h3>🔬 Parameter Optimization</h3>
      <ul>
        <li>Test various overlap percentages (12%, 15%, 18%, 20%, 25%, 30%)</li>
        <li>Compare multiple chunk sizes (256, 512, 768, 1024)</li>
        <li>Evaluate different semantic methods (standard, interquartile, percentile)</li>
        <li>Find optimal parameter settings automatically</li>
      </ul>
    </td>
    <td>
      <h3>📊 Advanced Visualization</h3>
      <ul>
        <li>Parameter impact charts</li>
        <li>Heatmaps of parameter interactions</li>
        <li>Parallel coordinates for multi-parameter analysis</li>
        <li>Optimal parameter recommendations</li>
      </ul>
    </td>
  </tr>
</table>

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MongoDB (local or remote)
- API keys for Azure OpenAI and/or Google Gemini

### Installation in 3 Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/betterrag.git
cd betterrag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your configuration
cp config.template.yaml config.yaml
# Edit config.yaml with your API keys and preferences
```

### Running Your First Evaluation

```bash
# Add your documents to data/documents/

# Run the evaluation
python -m app.main

# View the interactive dashboard
# Default: http://127.0.0.1:8050/
```

## 📊 Sample Results

BetterRAG provides clear visual comparisons between chunking strategies:

<div align="center">
  <img src="https://via.placeholder.com/800x400?text=Chunking+Strategy+Comparison+Chart" alt="Comparison Chart" width="80%">
</div>

Based on comprehensive metrics, BetterRAG will recommend the most effective chunking approach for your specific documents and queries.

## ⚙️ Configuration Options

BetterRAG uses a single YAML configuration file for all settings:

```yaml
# Chunking strategies to evaluate
chunking:
  fixed_size:
    enabled: true
    chunk_size: 500
    chunk_overlap: 50
  
  recursive:
    enabled: true
    chunk_size: 1000
    separators: ["\n\n", "\n", " ", ""]
  
  semantic:
    enabled: true
    model: "all-MiniLM-L6-v2"

# API credentials (or use environment variables)
api:
  azure_openai:
    api_key: ${AZURE_OPENAI_API_KEY}
    endpoint: ${AZURE_OPENAI_ENDPOINT}
```

See [config_setup.md](config_setup.md) for detailed configuration instructions.

## 🔧 Advanced Usage

```bash
# Run dashboard only (using previously processed data)
python -m app.main --dashboard-only

# Reset database before processing
python -m app.main --reset-db

# Use custom config file
python -m app.main --config my_custom_config.yaml

# Run parameter optimization with varied configurations
python -m app.main --parameter-optimization

# Run the parameter optimization dashboard
python -m app.main --optimization-dashboard
```

## 🔬 Parameter Optimization

BetterRAG supports automated parameter optimization to find the best chunking configurations for your specific documents and use case.

### Configuration

To enable parameter optimization, add the following to your `config.yaml`:

```yaml
general:
  enable_parameter_optimization: true
  # ... other settings

parameter_optimization:
  # Maximum configurations to evaluate per strategy type (fixed_size, recursive, semantic)
  # Set to null to use all configurations (can generate over 200 combinations total)
  max_configs_per_strategy: 10
  
  # Parameter ranges for optimization
  chunk_sizes: [256, 512, 768, 1024]
  overlap_percentages: [0.15, 0.20, 0.25]
  similarity_thresholds: [0.75, 0.85]
  semantic_methods: ["standard", "interquartile"]
  min_chunk_sizes: [100, 150]
  max_chunk_sizes: [800, 1000]
```

### Limiting Configuration Count

By default, parameter optimization generates a full cartesian product of all parameter combinations, which can result in over 200 different chunking configurations. To reduce the number of configurations that need to be evaluated:

1. Use the `max_configs_per_strategy` setting to limit configurations per chunking method
2. Reduce the number of values in each parameter list
3. Use the parameter optimization dashboard to visualize results and identify promising parameter regions

### Running Parameter Optimization

You can run parameter optimization in two ways:

1. As part of the main workflow:
   ```
   python -m app.main --config config.yaml
   ```

2. Standalone:
   ```
   python -m app.main --config config.yaml --parameter-optimization
   ```

3. View results in the dashboard:
   ```
   python -m app.main --config config.yaml --optimization-dashboard
   ```

### Output

Parameter optimization results will be saved to the specified output directory and can be visualized through the parameter optimization dashboard.

The system will identify and report the best chunking configuration based on your evaluation metrics.

## 🛠️ Extending BetterRAG

### Adding a New Chunking Strategy

1. Create a new chunker implementation in `app/chunkers/`
2. Register it in `app/chunkers/__init__.py`
3. Add configuration parameters in `config.yaml`

### Custom Metrics

Extend the `ChunkingEvaluator` class in `app/evaluation/metrics.py` to add new metrics.

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs and issues
- Suggest new features or enhancements
- Add support for additional LLM providers
- Implement new chunking strategies

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  <p>Built with ❤️ for the RAG community</p>
  <p>
    <a href="https://github.com/Kaos599/betterrag/issues">Report Bug</a>
    ·
    <a href="https://github.com/Kaos599/betterrag/issues">Request Feature</a>
  </p>
</div>
