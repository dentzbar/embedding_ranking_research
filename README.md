# Text Embedding Compression Research

This repository implements a research framework for text compression using BERT and OpenAI embeddings. Given a reference word, it ranks the full vocabulary by cosine similarity and saves the results as hash tables for efficient lookup.

## 🎯 Project Overview

The system creates vocabulary rankings based on semantic similarity to reference words, enabling analysis of text compression potential. It supports both BERT and OpenAI embedding models and provides comprehensive analysis tools.

## 📁 Project Structure

```
text_embedding_compression/
├── embedding_ranker.py      # Core classes for embedding-based ranking
├── run.py                   # Main script for generating rankings
├── utils.py                 # Utility functions for analysis
├── analysis.ipynb           # Jupyter notebook for comprehensive analysis
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── Generated files:
    ├── *_ranking.pkl        # Saved ranking dictionaries (pickle format)
    ├── *_ranking.json       # Saved ranking dictionaries (JSON format)
    ├── analysis_results.csv # Analysis results export
    └── text_summary_stats.csv # Summary statistics
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the repository
cd text_embedding_compression

# Install dependencies
pip install -r requirements.txt
```

### 2. Set up OpenAI API (if using OpenAI embeddings)

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Generate Rankings

#### Command Line Usage

```bash
# Generate BERT ranking for a word
python run.py --word "machine" --model bert --format pickle

# Generate OpenAI ranking for a word
python run.py --word "learning" --model openai --format json

# Use specific models
python run.py --word "computer" --model bert --bert-model bert-large-uncased
python run.py --word "science" --model openai --openai-model text-embedding-3-large
```

#### Interactive Mode

```bash
# Run without arguments for interactive mode
python run.py
```

### 4. Analyze Results

```bash
# Start Jupyter notebook
jupyter notebook analysis.ipynb

# Or run utility functions directly
python utils.py
```

## 📚 Core Components

### EmbeddingRanker Classes

#### Base Class: `EmbeddingRanker`
- Abstract base class defining the interface
- Handles cosine similarity computation and ranking logic
- Supports both pickle and JSON output formats

#### BERT Implementation: `BertEmbeddingRanker`
- Uses Hugging Face transformers
- Token-level embeddings from BERT's word embedding layer
- Keys: Token strings (e.g., "machine", "##ing")
- Default model: `bert-base-uncased`

#### OpenAI Implementation: `OpenAIEmbeddingRanker`
- Uses OpenAI's embedding API
- Supports latest embedding models
- Keys: Token IDs for faster lookup
- Default model: `text-embedding-3-small`

### Key Features

- **Efficient Storage**: Hash tables with vocabulary items as keys and ranks as values
- **Fast Lookup**: Optimized key structure (strings for BERT, IDs for OpenAI)
- **Comprehensive Analysis**: Statistical analysis of compression potential
- **Multiple Formats**: Support for both pickle (binary) and JSON (human-readable)
- **Batch Processing**: Analyze multiple texts simultaneously

## 🔧 Usage Examples

### Generate Rankings

```python
from embedding_ranker import BertEmbeddingRanker, OpenAIEmbeddingRanker

# BERT ranking
bert_ranker = BertEmbeddingRanker()
bert_ranking = bert_ranker.rank_vocabulary("machine")
bert_ranker.save_ranking(bert_ranking, "machine", format="pickle")

# OpenAI ranking
openai_ranker = OpenAIEmbeddingRanker()
openai_ranking = openai_ranker.rank_vocabulary("learning")
openai_ranker.save_ranking(openai_ranking, "learning", format="json")
```

### Analyze Text

```python
from utils import get_text_ranks, analyze_text_compression

# Get ranks for text tokens
text = "Machine learning algorithms are powerful"
ranks = get_text_ranks("bertembeddingranker_machine_ranking.pkl", text, "bert")

# Comprehensive analysis
analysis = analyze_text_compression(
    "bertembeddingranker_machine_ranking.pkl", 
    text, 
    "bert", 
    show_details=True
)

print(f"Compression potential: {analysis['compression_potential']:.2%}")
print(f"Mean rank: {analysis['mean_rank']:.1f}")
```

### Batch Analysis

```python
from utils import batch_analyze_texts

texts = [
    "Natural language processing",
    "Machine learning algorithms", 
    "Deep neural networks"
]

results = batch_analyze_texts(
    "bertembeddingranker_machine_ranking.pkl",
    texts,
    "bert"
)

for result in results:
    print(f"Text {result['text_id']}: {result['compression_potential']:.2%}")
```

## 📊 Analysis Features

### Compression Metrics

- **Mean Rank**: Average rank of tokens (lower = more similar to reference)
- **Compression Potential**: Percentage of high-rank (dissimilar) tokens
- **Coverage**: Percentage of tokens found in vocabulary
- **Rank Distribution**: Statistical analysis of rank patterns

### Visualization

The `analysis.ipynb` notebook provides:

- **Heatmaps**: Text vs reference word comparisons
- **Box plots**: Model performance comparisons  
- **Bar charts**: Coverage analysis across texts
- **Summary statistics**: Comprehensive performance metrics

### Export Options

- CSV files for further analysis
- JSON/pickle formats for programmatic access
- Visualization plots in various formats

## ⚡ Performance Considerations

### BERT
- **Fast**: Uses pre-computed token embeddings
- **Memory**: ~500MB for bert-base-uncased
- **Vocabulary**: ~30K tokens
- **Best for**: Quick prototyping and analysis

### OpenAI
- **API-dependent**: Requires internet and API key
- **Rate limits**: Batch processing recommended
- **Cost**: Pay-per-use API calls
- **Vocabulary**: ~100K tokens
- **Best for**: Production applications with budget

### Optimization Tips

1. **Use pickle format** for faster loading
2. **Cache embeddings** for repeated analysis
3. **Batch API calls** for OpenAI to reduce costs
4. **Use smaller models** for development/testing

## 🔍 File Formats

### Ranking Files

**BERT Example** (`bertembeddingranker_machine_ranking.pkl`):
```python
{
    "machine": 1,           # Most similar
    "machinery": 2,
    "mechanical": 3,
    # ... rest of vocabulary
    "[UNK]": 30522         # Least similar
}
```

**OpenAI Example** (`openaiembeddingranker_learning_ranking.json`):
```json
{
    "1234": 1,             # Token ID: rank
    "5678": 2,
    "9012": 3
}
```

### Analysis Output

**CSV Export** (`analysis_results.csv`):
```csv
Text_Name,Model,Reference_Word,Mean_Rank,Compression_Potential
Technical,BERT,machine,1250.5,0.25
Simple,BERT,machine,2100.3,0.15
```

## 🛠️ Advanced Usage

### Custom Models

```python
# Use different BERT variants
ranker = BertEmbeddingRanker(model_name="bert-large-uncased")

# Use different OpenAI models  
ranker = OpenAIEmbeddingRanker(model_name="text-embedding-3-large")
```

### Custom Analysis

```python
from utils import compare_rankings

# Compare multiple ranking files
results = compare_rankings(
    ["bert_machine.pkl", "openai_machine.json"],
    "The machine learning algorithm",
    ["bert", "openai"]
)
```

### Integration with Other Tools

```python
# Load rankings for external use
import pickle

with open("bertembeddingranker_machine_ranking.pkl", "rb") as f:
    ranking = pickle.load(f)

# Use in your own analysis
def custom_text_scorer(text, ranking):
    tokens = text.split()  # Simple tokenization
    scores = [ranking.get(token, len(ranking)) for token in tokens]
    return sum(scores) / len(scores)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is for research purposes. Please cite appropriately if used in academic work.

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API errors**: Check API key and rate limits
2. **Memory issues**: Use smaller models or reduce batch sizes
3. **Missing dependencies**: Run `pip install -r requirements.txt`
4. **CUDA issues**: Install appropriate PyTorch version for your system

### Performance Tips

- Use GPU acceleration for BERT models when available
- Cache ranking files to avoid recomputation
- Use appropriate batch sizes for your hardware
- Monitor API usage for OpenAI models

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the analysis notebook examples
3. Create an issue with detailed error information 