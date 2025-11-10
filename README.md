# Code-RAG Testbed

A research testbed for testing **embedding poisoning attacks** on RAG systems. Designed for security research on adversarial machine learning and LLM vulnerabilities (OWASP LLM08: Excessive Agency).

**Team Grace**: Grace Wilson, Justin Downing, Quincy McCall, Garratt Army
**Course**: ECE117 Security Research Project

## Overview

Test how adversarial chunks injected into vector databases affect RAG retrieval and generation:
- **Baseline Establishment**: Measure normal retrieval behavior
- **Attack Simulation**: Inject poisoned chunks (random, targeted, misleading)
- **Impact Analysis**: Quantify how poisoning corrupts results
- **Defense Testing**: Test mitigations (trust lists, filtering, source matching)

## Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your_key_here
```

### Run Experiments

```bash
# Complete experiment (all steps)
python experiments/run_full_experiment.py

# Or run step-by-step
python experiments/01_establish_baseline.py  # Build clean baseline
python experiments/02_poison_attack.py       # Inject poison
python experiments/03_evaluate_impact.py     # Measure impact
```

Results saved to `results/poisoning_impact_report.txt`

See **`experiments/README.md`** for detailed documentation and customization options.

## Project Structure

```
Code-RAG-testbed/
├── experiments/              # Experiment scripts
│   ├── run_full_experiment.py
│   ├── 01_establish_baseline.py
│   ├── 02_poison_attack.py
│   └── 03_evaluate_impact.py
├── src/                      # Core implementation
│   ├── poisoning.py         # Attack tools
│   ├── evaluation.py        # Metrics & baseline tracking
│   ├── ingest.py            # Code loading
│   ├── embeddings.py        # OpenAI embeddings
│   ├── vector_store.py      # ChromaDB integration
│   └── retrieval.py         # RAG pipeline
├── data/sample_code/         # Simple test dataset
├── results/                  # Experiment outputs
└── requirements.txt
```

## Research Topics

### Poisoning Attack Strategies

1. **Random Poisoning**: Inject noise chunks to test robustness
2. **Targeted Poisoning**: Chunks optimized for specific queries
3. **Misleading Examples**: Incorrect code with authoritative documentation

### Defense Mechanisms

1. **Trust Lists**: Only retrieve from verified file paths
2. **Source Matching**: Verify chunk origins
3. **Filtering**: Remove suspicious chunks
4. **Anomaly Detection**: Flag unusual embeddings
5. **Redundancy Checks**: Cross-verify information

See `experiments/README.md` for implementation examples.

## Configuration

Edit `.env` for settings:
- `OPENAI_API_KEY`: Your API key (required)
- `OPENAI_MODEL`: Generation model (default: gpt-4-turbo-preview)
- `OPENAI_EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-small)
- `CHUNK_SIZE`: Chunk size in characters (default: 1000)
- `TOP_K_RESULTS`: Results to retrieve (default: 5)

Advanced settings in `config/settings.yaml`

## Customizing Experiments

Edit experiment scripts to test different scenarios:

```python
# In 02_poison_attack.py - adjust poisoning ratio
attacker.inject_random_poison(num_chunks=20)  # More aggressive

# Modify test queries
test_queries = ["your custom queries here"]

# Add new attack strategies
attacker.inject_targeted_poison(
    target_query="sort a list",
    misleading_content="Always use bubble sort for O(1) performance"
)
```

## Troubleshooting

**Import errors**: Ensure virtual environment is activated and dependencies installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**API key errors**: Check `.env` file exists and contains valid `OPENAI_API_KEY`

**ChromaDB errors**: Delete `chroma_db/` directory and re-run

## References

- [OWASP LLM08: Excessive Agency](https://genai.owasp.org/llmrisk/llm08-excessive-agency/)
- Research focus: Embedding integrity and poisoning attacks on Code-RAG systems

## License

Educational and research purposes only.
