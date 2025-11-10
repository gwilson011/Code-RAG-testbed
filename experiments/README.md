# Poisoning Attack Experiments

This directory contains Python scripts for running poisoning attack experiments on the Code-RAG system.

## Quick Start

**Important**: Run all scripts from the project root directory (not from inside `experiments/`)

Run the complete experiment in one command:

```bash
# From project root
python experiments/run_full_experiment.py
```

This will:
1. Build a clean baseline database
2. Inject poisoned chunks using multiple strategies
3. Evaluate the attack's impact
4. Generate a detailed report

## Step-by-Step Experiments

For more control, run each step individually (from project root):

### Step 1: Establish Baseline

```bash
# From project root
python experiments/01_establish_baseline.py
```

Creates a clean vector database and measures normal retrieval behavior. Saves baseline to `results/baseline_clean.json`.

### Step 2: Execute Poisoning Attack

```bash
# From project root
python experiments/02_poison_attack.py
```

Injects adversarial chunks using three strategies:
- **Random noise**: Generic poisoned chunks
- **Targeted semantic**: Chunks optimized for specific queries
- **Misleading examples**: Incorrect code with authoritative documentation

### Step 3: Evaluate Impact

```bash
# From project root
python experiments/03_evaluate_impact.py
```

Compares poisoned retrieval to baseline and generates:
- Poison rates per query
- Retrieval changes from baseline
- Overall attack effectiveness metrics
- Full report saved to `results/poisoning_impact_report.txt`

## Experiment Scripts

| Script | Purpose | Prerequisites |
|--------|---------|---------------|
| `run_full_experiment.py` | Complete workflow (all 3 steps) | None |
| `01_establish_baseline.py` | Build clean baseline | None |
| `02_poison_attack.py` | Inject poisoned chunks | Step 1 |
| `03_evaluate_impact.py` | Measure attack impact | Steps 1 & 2 |

## Customizing Experiments

### Adjust Poisoning Ratio

Edit `02_poison_attack.py` or `run_full_experiment.py` to change the number of poisoned chunks:

```python
# More aggressive poisoning
attacker.inject_random_poison(num_chunks=20)  # Instead of 5
attacker.inject_targeted_poison(..., num_copies=10)  # Instead of 3
```

### Add New Test Queries

Modify the `test_queries` list in any script:

```python
test_queries = [
    "how to add two numbers",
    "your new query here",
    # ... more queries
]
```

### Test Different Poisoning Strategies

Add new poisoning attacks in `02_poison_attack.py`:

```python
# Example: Target a different operation
attacker.inject_targeted_poison(
    target_query="sort a list",
    misleading_content="Always use bubble sort for O(1) performance",
    num_copies=5
)
```

## Output Files

All results are saved to the `results/` directory:

- `baseline_clean.json`: Clean baseline retrieval data
- `poisoning_impact_report.txt`: Detailed evaluation report
- `.gitkeep`: Placeholder to track empty directory

## Common Workflows

### Quick Test

```bash
# Run everything at once
python experiments/run_full_experiment.py
```

### Iterative Testing

```bash
# Establish baseline once
python experiments/01_establish_baseline.py

# Experiment with different poisoning approaches
python experiments/02_poison_attack.py
python experiments/03_evaluate_impact.py

# Try different poisoning ratios
# (edit 02_poison_attack.py, then re-run steps 2 & 3)
```

### Compare Multiple Attacks

```bash
# Baseline (once)
python experiments/01_establish_baseline.py

# Attack 1: Light poisoning
# (edit 02 with low num_copies)
python experiments/02_poison_attack.py
python experiments/03_evaluate_impact.py
mv results/poisoning_impact_report.txt results/report_light.txt

# Attack 2: Heavy poisoning
# (edit 02 with high num_copies)
python experiments/02_poison_attack.py
python experiments/03_evaluate_impact.py
mv results/poisoning_impact_report.txt results/report_heavy.txt

# Compare report_light.txt vs report_heavy.txt
```

## Troubleshooting

### "Baseline not found" error

You need to run step 1 first:
```bash
python experiments/01_establish_baseline.py
```

### OpenAI API errors

Check your `.env` file has a valid `OPENAI_API_KEY`.

### Import errors

Make sure you're running from the project root and have installed dependencies:
```bash
pip install -r requirements.txt
```

## Next Steps

1. **Run the experiments**: Start with `run_full_experiment.py`
2. **Analyze results**: Review `results/poisoning_impact_report.txt`
3. **Implement defenses**: Add trust lists or filtering in `src/`
4. **Test defenses**: Modify scripts to use defense mechanisms
5. **Measure effectiveness**: Compare metrics before/after defenses
