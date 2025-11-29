"""
Step 1: Establish Clean Baseline

This script builds a clean vector database and establishes baseline
retrieval behavior before any poisoning attacks.

Usage:
    python experiments/01_establish_baseline.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.ingest import ingest_codebase
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retrieval import CodeRAG
from src.evaluation import BaselineTracker


def main():
    print("="*70)
    print("STEP 1: ESTABLISH CLEAN BASELINE")
    print("="*70)

    # Define test queries
    test_queries = [
        "how to add two numbers",
        "calculate factorial",
        "reverse a string",
        "find maximum value in list",
        "check if number is prime",
        "count vowels in string",
        "filter even numbers from list",
        "calculate average of numbers"
    ]

    print(f"\nTest queries defined: {len(test_queries)}")
    for i, q in enumerate(test_queries, 1):
        print(f"  {i}. {q}")

    # 1. Ingest clean code
    print("\n" + "-"*70)
    print("Ingesting clean codebase...")
    base_dir = Path(__file__).parent.parent / "data"
    code_dirs = [base_dir / "sample_code", base_dir / "unknown_source"]
    chunks = []
    for d in code_dirs:
        chunks.extend(ingest_codebase(d))
    print(f"✓ Loaded {len(chunks)} code chunks from {len(code_dirs)} sources")

    # 2. Generate embeddings
    print("\nGenerating embeddings...")
    embedding_gen = EmbeddingGenerator()
    embeddings = embedding_gen.embed_code_chunks(chunks)
    print(f"✓ Generated {len(embeddings)} embeddings")

    # 3. Create clean vector store
    print("\nCreating clean vector store...")
    vector_store = VectorStore(collection_name="clean_baseline", reset=True)
    vector_store.add_chunks(chunks, embeddings)
    stats = vector_store.get_collection_stats()
    print(f"✓ Vector store created: {stats['total_chunks']} chunks")

    # 4. Create RAG pipeline
    print("\nInitializing RAG pipeline...")
    rag = CodeRAG(vector_store=vector_store, embedding_generator=embedding_gen)
    print("✓ RAG pipeline ready")

    # 5. Establish baseline
    print("\n" + "-"*70)
    print("Establishing baseline retrieval behavior...")
    baseline_tracker = BaselineTracker()
    baseline_data = baseline_tracker.establish_baseline(rag, test_queries, top_k=5)

    # 6. Save baseline
    baseline_tracker.save_baseline("baseline_clean.json")
    print(f"✓ Baseline saved to {Config.RESULTS_DIR / 'baseline_clean.json'}")

    # 7. Display baseline statistics
    print("\n" + "="*70)
    print("BASELINE STATISTICS")
    print("="*70)
    stats = baseline_data["statistics"]
    print(f"Total queries: {stats['total_queries']}")
    print(f"Unique files retrieved: {stats['unique_files_retrieved']}")
    print(f"Average distance: {stats['avg_distance']:.4f}")
    print(f"\nMost frequently retrieved files:")
    for file, count in stats['most_frequent_files'][:5]:
        print(f"  {file}: {count} times")

    # 8. Example query
    print("\n" + "="*70)
    print("EXAMPLE QUERY (CLEAN BASELINE)")
    print("="*70)
    example_query = "how to add two numbers"
    print(f"Query: '{example_query}'\n")

    results = rag.retrieve(example_query, top_k=5)
    print(f"Retrieved {len(results)} chunks:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['metadata']['filepath']}")
        print(f"   Distance: {result['distance']:.4f}")
        print(f"   Preview: {result['content'][:100]}...\n")

    print("="*70)
    print("✓ BASELINE ESTABLISHMENT COMPLETE")
    print("="*70)
    print("\nNext step: Run 02_poison_attack.py to inject adversarial chunks")


if __name__ == "__main__":
    main()
