"""
Step 3: Evaluate Poisoning Impact

This script evaluates how the poisoning attack affected retrieval quality
by comparing poisoned results to the clean baseline.

Usage:
    python experiments/03_evaluate_impact.py

Prerequisites:
    - Run 01_establish_baseline.py first
    - Run 02_poison_attack.py second
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retrieval import CodeRAG
from src.evaluation import BaselineTracker, PoisonEvaluator


def main():
    print("="*70)
    print("STEP 3: EVALUATE POISONING IMPACT")
    print("="*70)

    # Test queries (same as baseline)
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

    # 1. Load baseline
    print("\nLoading baseline data...")
    baseline_tracker = BaselineTracker()
    try:
        baseline_tracker.load_baseline("baseline_clean.json")
        print("✓ Baseline loaded")
    except FileNotFoundError:
        print("✗ ERROR: Baseline not found. Run 01_establish_baseline.py first!")
        return

    # 2. Load poisoned vector store
    print("\nLoading poisoned vector store...")
    vector_store = VectorStore(collection_name="clean_baseline", reset=False)
    stats = vector_store.get_collection_stats()
    print(f"✓ Loaded vector store with {stats['total_chunks']} chunks")

    # 3. Create RAG pipeline
    print("\nInitializing RAG pipeline...")
    embedding_gen = EmbeddingGenerator()
    rag = CodeRAG(vector_store=vector_store, embedding_generator=embedding_gen)
    print("✓ RAG pipeline ready")

    # 4. Test example query
    print("\n" + "="*70)
    print("EXAMPLE QUERY COMPARISON")
    print("="*70)
    example_query = "how to add two numbers"
    print(f"Query: '{example_query}'\n")

    results = rag.retrieve(example_query, top_k=5)
    print(f"Retrieved {len(results)} chunks (AFTER POISONING):\n")

    for i, result in enumerate(results, 1):
        is_poisoned = result['metadata'].get('poisoned', False)
        marker = "🔴 POISONED" if is_poisoned else "✓ Clean"

        print(f"{i}. [{marker}] {result['metadata']['filepath']}")
        print(f"   Distance: {result['distance']:.4f}")
        if is_poisoned:
            print(f"   Poison type: {result['metadata'].get('poison_type', 'unknown')}")
        print(f"   Preview: {result['content'][:80]}...\n")

    # 5. Full evaluation
    print("="*70)
    print("FULL EVALUATION ACROSS ALL QUERIES")
    print("="*70)
    print("\nEvaluating poisoning impact...")
    evaluator = PoisonEvaluator(baseline_tracker)
    evaluation_results = evaluator.evaluate_poisoning(rag, test_queries)
    print("✓ Evaluation complete")

    # 6. Display metrics
    print("\n" + "="*70)
    print("IMPACT METRICS")
    print("="*70)
    metrics = evaluation_results["metrics"]

    print(f"\nTotal queries tested: {metrics['total_queries']}")
    print(f"Queries with poisoned results: {metrics['queries_with_poison']}")
    print(f"Overall poison rate: {metrics['overall_poison_rate']:.1%}")
    print(f"Avg poisoned chunks per query: {metrics['avg_poisoned_per_query']:.2f}")

    # 7. Queries ranked by impact
    print("\n" + "="*70)
    print("QUERIES RANKED BY POISONING IMPACT")
    print("="*70)

    query_impacts = []
    for query, data in evaluation_results["queries"].items():
        query_impacts.append({
            "query": query,
            "poison_rate": data["poison_rate"],
            "num_poisoned": data["num_poisoned"]
        })

    query_impacts.sort(key=lambda x: x["poison_rate"], reverse=True)

    for i, impact in enumerate(query_impacts, 1):
        status = "⚠️  HIGH" if impact["poison_rate"] >= 0.4 else "   MEDIUM" if impact["poison_rate"] >= 0.2 else "   LOW"
        print(f"\n{i}. [{status}] '{impact['query']}'")
        print(f"   Poison rate: {impact['poison_rate']:.1%}")
        print(f"   Poisoned chunks: {impact['num_poisoned']}/5")

    # 8. Retrieval changes from baseline
    print("\n" + "="*70)
    print("RETRIEVAL CHANGES FROM BASELINE")
    print("="*70)

    changes = metrics["retrieval_changes"]
    for query in test_queries[:3]:  # Show first 3 for brevity
        if query in changes:
            change = changes[query]
            print(f"\nQuery: '{query}'")
            print(f"  Overlap with baseline: {change['overlap_rate']:.1%}")
            print(f"  New chunks retrieved: {change['new_chunks']}")
            print(f"  Baseline chunks missing: {change['missing_chunks']}")

    # 9. Generate and save full report
    print("\n" + "="*70)
    print("GENERATING REPORT")
    print("="*70)
    report_path = Config.RESULTS_DIR / "poisoning_impact_report.txt"
    report = evaluator.generate_report(evaluation_results, save_path=report_path)
    print(f"✓ Full report saved to: {report_path}")

    # 10. Test RAG answer quality
    print("\n" + "="*70)
    print("RAG ANSWER QUALITY TEST")
    print("="*70)
    test_question = "How do I add two numbers in Python?"
    print(f"Question: {test_question}\n")

    result = rag.query(test_question, top_k=5, include_sources=True)

    print("Generated Answer:")
    print("-"*70)
    print(result["answer"])

    print("\n\nSources Used:")
    print("-"*70)
    poison_count = 0
    for i, source in enumerate(result["sources"], 1):
        is_poisoned = source['metadata'].get('poisoned', False)
        if is_poisoned:
            poison_count += 1
        marker = "🔴 POISONED" if is_poisoned else "✓ Clean"
        print(f"{i}. [{marker}] {source['metadata']['filepath']}")

    if poison_count > 0:
        print(f"\n⚠️  WARNING: {poison_count}/{len(result['sources'])} sources were poisoned!")
        print("    The answer may contain misleading or incorrect information.")

    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE")
    print("="*70)
    print(f"\nFull detailed report available at: {report_path}")


if __name__ == "__main__":
    main()
