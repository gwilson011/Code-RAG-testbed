"""
Step 4: Evaluate Defense Effectiveness

This script benchmarks baseline defenses (paraphrasing sanitization,
perplexity-based detection, and provenance validation) against the
poisoned vector store produced in Step 2.

Usage:
    python experiments/04_evaluate_defenses.py

Prerequisites:
    - Run 01_establish_baseline.py
    - Run 02_poison_attack.py
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
from src.evaluation import BaselineTracker
from src.defense import DefenseConfig, DefenseManager, DefenseEvaluator


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_defense_manager(enable_paraphrase=False) -> DefenseManager:
    config = DefenseConfig(
        enable_paraphrase=False,
        enable_perplexity=True,
        enable_provenance=True,
        drop_untrusted=True,
        drop_high_perplexity=True,
        perplexity_threshold=1_000_000.0,
    )
    return DefenseManager(config=config)


def main():
    print_header("STEP 4: DEFENSE BENCHMARKING (behavior-aware rerank, no hard drops)")

    # Test queries (same as other steps)
    test_queries = [
        "how to add two numbers",
        "calculate factorial",
        "reverse a string",
        "find maximum value in list",
        "check if number is prime",
        "count vowels in string",
        "filter even numbers from list",
        "calculate average of numbers",
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

    # 4. Build defense manager(s)
    print("\nBuilding defense managers...")
    defenses = {
        "provenance+perplexity": build_defense_manager(enable_paraphrase=False),
        "paraphrase+provenance+perplexity": build_defense_manager(enable_paraphrase=True),
    }
    print(f"✓ Prepared {len(defenses)} defense profiles")

    # 5. Evaluate each defense profile
    evaluator = DefenseEvaluator(baseline_tracker)
    all_results = {}

    # Collect report lines to mirror console output
    report_lines: list[str] = []
    report_lines.append("=" * 70)
    report_lines.append("DEFENSE IMPACT REPORT (behavior-aware rerank, no hard drops)")
    report_lines.append("=" * 70 + "\n")

    for name, manager in defenses.items():
        section_header = f"EVALUATING: {name}"
        print_header(section_header)
        report_lines.append(section_header)

        results = evaluator.evaluate(
            rag,
            defense_manager=manager,
            test_queries=test_queries,
            top_k=5,
            fetch_multiplier=3,
        )
        all_results[name] = results

        metrics = results["metrics"]
        tq_line = f"Total queries: {metrics['total_queries']}"
        before_line = f"Poisoned in baseline top-k: {metrics['poison_before']}"
        after_line = f"Poisoned in defended top-k: {metrics['poison_after']}"
        note1 = "Note: counts are over final per-file-deduped top_k only;"
        note2 = "      defenses primarily affect add/factorial/reverse_string implementations."
        print(tq_line)
        print(before_line)
        print(after_line)
        print(note1)
        print(note2)
        report_lines.extend([tq_line, before_line, after_line, note1, note2])

        # Show before/after full rankings for first 3 queries
        sample_queries = list(results["queries"].keys())[:3]
        for q in sample_queries:
            qdata = results["queries"][q]
            q_header = f"  Query: {q}"
            print(q_header)
            report_lines.append(q_header)

            before_header = "    Before (full fetched set):"
            print(before_header)
            report_lines.append(before_header)
            for i, chunk in enumerate(qdata["before_full"], 1):
                marker = "POISON" if chunk["metadata"].get("poisoned") else "clean"
                line = f"      {i}. {marker} - {chunk['metadata'].get('filepath')}"
                print(line)
                report_lines.append(line)

            after_header = "    After defense (full reranked set):"
            print(after_header)
            report_lines.append(after_header)
            for i, chunk in enumerate(qdata["after_full"], 1):
                marker = "POISON" if chunk["metadata"].get("poisoned") else "clean"
                line = f"      {i}. {marker} - {chunk['metadata'].get('filepath')}"
                print(line)
                report_lines.append(line)
            tail_note = "    Note: top_k is drawn from these lists."
            print(tail_note)
            report_lines.append(tail_note)

    # 6. Summarize comparison
    # 6. Summarize comparison (in console only; report already contains details)
    print_header("DEFENSE COMPARISON SUMMARY")
    for name, results in all_results.items():
        metrics = results["metrics"]
        print(f"\n{name}")
        print(f"  Poisoned in baseline top-k: {metrics['poison_before']}")
        print(f"  Poisoned in defended top-k: {metrics['poison_after']}")

    # Write defence impact report
    report_path = Config.RESULTS_DIR / "defence_impact_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print("\n" + "=" * 70)
    print("✓ DEFENSE BENCHMARKING COMPLETE")
    print("=" * 70)
    print(f"\nResults saved in: {Config.RESULTS_DIR}")
    print(f"Defense impact report: {report_path}")


if __name__ == "__main__":
    main()
