"""
Evaluation utilities for measuring poisoning attack effectiveness.

This module provides tools to establish baselines and track how poisoning
affects retrieval quality and accuracy.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .config import Config
from .retrieval import CodeRAG

logger = logging.getLogger(__name__)


class BaselineTracker:
    """Tracks baseline retrieval behavior before poisoning."""

    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize baseline tracker.

        Args:
            save_dir: Directory to save baseline data
        """
        self.save_dir = save_dir or Config.RESULTS_DIR
        self.baseline_data = {
            "timestamp": None,
            "queries": {},
            "statistics": {}
        }

    def establish_baseline(
        self,
        rag: CodeRAG,
        test_queries: List[str],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Establish baseline retrieval behavior.

        Args:
            rag: RAG pipeline
            test_queries: List of queries to test
            top_k: Number of results to retrieve

        Returns:
            Baseline data dictionary
        """
        logger.info(f"Establishing baseline with {len(test_queries)} queries")

        self.baseline_data["timestamp"] = datetime.now().isoformat()
        self.baseline_data["top_k"] = top_k

        for query in test_queries:
            # Retrieve chunks for this query
            results = rag.retrieve(query, top_k=top_k)

            # Store baseline results
            self.baseline_data["queries"][query] = {
                "retrieved_chunks": [
                    {
                        "id": r["id"],
                        "filepath": r["metadata"]["filepath"],
                        "distance": r["distance"],
                        "content_preview": r["content"][:200]
                    }
                    for r in results
                ],
                "num_results": len(results),
                "top_files": self._extract_top_files(results)
            }

        # Calculate statistics
        self._calculate_baseline_statistics()

        logger.info("Baseline established successfully")
        return self.baseline_data

    def _extract_top_files(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract unique file paths from results."""
        files = [r["metadata"]["filepath"] for r in results]
        return list(dict.fromkeys(files))  # Preserve order, remove duplicates

    def _calculate_baseline_statistics(self):
        """Calculate baseline statistics."""
        all_files = []
        all_distances = []

        for query_data in self.baseline_data["queries"].values():
            all_files.extend(query_data["top_files"])
            all_distances.extend([
                chunk["distance"] for chunk in query_data["retrieved_chunks"]
            ])

        # File frequency
        file_counts = defaultdict(int)
        for file in all_files:
            file_counts[file] += 1

        self.baseline_data["statistics"] = {
            "total_queries": len(self.baseline_data["queries"]),
            "unique_files_retrieved": len(set(all_files)),
            "most_frequent_files": sorted(
                file_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "avg_distance": sum(all_distances) / len(all_distances) if all_distances else 0
        }

    def save_baseline(self, filename: str = "baseline.json"):
        """Save baseline data to file."""
        filepath = self.save_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.baseline_data, f, indent=2)
        logger.info(f"Baseline saved to {filepath}")

    def load_baseline(self, filename: str = "baseline.json") -> Dict[str, Any]:
        """Load baseline data from file."""
        filepath = self.save_dir / filename
        with open(filepath, "r") as f:
            self.baseline_data = json.load(f)
        logger.info(f"Baseline loaded from {filepath}")
        return self.baseline_data


class PoisonEvaluator:
    """Evaluates the effectiveness of poisoning attacks."""

    def __init__(self, baseline_tracker: BaselineTracker):
        """
        Initialize evaluator.

        Args:
            baseline_tracker: Baseline tracker with established baseline
        """
        self.baseline_tracker = baseline_tracker
        self.baseline_data = baseline_tracker.baseline_data

    def evaluate_poisoning(
        self,
        rag: CodeRAG,
        test_queries: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval after poisoning attack.

        Args:
            rag: RAG pipeline (with poisoned vector store)
            test_queries: Queries to test (uses baseline queries if None)
            top_k: Number of results (uses baseline top_k if None)

        Returns:
            Evaluation results comparing to baseline
        """
        if test_queries is None:
            test_queries = list(self.baseline_data["queries"].keys())

        if top_k is None:
            top_k = self.baseline_data.get("top_k", 5)

        logger.info(f"Evaluating poisoning impact with {len(test_queries)} queries")

        results = {
            "timestamp": datetime.now().isoformat(),
            "queries": {},
            "metrics": {}
        }

        # Test each query
        for query in test_queries:
            retrieved = rag.retrieve(query, top_k=top_k)

            results["queries"][query] = {
                "retrieved_chunks": [
                    {
                        "id": r["id"],
                        "filepath": r["metadata"]["filepath"],
                        "distance": r["distance"],
                        "is_poisoned": r["metadata"].get("poisoned", False),
                        "poison_type": r["metadata"].get("poison_type", None)
                    }
                    for r in retrieved
                ],
                "num_poisoned": sum(
                    1 for r in retrieved if r["metadata"].get("poisoned", False)
                ),
                "poison_rate": sum(
                    1 for r in retrieved if r["metadata"].get("poisoned", False)
                ) / len(retrieved) if retrieved else 0
            }

        # Calculate metrics
        results["metrics"] = self._calculate_metrics(results, test_queries)

        return results

    def _calculate_metrics(
        self,
        current_results: Dict[str, Any],
        test_queries: List[str]
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        metrics = {
            "overall_poison_rate": 0.0,
            "queries_with_poison": 0,
            "total_queries": len(test_queries),
            "retrieval_changes": {},
            "poisoned_chunks_by_query": {}
        }

        total_chunks = 0
        total_poisoned = 0

        for query in test_queries:
            query_results = current_results["queries"][query]
            num_poisoned = query_results["num_poisoned"]

            total_chunks += len(query_results["retrieved_chunks"])
            total_poisoned += num_poisoned

            if num_poisoned > 0:
                metrics["queries_with_poison"] += 1

            metrics["poisoned_chunks_by_query"][query] = num_poisoned

            # Compare to baseline
            if query in self.baseline_data["queries"]:
                baseline_ids = set(
                    chunk["id"] for chunk in
                    self.baseline_data["queries"][query]["retrieved_chunks"]
                )
                current_ids = set(
                    chunk["id"] for chunk in query_results["retrieved_chunks"]
                )

                overlap = len(baseline_ids & current_ids)
                metrics["retrieval_changes"][query] = {
                    "baseline_ids": list(baseline_ids),
                    "current_ids": list(current_ids),
                    "overlap": overlap,
                    "overlap_rate": overlap / len(baseline_ids) if baseline_ids else 0,
                    "new_chunks": len(current_ids - baseline_ids),
                    "missing_chunks": len(baseline_ids - current_ids)
                }

        # Overall metrics
        metrics["overall_poison_rate"] = (
            total_poisoned / total_chunks if total_chunks > 0 else 0
        )
        metrics["avg_poisoned_per_query"] = (
            total_poisoned / len(test_queries) if test_queries else 0
        )

        return metrics

    def generate_report(
        self,
        evaluation_results: Dict[str, Any],
        save_path: Optional[Path] = None
    ) -> str:
        """
        Generate a human-readable evaluation report.

        Args:
            evaluation_results: Results from evaluate_poisoning
            save_path: Optional path to save report

        Returns:
            Report string
        """
        metrics = evaluation_results["metrics"]

        report = []
        report.append("=" * 70)
        report.append("POISONING ATTACK EVALUATION REPORT")
        report.append("=" * 70)
        report.append(f"\nTimestamp: {evaluation_results['timestamp']}")
        report.append(f"\nBaseline established: {self.baseline_data['timestamp']}")

        report.append("\n\n" + "=" * 70)
        report.append("OVERALL METRICS")
        report.append("=" * 70)
        report.append(f"Total queries tested: {metrics['total_queries']}")
        report.append(f"Queries with poisoned results: {metrics['queries_with_poison']}")
        report.append(f"Overall poison rate: {metrics['overall_poison_rate']:.2%}")
        report.append(f"Avg poisoned chunks per query: {metrics['avg_poisoned_per_query']:.2f}")

        report.append("\n\n" + "=" * 70)
        report.append("RETRIEVAL CHANGES FROM BASELINE")
        report.append("=" * 70)

        for query, changes in metrics["retrieval_changes"].items():
            report.append(f"\nQuery: {query}")
            report.append(f"  Overlap with baseline: {changes['overlap_rate']:.2%}")
            report.append(f"  New chunks retrieved: {changes['new_chunks']}")
            report.append(f"  Baseline chunks missing: {changes['missing_chunks']}")
            report.append(f"  Poisoned chunks: {metrics['poisoned_chunks_by_query'][query]}")

        report.append("\n\n" + "=" * 70)
        report.append("QUERY-BY-QUERY BREAKDOWN")
        report.append("=" * 70)

        for query, data in evaluation_results["queries"].items():
            report.append(f"\nQuery: {query}")
            report.append(f"  Total retrieved: {len(data['retrieved_chunks'])}")
            report.append(f"  Poisoned: {data['num_poisoned']}")
            report.append(f"  Poison rate: {data['poison_rate']:.2%}")

            if data["num_poisoned"] > 0:
                report.append("  Poisoned chunks:")
                for chunk in data["retrieved_chunks"]:
                    if chunk["is_poisoned"]:
                        report.append(f"    - {chunk['id']} (type: {chunk['poison_type']})")

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, "w") as f:
                f.write(report_text)
            logger.info(f"Report saved to {save_path}")

        return report_text


def create_evaluator(
    rag: CodeRAG,
    test_queries: List[str],
    save_dir: Optional[Path] = None
) -> tuple[BaselineTracker, PoisonEvaluator]:
    """
    Convenience function to create baseline and evaluator.

    Args:
        rag: RAG pipeline (unpoisoned)
        test_queries: Test queries for baseline
        save_dir: Directory to save results

    Returns:
        Tuple of (BaselineTracker, PoisonEvaluator)
    """
    tracker = BaselineTracker(save_dir=save_dir)
    tracker.establish_baseline(rag, test_queries)

    evaluator = PoisonEvaluator(tracker)

    return tracker, evaluator
