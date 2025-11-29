"""
Defense utilities for mitigating poisoning attacks in the Code-RAG testbed.

Implements three baseline defenses:
1) Paraphrasing sanitization: normalize suspicious text before use.
2) Perplexity-based detection: flag outlier text relative to a trusted corpus.
3) Provenance validation: enforce trusted source paths / hashes.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .config import Config

logger = logging.getLogger(__name__)


# -------------------------
# Utility helpers
# -------------------------


def sha256_text(text: str) -> str:
    """Return the SHA256 hex digest of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest of a file."""
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha256(data).hexdigest()


# -------------------------
# Perplexity scoring
# -------------------------


class PerplexityScorer:
    """
    Simple character-level n-gram perplexity scorer.

    This is lightweight and self-contained to avoid heavy dependencies.
    The scorer is trained on trusted corpus text and used to flag
    high-perplexity (unnatural) chunks that may be adversarial.
    """

    def __init__(self, n: int = 3, floor: float = 1e-6):
        self.n = n
        self.floor = floor
        self.counts: Dict[str, int] = {}
        self.total = 0

    @classmethod
    def from_texts(cls, texts: List[str], n: int = 3) -> "PerplexityScorer":
        scorer = cls(n=n)
        for text in texts:
            scorer._ingest_text(text)
        return scorer

    def _ingest_text(self, text: str):
        padded = ("~" * (self.n - 1)) + text
        for i in range(len(padded) - self.n + 1):
            ngram = padded[i : i + self.n]
            self.counts[ngram] = self.counts.get(ngram, 0) + 1
            self.total += 1

    def score(self, text: str) -> float:
        """
        Compute a perplexity-like score (lower is more likely / natural).
        """
        if not text:
            return 0.0

        padded = ("~" * (self.n - 1)) + text
        log_prob_sum = 0.0
        ngrams = 0

        for i in range(len(padded) - self.n + 1):
            ngram = padded[i : i + self.n]
            count = self.counts.get(ngram, 0)
            prob = count / self.total if self.total > 0 else 0
            prob = max(prob, self.floor)  # avoid log(0)
            log_prob_sum += -math.log(prob)
            ngrams += 1

        if ngrams == 0:
            return float("inf")

        avg_log_prob = log_prob_sum / ngrams
        return math.exp(avg_log_prob)


# -------------------------
# Defense configuration
# -------------------------


@dataclass
class DefenseConfig:
    enable_paraphrase: bool = False
    enable_perplexity: bool = True
    enable_provenance: bool = True
    perplexity_threshold: float = 500.0  # moderate default; adjust to tune sensitivity
    drop_untrusted: bool = False  # default to rerank instead of hard drop
    drop_high_perplexity: bool = False
    mode: str = "rerank"  # "rerank" (default) or "drop"
    trust_paths: List[Path] = field(
        default_factory=lambda: [Config.PROJECT_ROOT / "data" / "sample_code"]
    )
    max_paraphrase_chars: int = 1200  # guard against very long paraphrases


@dataclass
class DefenseDecision:
    keep: bool
    flags: List[str]
    trusted: bool
    penalty: float
    perplexity: Optional[float] = None
    sanitized_content: Optional[str] = None


@dataclass
class DefenseOutcome:
    retained: List[Dict[str, Any]]
    dropped: List[Dict[str, Any]]
    stats: Dict[str, Any]


# -------------------------
# Defense manager
# -------------------------


class DefenseManager:
    """
    Applies defense checks (provenance, perplexity, sanitization) to retrieved chunks.
    """

    def __init__(
        self,
        config: Optional[DefenseConfig] = None,
        trust_map: Optional[Dict[str, str]] = None,
        perplexity_scorer: Optional[PerplexityScorer] = None,
    ):
        self.config = config or DefenseConfig()
        self.trust_map = trust_map or self._build_trust_map(self.config.trust_paths)
        self.trusted_hashes = set(self.trust_map.values())
        self.perplexity_scorer = (
            perplexity_scorer or self._build_default_perplexity_scorer()
        )
        self._paraphraser = None
        if self.config.enable_paraphrase:
            Config.validate()
            self._paraphraser = OpenAI(api_key=Config.OPENAI_API_KEY)
        self._hash_counts: Counter[str] = Counter()
        self._distance_stats: Tuple[float, float] = (0.0, 1.0)
        self._min_trusted_distance: Optional[float] = None
        self._avg_trusted_distance: Optional[float] = None

    # -------------------------
    # Public API
    # -------------------------

    def filter_chunks(
        self, chunks: List[Dict[str, Any]], query: str
    ) -> DefenseOutcome:
        """
        Apply defenses to retrieved chunks.

        Args:
            chunks: Retrieved chunk dictionaries
            query: Original query text (for context; used in paraphrasing)
        """
        retained = []
        dropped = []
        flagged = 0
        processed = []

        # Track hash frequencies for consensus-like scoring
        hashes = []
        distances = []
        trusted_distances = []
        for chunk in chunks:
            h = chunk.get("metadata", {}).get("content_hash")
            if h:
                hashes.append(h)
            dist = chunk.get("distance")
            if dist is not None:
                distances.append(dist)
                if self._is_trusted_chunk(chunk):
                    trusted_distances.append(dist)
        self._hash_counts = Counter(hashes)
        if distances:
            mean = sum(distances) / len(distances)
            var = sum((d - mean) ** 2 for d in distances) / len(distances)
            std = math.sqrt(var) if var > 0 else 1.0
            self._distance_stats = (mean, std if std > 0 else 1.0)
        self._min_trusted_distance = min(trusted_distances) if trusted_distances else None
        self._avg_trusted_distance = (
            sum(trusted_distances) / len(trusted_distances) if trusted_distances else None
        )

        for chunk in chunks:
            decision = self._evaluate_chunk(chunk)
            chunk_copy = dict(chunk)
            chunk_copy.setdefault("defense", {})
            chunk_copy["defense"].update(
                {
                    "flags": decision.flags,
                    "perplexity": decision.perplexity,
                    "sanitized": bool(decision.sanitized_content),
                    "trusted": decision.trusted,
                    "penalty": decision.penalty,
                }
            )

            if decision.sanitized_content:
                chunk_copy["content"] = decision.sanitized_content

            if decision.flags:
                flagged += 1

            if self.config.mode == "drop" and not decision.keep:
                dropped.append(chunk_copy)
            else:
                processed.append(chunk_copy)

        if self.config.mode == "rerank":
            # Lower penalty = better rank
            processed.sort(key=lambda c: c["defense"].get("penalty", 0))
            retained = processed
        else:
            retained = processed

        stats = {
            "total": len(chunks),
            "retained": len(retained),
            "dropped": len(dropped),
            "flagged": flagged,
            "drop_rate": (len(dropped) / len(chunks)) if chunks else 0,
        }

        return DefenseOutcome(retained=retained, dropped=dropped, stats=stats)

    # -------------------------
    # Internal helpers
    # -------------------------

    def _evaluate_chunk(self, chunk: Dict[str, Any]) -> DefenseDecision:
        flags: List[str] = []
        perplexity_score = None

        # Provenance validation
        is_trusted = self._is_trusted_chunk(chunk) if self.config.enable_provenance else True
        if self.config.enable_provenance and not is_trusted:
            flags.append("untrusted_source")

        # Perplexity detection
        if self.config.enable_perplexity and self.perplexity_scorer:
            perplexity_score = self.perplexity_scorer.score(chunk["content"])
            if perplexity_score > self.config.perplexity_threshold:
                flags.append("high_perplexity")

        # Paraphrasing sanitization (best-effort, may be disabled)
        sanitized_content = None
        if self.config.enable_paraphrase and flags:
            sanitized_content = self._paraphrase(chunk["content"])

        keep = True
        if self.config.mode == "drop":
            if "untrusted_source" in flags and self.config.drop_untrusted:
                keep = False
            elif "high_perplexity" in flags and self.config.drop_high_perplexity and not is_trusted:
                keep = False

        # Lightweight behavioral checks for code chunks
        behavior_result = BehaviorChecker.evaluate_chunk(chunk)
        if behavior_result == "fail":
            flags.append("behavior_fail")
        elif behavior_result == "pass" and not is_trusted:
            flags.append("behavior_pass_unknown")

        penalty = self._compute_penalty(flags, is_trusted, chunk)

        return DefenseDecision(
            keep=keep,
            flags=flags,
            trusted=is_trusted,
            penalty=penalty,
            perplexity=perplexity_score,
            sanitized_content=sanitized_content,
        )

    def _is_trusted_chunk(self, chunk: Dict[str, Any]) -> bool:
        metadata = chunk.get("metadata", {})
        content_hash = metadata.get("content_hash")
        if not content_hash:
            return False

        # Trust if content hash matches a known trusted hash (regardless of path)
        return content_hash in self.trusted_hashes

    def _paraphrase(self, text: str) -> Optional[str]:
        """
        Best-effort paraphrasing using OpenAI; returns None on failure.
        """
        if not self._paraphraser:
            return None

        shortened = text[: self.config.max_paraphrase_chars]
        prompt = (
            "Paraphrase the following text to preserve meaning but remove persuasion, "
            "marketing language, and misleading claims. Keep code intact if present.\n\n"
            f"{shortened}"
        )
        try:
            response = self._paraphraser.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a neutral rewriting assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Paraphrasing failed: {e}")
            return None

    def _compute_penalty(self, flags: List[str], is_trusted: bool, chunk: Dict[str, Any]) -> float:
        """
        Assign a penalty score used for reranking. Lower is better.
        Simple combination of provenance, perplexity, distance, and behavioral checks.
        """
        # Base: provenance
        penalty = -0.5 if is_trusted else 0.5

        # Perplexity / untrusted flags (small adjustments)
        if "high_perplexity" in flags:
            penalty += 0.5
        if "untrusted_source" in flags:
            penalty += 0.5
        if "behavior_fail" in flags:
            penalty += 3.0
        if "behavior_pass_unknown" in flags:
            penalty -= 0.5

        # Distance-based outlier penalty (higher distance = more outlier)
        dist = chunk.get("distance")
        mean, std = self._distance_stats
        if dist is not None and std > 0:
            z = (dist - mean) / std
            if z > 1.0:
                penalty += min(z, 3) * 0.3  # cap contribution
        return penalty

    def _build_trust_map(self, roots: List[Path]) -> Dict[str, str]:
        trust_map: Dict[str, str] = {}
        for root in roots:
            root_path = Path(root)
            if not root_path.exists():
                continue
            for path in root_path.rglob("*"):
                if path.is_file():
                    trust_map[str(path)] = sha256_file(path)
        logger.info(f"Built trust map with {len(trust_map)} files from {len(roots)} roots")
        return trust_map

    def _build_default_perplexity_scorer(self) -> Optional[PerplexityScorer]:
        corpus_texts: List[str] = []
        for root in self.config.trust_paths:
            root_path = Path(root)
            if not root_path.exists():
                continue
            for path in root_path.rglob("*.py"):
                try:
                    corpus_texts.append(path.read_text())
                except Exception:
                    continue

        if not corpus_texts:
            logger.warning("Perplexity scorer: no corpus texts found; disabling perplexity defense")
            return None

        return PerplexityScorer.from_texts(corpus_texts, n=3)


class BehaviorChecker:
    """
    Very simple behavioral checks for key code functions.
    Used as an additional signal on top of similarity.
    """

    @staticmethod
    def evaluate_chunk(chunk: Dict[str, Any]) -> Optional[str]:
        content = chunk.get("content", "")
        filepath = chunk.get("metadata", {}).get("filepath", "")

        # Only attempt checks for Python files with obvious function definitions
        if not filepath.endswith(".py"):
            return None
        if "def " not in content:
            return None

        env: Dict[str, Any] = {}
        try:
            exec(content, env, env)
        except Exception:
            return None

        tests: List[bool] = []

        # Math utilities
        add_fn = env.get("add")
        if callable(add_fn):
            try:
                tests.append(add_fn(2, 3) == 5)
                tests.append(add_fn(-1, 1) == 0)
            except Exception:
                tests.append(False)

        factorial_fn = env.get("factorial")
        if callable(factorial_fn):
            try:
                tests.append(factorial_fn(0) == 1)
                tests.append(factorial_fn(5) == 120)
            except Exception:
                tests.append(False)

        # String utilities
        rev_fn = env.get("reverse_string")
        if callable(rev_fn):
            try:
                tests.append(rev_fn("abc") == "cba")
                tests.append(rev_fn("") == "")
            except Exception:
                tests.append(False)

        # If no relevant functions found, no signal
        if not tests:
            return None

        return "pass" if all(tests) else "fail"


# -------------------------
# Defense evaluation
# -------------------------


class DefenseEvaluator:
    """
    Evaluate defenses by comparing poisoned retrievals before/after defenses.
    """

    def __init__(self, baseline_tracker):
        self.baseline_tracker = baseline_tracker
        self.baseline_data = baseline_tracker.baseline_data

    def evaluate(
        self,
        rag,
        defense_manager: DefenseManager,
        test_queries: List[str],
        top_k: int = 5,
        fetch_multiplier: int = 3,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "queries": {},
            "metrics": {},
        }

        total_poison_before = 0
        total_poison_after = 0
        total_dropped = 0
        total_retrieved = 0

        for query in test_queries:
            # Pull a wider pool so reranking can push poisoned chunks out of top_k
            fetch_k = top_k * fetch_multiplier if fetch_multiplier > 1 else top_k
            retrieved = rag.retrieve(query, top_k=fetch_k)
            defense_outcome = defense_manager.filter_chunks(retrieved, query)

            before_top = retrieved[:top_k]
            poisoned_before = sum(
                1 for r in before_top if r["metadata"].get("poisoned", False)
            )
            # Evaluate top_k after defenses (reranked/dropped)
            retained_top = self._unique_per_file(defense_outcome.retained, top_k)
            poisoned_after = sum(
                1 for r in retained_top if r["metadata"].get("poisoned", False)
            )

            total_poison_before += poisoned_before
            total_poison_after += poisoned_after
            total_dropped += len(defense_outcome.dropped)
            total_retrieved += len(retained_top)

            results["queries"][query] = {
                "retrieved_before": len(before_top),
                "retrieved_after": len(retained_top),
                "poison_before": poisoned_before,
                "poison_after": poisoned_after,
                "drop_stats": defense_outcome.stats,
                "before_top": before_top,
                "retained_top": retained_top,
                "before_full": retrieved,
                "after_full": defense_outcome.retained,
            }

        metrics = {
            "total_queries": len(test_queries),
            "poison_before": total_poison_before,
            "poison_after": total_poison_after,
            "poison_reduction": (
                (total_poison_before - total_poison_after) / total_poison_before
                if total_poison_before
                else 0
            ),
            "avg_drop_rate": (total_dropped / total_retrieved) if total_retrieved else 0,
        }

        results["metrics"] = metrics
        return results

    def _unique_per_file(self, chunks: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Return up to `limit` chunks with at most one per filepath."""
        seen = set()
        result = []
        for chunk in chunks:
            filepath = chunk.get("metadata", {}).get("filepath")
            if filepath in seen:
                continue
            seen.add(filepath)
            result.append(chunk)
            if len(result) >= limit:
                break
        return result
