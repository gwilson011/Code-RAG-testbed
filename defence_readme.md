# Code-RAG Testbed – Defense Design Notes

General overview of updates that have been made as a result of defense structure.

---

## 1. File Groups and Their Roles

### 1.1 Core Defense Logic

`src/defense.py`

- **DefenseConfig**
  - Central config for defense behavior.
  - Key fields:
    - `enable_provenance`: whether to use hash‑based trust (should be set to "on").
    - `enable_perplexity`: enable/disable lightweight perplexity scoring.
    - `mode`: `"rerank"` or `"drop"` (we use rerank in experiments).
    - `trust_paths`: where we consider “known-good” code to live (currently `data/sample_code`).
- **DefenseManager**
  - Wraps the RAG retrieval results and applies defense logic.
  - Inputs: list of retrieved chunks (with `content`, `metadata`, `distance`).
  - Outputs: `DefenseOutcome` with:
    - `retained`: same chunks, but sorted by a `penalty` score.
    - `dropped`: unused in rerank mode.
  - Signals used:
    - **Provenance**: a chunk is “trusted” if its content hash is in the manifest built from `trust_paths`.
    - **Perplexity**: char n‑gram scorer trained on the trusted corpus. Used to assign mild-weights for trustworthiness.
    - **Distance stats**: mean/std of Chroma distances for this query’s candidate set.
    - **Behavioral checks** (via `BehaviorChecker`, see below).
  - Sorting:
    - Instead of overriding Chroma's similarity scores (explained later in these notes), penalties are added to the additional ranking. Lower penalties mean files show up higher within the candidate sets.
    - Note that while files are deduped in the retained top_k, console outputs for Step 4 (Evaluation of Defence) list an entire retrieved list to show the effect of the defence.
- **BehaviorChecker**
  - This component is meant to objectively assess the correctness of the suggested logic from files. Penalties are assessed to files that provide functions that do not perform as expected:
    - `add(a, b)` should satisfy `add(2, 3) == 5`, `add(-1, 1) == 0`.
    - `factorial(n)` should satisfy `factorial(0) == 1`, `factorial(5) == 120`.
    - `reverse_string(s)` should satisfy `reverse_string("abc") == "cba"`, `reverse_string("") == ""`.
  - Behavior:
    - If no relevant functions are present → no signal.
    - If tests all pass → `"pass"`; if any fails → `"fail"`.
    - `DefenseManager`:
      - Adds `behavior_fail` flag → large penalty (+3) to push chunk down.
      - Adds `behavior_pass_unknown` flag for files that are unknown/not trusted and provides a small bonus (−0.5) so unknown clean implementations get a bit of credit and can edge out files that provide incorrect implementations.
  - This helps verify that chunks/files that appear similar to query embeddings actually do what they say.
- **DefenseEvaluator**
  - This component helps us evaluate our defence.
  - For each query:
    - Fetches `fetch_multiplier × top_k` candidates based purely on similarity before defence.
    - Passes them through `DefenseManager`.
    - Dedupes to at most one chunk per file and then takes the top_k.
  - Records:
    - `poison_before`: poisoned chunks in the baseline top_k (from raw similarity).
    - `poison_after`: poisoned chunks in the defended top_k.
    - For each of the three queries that we test, we use to print the full rankings of files. Note that the deduping is not shown in console logs as the idea is to provide a clearer understanding of how the defence affects the ranking system.
  - While metadata[poisoned] appears here, it is only used in evaluation metrics and not to influence the actual process of defence..

### 1.2 Trusted “Clean” Corpus

`data/sample_code/`

- `math_utils.py`, `list_utils.py`, `string_utils.py`
  - Simple, correct reference implementations (add, factorial, reverse_string, etc.).
  - We treat these as the ground truth/baseline code for the repository:
    - Their content hashes form the initial `trusted_hashes` in `DefenseManager`.
    - The aforementioned behavior checker tests make use of these implementations.
  - The perplexity scorer is built from the text of the trusted sample_code files.

### 1.3 Unknown but Clean Alternate Sources

`data/unknown_source/`

- `clean_alt_math_utils.py`, `clean_alt_math_utils_01.py`, `clean_alt_math_utils_02.py`
- `clean_alt_list_utils.py`, `clean_alt_list_utils_01.py`, `clean_alt_list_utils_02.py`
- `clean_alt_string_utils.py`, `clean_alt_string_utils_01.py`, `clean_alt_string_utils_02.py`
- `clean_alt_string_utils_support.py`
  - These files were added to imitate a situation where we expect that a RAG db has "more" clean information than it does malicious datapoints.
  - Purpose:
    - Simulate “unknown but safe” sources you might retrieve from public or user content.
    - Effectively independent implementations with slightly different descriptors for the functions we are testing against.
    - Allow the behavior checker to detect that some unknown code is actually correct, and give it a small boost.
  - These files are included in the baseline ingest (`01_establish_baseline.py` and `run_full_experiment.py`), so they participate in retrieval and defense.

---

## 2. Defense Strategy and Rationale

### 2.1 Baseline: Pure Similarity

Without defenses, Chroma ranks chunks based on similarities between the query and the chunk embeddings. Generally speaking:

- Clean code often ranks higher than unrelated/malicious code as:
  - Docstrings and names match the queries
  - The corpus is small and curated.
- However, targetted poisons can also rank highly because:
  - It is deliberately written to be semantically similar to the query, except the code it attempts to execute is different.

### 2.2 Fragile Solutions

We iterated on several heuristics:

- Path‑based trust (directory as proxy for trust).
- Hash‑based trust (sample_code vs. everything else).
- Perplexity thresholds.
- Consensus bonuses/penalties.
- Distance z‑score outlier penalties.

These approaches might intuitively make sense, but introduce some loopholes in the defence system:

1. **Overfitting and brittleness**:
   - Small weight changes flipped rankings in unintuitive ways because of dataset size.
   - Sometimes poison was pushed down, sometimes it was accidentally pushed up.
2. **Weak signals**:
   - “Consensus” among unknown hashes isn’t inherently good (a cluster of consistent poison is still poison).
   - Perplexity/distance are rough heuristics that don’t correlate strongly with correctness.

After experimenting with different values for assigning these trust signals, we realized that trying to "outsmart" poisoning attacks/similarities wouldn't reliably produce a consistent or realistic defence..

### 2.3 What Was More Reliable

1. **Provenance (hash‑based)**

   - If a chunk’s `content_hash` matches a file in `trust_paths` (`data/sample_code`), it is “trusted”.
   - Trusted chunks within our trusted path get a small base penalty reduction.
   - All other chunks get a small base penalty.
   - This simulates the idea of “we trust our own repo/dataset more than arbitrary user data”. However, the relatively minor differences in score (-0.5 vs +0.5) ensures we do not overbias the trusted_path and allow clean user data to influence retrieval.

2. **Behavioral checks (BehaviorChecker)**

   - Mentioned more in detail earlier, but is effectively a safeguard against queries that are semantically similar but attempt to replace "true" implementations with something unknown/malicious

3. **Perplexity and distance (minor signal)**

   - Perplexity:
     - If a chunk is very off‑distribution compared to trusted code, it gets a small penalty (+0.5).
   - Distance:
     - If a chunk is an outlier in distance relative to the candidate set, it gets a small penalty via distance z‑score.
   - The signal here isn't meant to be overpowering. The idea we had is that the penalty should be just enough to nudge odd chunks down in favor of those with more similarity.

### 2.4 Dealing with Overcompensation vs a Realistic Environment

Experimenting with RAG given smaller access to datasets and storage means that some solutions that may appear feasible are actually overfitting to the size of our dataset. We kept this in mind while working on the defence to provide a more generally applicable way of addressing the issue. These are some of our considerations.

- We **do not**:
  - Use `metadata["poisoned"]` in scoring; it is only for evaluation.
  - Hard‑code directory names like `poisoned/` as a trust signal.
- We **do**:
  - Use a small, explicit trust list (`sample_code`) as a stand‑in for an internal, vetted repo.
  - Let unknown sources compete on similarity and behavior; they can earn a small trust bump by passing behavioral checks.
  - Penalize chunks whose behavior is provably wrong for core functions.

This better matches realistic RAG defense patterns in code-heavy systems based on the following assumptions:

1. You trust your own codebase more.
2. You still want to use external snippets when they look useful, but you verify behavior where possible.
3. You don’t rely solely on metrics that may sometimes be ambiguous like perplexity or directory heuristics.

---

## 3. Benefits and Tradeoffs

### 3.1 What We Gain

- **Stronger protection against obvious code poison**:
  - Poisoned implementations of `add`, `factorial`, `reverse_string` are caught by the behavioral tests and get a heavy penalty.
  - Even if they are semantically close to the query, they get pushed down.
- **Unknown but correct code can still surface**:
  - Clean alternate implementations pass behavioral checks and get a small bonus, partially offsetting their “unknown” status.
  - This lets us benefit from additional clean data points, not just the original trusted files.
- **Simpler, explainable scoring**:
  - Easier to explain on a chunk and file level why certain chunks/files rank lower than others.
  - This is more intuitive in terms of experimentation as well as debugging.

### 3.2 What We Still Don’t Solve

- **Coverage**:
  - Behavioral checks only exist for a few known functions.
  - New functions or APIs without tests fall back to similarity + mild trust/perplexity.
- **Subtle or data‑dependent bugs**:
  - Tiny tests like `factorial(5)` won’t catch all bugs or malicious behaviors.
  - An attacker could pass simple tests but still be harmful in edge cases.
- **Real‑world provenance**:
  - Our “trusted hashes” are a stand‑in for real attestation (such as signatures and org ownership), which isn't modeled here.
  - In real deployments, provenance would be more nuanced than “in this path” or “during this baseline ingest”.

### 3.3 The Issues with Adjusting Penalties

Our experiments showed that:

- Small, hand‑chosen weight changes on perplexity/consensus/distance can easily make things worse than the baseline.
- The dataset is small and synthetic, making it easy to overfit to a specific anecdotes when it may not generalize as well to more sophisticated datasets.
- The only consistently meaningful extra signal we can rely on in this testbed is behavior for known functions.

As a result:

1. Penalties within the defence system are relatively small and interpretable
2. Focused on investigating behavioral correctness as the main way of pushing potentially malicious code out of the retrieval list
3. Accept that while this solution can generalize well, it can be database/repo specific and is unlikely to be a universal solution.

---

## 4. How to Interpret Experimental Results

When you run:

`python experiments/run_full_experiment.py`

Look for the following in the console logs:

- Look at **before vs. after poison counts**:
  - For queries like “add two numbers”, “calculate factorial”, “reverse a string”.
  - Expect fewer poisoned implementations of those functions in defended top_k vs. baseline.
- Read the **before/after full rankings**:
  - Confirm that:
    - Trusted `sample_code` implementations and clean alt implementations that pass tests bubble up.
    - Poisoned code that fails tests drops down.
- Remember the limitations:
  - For functions we don’t test, defenses will be weak and rankings may look similar to baseline.

---
