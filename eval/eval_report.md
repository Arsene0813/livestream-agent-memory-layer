# Evaluation Report

## Purpose

This evaluation checks whether the livestream memory layer behaves reliably in commerce-oriented scenarios.

The goal is not to claim broad benchmark performance. Instead, the evaluation verifies observable memory-layer behaviors such as structured fact retrieval, fact overwrite, entity separation, non-fact filtering, and fallback/refusal.

## Current Result

| Result | Count |
|---|---:|
| Passed | 11 |
| Failed | 0 |
| Total | 11 |

| Metric | Value |
|---|---:|
| Pass rate | 100% |
| Behavior groups covered | 5 |
| Fact types covered | 5 |
| Overwrite / lifecycle-update cases | 3 |
| Unsupported-query cases | 1 |
| Non-fact filtering cases | 1 |

Saved machine-readable summaries:

- `eval/results/eval_summary.csv`
- `eval/results/eval_summary.json`

## Data-Science-Oriented Framing

This evaluation can be understood as a small scenario-based test set.

Each case defines a behavior category, an expected behavior, and a pass/fail result. This makes the evaluation more transparent and extensible than only reporting "11/11 passed."

## Limitations

This evaluation is small-scale and scenario-based. It verifies core system behavior rather than providing a general language-model benchmark.

The current evaluation does not yet measure large-scale product coverage, noisy real-world inputs, timestamp-controlled stale-fact rejection, ambiguous product-name disambiguation, or multi-store retail operations decision support.

## Next Steps

1. Add timestamp-controlled freshness tests.
2. Add stale promotion and stale stock cases.
3. Add ambiguous product-name cases.
4. Add similar-product separation cases.
5. Update the evaluation runner so it automatically regenerates `eval/results/eval_summary.csv` and `eval/results/eval_summary.json` after each run.