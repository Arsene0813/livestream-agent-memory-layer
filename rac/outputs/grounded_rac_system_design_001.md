# Grounded RAC Report

## 1. Direct Answer

RAC operates as a review layer above the existing typed memory system while leaving existing endpoints unchanged.

Deterministic local-file review; routing scores summarize evidence coverage under the current rules.

## 2. Question Type

- Question type: technical_design
- Domain: ai_system_design

## 3. Factor Weights

### 3a. How Factor Weights Are Generated

Factor weights are fixed review-priority buckets assigned by explicit rules in `rac/src/mock_pipeline.py`. They order review attention within the current evidence scope.

| Bucket | Weight | Rule | Factors in This Report |
|---|---:|---|---|
| high | 0.85 | Central to avoiding overconfident or misleading conclusions. | evidence_packets, belief_records, retrieval_trace |
| medium | 0.72 | Important context but not sufficient on its own. | typed_memory, hypotheses, confidence, limitations, active_state_filtering |
| default | 0.60 | Potentially relevant but requires stronger evidence. | none |

Weighting boundary:

- Use these values only to order review attention within the current evidence scope.

### 3b. Factor Weights Used in This Report

| Decision Factor ID | Weight | Bucket | Evidence Status | Why It Matters |
|---|---:|---|---|---|
| typed_memory | 0.72 | medium | partially_supported | Important context but not sufficient on its own. |
| evidence_packets | 0.85 | high | partially_supported | Central to avoiding overconfident or misleading conclusions. |
| hypotheses | 0.72 | medium | partially_supported | Important context but not sufficient on its own. |
| belief_records | 0.85 | high | partially_supported | Central to avoiding overconfident or misleading conclusions. |
| confidence | 0.72 | medium | partially_supported | Important context but not sufficient on its own. |
| limitations | 0.72 | medium | partially_supported | Important context but not sufficient on its own. |
| retrieval_trace | 0.85 | high | partially_supported | Central to avoiding overconfident or misleading conclusions. |
| active_state_filtering | 0.72 | medium | partially_supported | Important context but not sufficient on its own. |

## 4. Local Evidence Grounding

- Total evidence packets: 8
- Record matched packets: 0
- Keyword matched packets: 8
- Boundary matched packets: 0
- Fallback packets: 0
- Missing source files: 0

For CSV evidence, `Source Locator` shows the selected record scope and `Selected Values` shows values read from those records. For Markdown evidence, the locator remains a local line-range pointer.
`Decision Factor ID` is an internal RAC review identifier. The field column shows canonical project fields where available and labels unresolved requirements explicitly.

| Decision Factor ID | Source | Evidence Type | Status | Source Locator | Canonical Evidence Fields / Requirement | Selected Values |
|---|---|---|---|---|---|---|
| typed_memory | rac/README.md | default_evidence | keyword_matched | lines 1-2 | memory schema requirement | n/a |
| evidence_packets | rac/README.md | default_evidence | keyword_matched | lines 5-7 | source_path, claim_supported, limitations | n/a |
| hypotheses | rac/README.md | default_evidence | keyword_matched | lines 41-43 | hypothesis records | n/a |
| belief_records | rac/README.md | default_evidence | keyword_matched | lines 135-137 | belief update schema | n/a |
| confidence | rac/README.md | default_evidence | keyword_matched | lines 152-154 | confidence field | n/a |
| limitations | rac/README.md | default_evidence | keyword_matched | lines 7-9 | limitations field | n/a |
| retrieval_trace | rac/README.md | default_evidence | keyword_matched | lines 39-41 | source metadata | n/a |
| active_state_filtering | rac/README.md | default_evidence | keyword_matched | lines 7-9 | active flag, freshness policy | n/a |


## 5. Competing Hypotheses

The `Scenario-Template Confidence` column records deterministic review labels assigned by `generate_hypotheses(question_type)` in `rac/src/mock_pipeline.py`.

| Hypothesis | Scenario-Template Confidence | Status | Weakness |
|---|---:|---|---|
| RAC operates as a reasoning layer above the existing typed memory layer. | 0.86 | strong | The current RAC path resolves evidence from committed local project files. |
| The current deterministic implementation keeps evidence routing and review states inspectable. | 0.80 | strong | Fixed rules trade flexibility for inspectability. |

## 6. Critic Findings

- [high] Observational evidence supports bounded association claims only. Recommendation: Keep attribution language conditional and record unresolved alternatives.
- [medium] Current evidence scope is limited to committed local project files. Recommendation: Keep source paths and unresolved external requirements explicit.

## 7. Claim and Definition Check

- Status: pass
- Unsupported claims detected by current rules: none
- Definition conflicts detected by current rules: none

## 8. Final Judgment

RAC operates as a review layer above the existing typed memory system while leaving existing endpoints unchanged.

The judgment is bounded by the cited local evidence and the unresolved requirements recorded above.

## 9. Evidence-Routing Coverage

Packet composition:

- Total packets: 8
- Record matched packets: 0
- Keyword matched packets: 8
- Boundary matched packets: 0
- Fallback packets: 0
- Missing source files: 0

- Routing coverage score: 1.00
- Read this value as route resolution under the current rules, not as evidence strength or decision quality.

How this score is calculated:

```text
evidence_coverage_score =
  0.45 * direct_evidence_rate
+ 0.25 * supported_or_boundary_rate
+ 0.15 * no_missing_source_file_score
+ 0.15 * no_fallback_score
```

Weight rationale:

| Component | Weight | Why |
|---|---:|---|
| `direct_evidence_rate` | 0.45 | Highest priority because actual local evidence should matter more than boundary-only evidence. |
| `supported_or_boundary_rate` | 0.25 | Boundary evidence is valuable because it explicitly records missing requirements instead of hiding them. |
| `no_missing_source_file_score` | 0.15 | Source files must exist, but this is a basic traceability check rather than evidence strength. |
| `no_fallback_score` | 0.15 | Fallback packets indicate unresolved routing and reduce the current coverage score. |

Score contract:

- Component weights are fixed prototype heuristics.
- The score summarizes evidence-routing coverage under the current rules.
- Alternative weights are a formula sensitivity check; the report judgment is produced separately.

Score inputs (contract fields):

- total_packets = 8
- record_matched_packets = 0
- keyword_matched_packets = 8
- boundary_matched_packets = 0
- fallback_packets = 0
- missing_source_files = 0

Derived rates and checks:

- direct_evidence_rate = (record_matched_packets + keyword_matched_packets) / total_packets = 1.00
- supported_or_boundary_rate = (record_matched_packets + keyword_matched_packets + boundary_matched_packets) / total_packets = 1.00
- no_missing_source_file_score = 1.00
- no_fallback_score = 1.00

Reading the score:

- A higher value means more requested evidence routes were resolved or explicitly bounded.
- Boundary evidence contributes when it documents a missing requirement.
- Read the score as coverage rather than evidence strength, causal validity, decision quality, or business impact.

## 10. What Cannot Be Concluded

- The current RAC path is limited to committed local evidence.

## 11. Review-State Update

- review_state_id: rac_should_layer_above_existing_memory
- status: active
- validity_conditions:
  - Current project architecture stage.
