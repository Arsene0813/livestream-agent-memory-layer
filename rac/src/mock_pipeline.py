from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rac.src.state_validation import validate_cognition_state


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")[:80] or "question"


def join_sentence_fragments(
    items: list[str],
) -> str:
    """Join sentence fragments without producing '.;'."""
    fragments = [
        item.strip().rstrip(".")
        for item in items
        if item.strip()
    ]

    if not fragments:
        return ""

    return "; ".join(fragments) + "."


def analyze_question(question: str) -> dict[str, Any]:
    q = question.lower()

    if "attributed" in q or "caused" in q:
        question_type = "causal_diagnostic"
        domain = "retail_operations"
    elif "comparable" in q or "stores b-f" in q:
        question_type = "comparability_judgment"
        domain = "retail_operations"
    elif "promotion" in q or "checked before changing" in q:
        question_type = "strategic_recommendation"
        domain = "retail_operations"
    else:
        question_type = "technical_design"
        domain = "ai_system_design"

    return {
        "question_type": question_type,
        "domain": domain,
        "requires_evidence": True,
        "requires_internal_memory": True,
        "requires_fresh_external_information": False,
        "risk_level": "high" if question_type in {"causal_diagnostic", "comparability_judgment"} else "medium",
        "reason": "The question requires structured reasoning, evidence boundaries, and explicit limitations."
    }


FACTOR_LIBRARY: dict[str, list[dict[str, Any]]] = {
    "causal_diagnostic": [
        {"factor_id": "search_exposure", "name": "Search exposure", "description": "Search visibility may contribute to traffic but cannot prove attribution alone.", "evidence_needed": ["search exposure users", "search entry users", "search average rank"]},
        {"factor_id": "entry_conversion", "name": "Entry conversion", "description": "Places entry metrics alongside exposure metrics for review.", "evidence_needed": ["entry users", "exposure users", "entry conversion rate"]},
        {"factor_id": "order_conversion", "name": "Order conversion", "description": "Places order-conversion metrics alongside entry metrics for review.", "evidence_needed": ["order users", "entry users", "order conversion rate"]},
        {"factor_id": "promotion_intensity", "name": "Promotion intensity", "description": "Provides activity-order and activity-cost context for reviewing transaction outcomes.", "evidence_needed": ["activity orders", "activity cost", "activity original transaction amount"]},
        {
            "factor_id": "transaction_orders",
            "name": "Transaction orders",
            "description": (
                "Provides accepted and not-cancelled order-volume "
                "context without establishing attribution by itself."
            ),
            "evidence_needed": [
                "transaction_orders",
            ],
        },
    ],
    "comparability_judgment": [
        {"factor_id": "same_reporting_period", "name": "Same reporting period", "description": "Stores must first share the same reporting window.", "evidence_needed": ["period start", "period end"]},
        {"factor_id": "store_type", "name": "Store type", "description": "Different store types may not be directly comparable.", "evidence_needed": ["store type"]},
        {"factor_id": "order_volume", "name": "Order volume", "description": "Provides order-volume scale context for the comparison review.", "evidence_needed": ["transaction order count", "transaction orders"]},
        {"factor_id": "transaction_amount", "name": "Transaction amount", "description": "Gives scale context but is not sufficient alone.", "evidence_needed": ["transaction amount"]},
        {"factor_id": "activity_intensity", "name": "Activity involvement and intensity", "description": "Provides activity-involvement and cost-ratio context for the comparison review.", "evidence_needed": ["activity_orders", "activity_order_share_pct", "activity_cost", "activity_cost_ratio_pct"]},
        {"factor_id": "region_context", "name": "Region context", "description": "Provides weak regional background for the comparison review.", "evidence_needed": ["region type", "business district context"]},
        {"factor_id": "competition", "name": "Competition", "description": "Requires competitor price and order-trend context before stronger comparison claims.", "evidence_needed": ["competitor price", "competitor order trend"]},
        {"factor_id": "sku_structure", "name": "SKU structure", "description": "Provides product-mix context; margin effects are not observed in the current evidence.", "evidence_needed": ["top SKUs", "SKU transaction amount"]},
        {"factor_id": "repeated_reporting_windows", "name": "Repeated reporting windows", "description": "Three monthly B-F reporting windows are available for descriptive review, but they do not establish stable pairwise comparability.", "evidence_needed": ["multi-period data"]}
    ],
    "strategic_recommendation": [
        {"factor_id": "activity_orders", "name": "Activity orders", "description": "Records the backend activity-order count used in the promotion review.", "evidence_needed": ["activity order count"]},
        {"factor_id": "activity_cost", "name": "Activity cost", "description": "Records the backend activity-cost measure used in the promotion review.", "evidence_needed": ["activity cost"]},
        {"factor_id": "merchant_subsidy", "name": "Merchant subsidy", "description": "Records the merchant subsidy amount where available.", "evidence_needed": ["merchant subsidy amount"]},
        {"factor_id": "platform_subsidy", "name": "Platform subsidy", "description": "Records the platform subsidy amount where available.", "evidence_needed": ["platform subsidy amount"]},
        {"factor_id": "order_conversion", "name": "Order conversion", "description": "Includes the backend order-conversion metric in the promotion review.", "evidence_needed": ["order conversion rate"]},
        {"factor_id": "payment_conversion", "name": "Payment conversion", "description": "Includes the backend payment-conversion metric in the promotion review.", "evidence_needed": ["payment conversion rate"]},
        {"factor_id": "sku_margin_structure", "name": "SKU margin structure", "description": "Promotion decisions require margin context.", "evidence_needed": ["SKU margin", "SKU activity participation"]},
        {"factor_id": "competitor_context", "name": "Competitor context", "description": "Requires competitor pricing and order-trend context before stronger promotion conclusions.", "evidence_needed": ["competitor prices", "competitor order trend"]}
    ],
    "technical_design": [
        {"factor_id": "typed_memory", "name": "Typed memory", "description": "Preserve existing typed facts.", "evidence_needed": ["memory schema"]},
        {"factor_id": "evidence_packets", "name": "Evidence packets", "description": "Pass structured evidence rather than free-form context.", "evidence_needed": ["source path", "claim supported", "limitations"]},
        {"factor_id": "hypotheses", "name": "Hypotheses", "description": "Preserve competing explanations before final synthesis.", "evidence_needed": ["hypothesis records"]},
        {"factor_id": "belief_records", "name": "Belief records", "description": "Store conclusions with confidence and validity conditions.", "evidence_needed": ["belief update schema"]},
        {"factor_id": "confidence", "name": "Confidence", "description": "Expose uncertainty explicitly.", "evidence_needed": ["confidence field"]},
        {"factor_id": "limitations", "name": "Limitations", "description": "State what cannot be concluded.", "evidence_needed": ["limitations field"]},
        {"factor_id": "retrieval_trace", "name": "Retrieval trace", "description": "Make evidence traceable to sources.", "evidence_needed": ["source metadata"]},
        {"factor_id": "active_state_filtering", "name": "Active-state filtering", "description": "Avoid stale or deprecated memory records.", "evidence_needed": ["active flag", "freshness policy"]}
    ]
}


def expand_factors(question_type: str) -> list[dict[str, Any]]:
    return FACTOR_LIBRARY[question_type]


FACTOR_WEIGHT_BUCKETS: dict[str, set[str]] = {
    "high": {
        "promotion_intensity",
        "activity_intensity",
        "order_conversion",
        "sku_margin_structure",
        "evidence_packets",
        "belief_records",
        "retrieval_trace",
    },
    "medium": {
        "search_exposure",
        "entry_conversion",
        "same_reporting_period",
        "store_type",
        "order_volume",
        "transaction_amount",
        "transaction_orders",
        "payment_conversion",
        "typed_memory",
        "hypotheses",
        "confidence",
        "limitations",
        "active_state_filtering",
    },
}

FACTOR_WEIGHT_VALUES: dict[str, float] = {
    "high": 0.85,
    "medium": 0.72,
    "default": 0.60,
}

FACTOR_WEIGHT_REASONS: dict[str, str] = {
    "high": "Central to avoiding overconfident or misleading conclusions.",
    "medium": "Important context but not sufficient on its own.",
    "default": "Potentially relevant but requires stronger evidence.",
}

FACTOR_WEIGHTING_METHOD = (
    "fixed review-priority bucket assignment; "
    "source: rac/src/mock_pipeline.py; "
    "used to order review attention within the current evidence scope"
)


def classify_factor_weight(factor_id: str) -> dict[str, Any]:
    """Return the deterministic heuristic weight bucket for one factor."""

    if factor_id in FACTOR_WEIGHT_BUCKETS["high"]:
        bucket = "high"
    elif factor_id in FACTOR_WEIGHT_BUCKETS["medium"]:
        bucket = "medium"
    else:
        bucket = "default"

    return {
        "weight_bucket": bucket,
        "weight": FACTOR_WEIGHT_VALUES[bucket],
        "weight_reason": FACTOR_WEIGHT_REASONS[bucket],
        "weighting_method": FACTOR_WEIGHTING_METHOD,
        "weight_source": "rac/src/mock_pipeline.py",
    }


def build_factor_weighting_explanation(
    question_type: str,
    factor_weights: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a report-facing explanation of how factor weights were generated."""

    bucket_rows: dict[str, list[str]] = {
        "high": [],
        "medium": [],
        "default": [],
    }

    for row in factor_weights:
        bucket = row.get("weight_bucket", "default")
        bucket_rows.setdefault(bucket, []).append(row["factor_id"])

    return {
        "question_type": question_type,
        "method": FACTOR_WEIGHTING_METHOD,
        "source": "rac/src/mock_pipeline.py::classify_factor_weight",
        "bucket_values": FACTOR_WEIGHT_VALUES,
        "bucket_reasons": FACTOR_WEIGHT_REASONS,
        "bucket_members": bucket_rows,
        "limitations": [
            "Use the weights only to order review attention within the current RAC evidence scope.",
        ],
    }


def weight_factors(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []

    for factor in factors:
        fid = factor["factor_id"]
        weight_info = classify_factor_weight(fid)
        rows.append({
            "factor_id": fid,
            "weight": weight_info["weight"],
            "weight_bucket": weight_info["weight_bucket"],
            "weight_reason": weight_info["weight_reason"],
            "weighting_method": weight_info["weighting_method"],
            "weight_source": weight_info["weight_source"],
            "evidence_status": "partially_supported",
        })

    return rows


def route_evidence(question_type: str, factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if question_type == "causal_diagnostic":
        source_path = "retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md"
    elif question_type == "comparability_judgment":
        source_path = "retail_ops/data/demo2_source_notes.md"
    elif question_type == "strategic_recommendation":
        source_path = "retail_ops/data/DATA_DICTIONARY.md"
    else:
        source_path = "rac/README.md"

    packets = []
    for factor in factors:
        fid = factor["factor_id"]
        packets.append({
            "evidence_id": f"evidence_{fid}",
            "source_type": "markdown",
            "source_path": source_path,
            "claim_supported": f"Provides context needed to evaluate factor: {fid}.",
            "limitations": [
                "Evidence is resolved from committed local project files.",
                "Coverage is limited to the sources listed in this packet."
            ]
        })
    return packets


def generate_hypotheses(question_type: str) -> list[dict[str, Any]]:
    if question_type == "causal_diagnostic":
        return [
            {
                "hypothesis_id": "H1",
                "claim": "Search exposure is relevant to Store A's March-to-April increases in transaction amount and transaction orders, but it is not sufficient as a single explanation.",
                "confidence": 0.52,
                "supporting_factors": ["search_exposure", "entry_conversion"],
                "weaknesses": ["Does not isolate promotion effects.", "Does not prove source-field improvement."],
                "status": "plausible"
            },
            {
                "hypothesis_id": "H2",
                "claim": (
                    "Store A's March-to-April increases in transaction "
                    "amount and transaction orders should be reviewed alongside "
                    "search exposure, "
                    "entry conversion, order conversion, promotion "
                    "intensity, and transaction orders."
                ),
                "confidence": 0.74,
                "supporting_factors": [
                    "search_exposure",
                    "entry_conversion",
                    "order_conversion",
                    "promotion_intensity",
                    "transaction_orders",
                ],
                "weaknesses": ["Observational evidence cannot establish strict causality."],
                "status": "strong"
            },
            {
                "hypothesis_id": "H3",
                "claim": "The available evidence is insufficient for single-cause attribution.",
                "confidence": 0.82,
                "supporting_factors": [
                    "search_exposure",
                    "entry_conversion",
                    "order_conversion",
                    "promotion_intensity",
                    "transaction_orders",
                ],
                "weaknesses": ["Conservative rather than complete causal explanation."],
                "status": "strong"
            }
        ]

    if question_type == "comparability_judgment":
        return [
            {
                "hypothesis_id": "H1",
                "claim": "Stores B-F can be organized in a same-period diagnostic table.",
                "confidence": 0.78,
                "supporting_factors": ["same_reporting_period"],
                "weaknesses": ["Same-period diagnostic organization does not establish robust comparability."],
                "status": "strong"
            },
            {
                "hypothesis_id": "H2",
                "claim": "Stores B-F should not be treated as directly comparable without pairwise gates.",
                "confidence": 0.86,
                "supporting_factors": [
                    "same_reporting_period",
                    "store_type",
                    "order_volume",
                    "transaction_amount",
                    "activity_intensity",
                    "region_context",
                    "competition",
                    "sku_structure",
                    "repeated_reporting_windows",
                ],
                "weaknesses": [
                    "Pairwise quantitative thresholds are outside the current review contract."
                ],
                "status": "strong"
            }
        ]

    if question_type == "strategic_recommendation":
        return [
            {
                "hypothesis_id": "H1",
                "claim": (
                    "A bounded promotion review should cover activity cost, "
                    "merchant and platform subsidy, and order and payment "
                    "conversion."
                ),
                "confidence": 0.84,
                "supporting_factors": [
                    "activity_orders",
                    "activity_cost",
                    "merchant_subsidy",
                    "platform_subsidy",
                    "order_conversion",
                    "payment_conversion",
                ],
                "weaknesses": [
                    "The available evidence defines review dimensions but "
                    "does not establish a promotion outcome."
                ],
                "status": "strong"
            },
            {
                "hypothesis_id": "H2",
                "claim": (
                    "The current evidence can support a bounded "
                    "promotion review checklist, but not an automatic "
                    "promotion change."
                ),
                "confidence": 0.68,
                "supporting_factors": [
                    "activity_orders",
                    "activity_cost",
                    "merchant_subsidy",
                    "platform_subsidy",
                    "order_conversion",
                    "payment_conversion",
                ],
                "weaknesses": [
                    "Repeated-period cost evidence is required for "
                    "trend interpretation.",
                    "SKU margin and competitor context remain "
                    "unresolved for final action."
                ],
                "status": "plausible"
            }
        ]

    return [
        {
            "hypothesis_id": "H1",
            "claim": "RAC operates as a reasoning layer above the existing typed memory layer.",
            "confidence": 0.86,
            "supporting_factors": ["typed_memory", "evidence_packets", "hypotheses", "belief_records", "retrieval_trace", "active_state_filtering"],
            "weaknesses": [
                "The current RAC path resolves evidence from committed local project files."
            ],
            "status": "strong"
        },
        {
            "hypothesis_id": "H2",
            "claim": "The current deterministic implementation keeps evidence routing and review states inspectable.",
            "confidence": 0.80,
            "supporting_factors": ["confidence", "limitations", "retrieval_trace"],
            "weaknesses": [
                "Fixed rules trade flexibility for inspectability."
            ],
            "status": "strong"
        }
    ]


def critique(question_type: str) -> list[dict[str, str]]:
    findings = [
        {
            "issue": "Observational evidence supports bounded association claims only.",
            "severity": "high",
            "recommendation": "Keep attribution language conditional and record unresolved alternatives."
        },
        {
            "issue": "Current evidence scope is limited to committed local project files.",
            "severity": "medium",
            "recommendation": "Keep source paths and unresolved external requirements explicit."
        }
    ]

    if question_type == "comparability_judgment":
        findings.append({
            "issue": "Same-period diagnostic organization must not be described as a completed pairwise comparability gate.",
            "severity": "critical",
            "recommendation": "Separate same-period diagnostic review from pairwise comparability."
        })

    if question_type == "strategic_recommendation":
        findings.append({
            "issue": (
                "SKU margin and competitor context remain "
                "unresolved for final promotion action."
            ),
            "severity": "high",
            "recommendation": (
                "Keep the output at bounded "
                "review-checklist level."
            )
        })

    return findings


def fact_check(question_type: str, claims: list[str]) -> dict[str, Any]:
    unsupported_claims = []
    definition_conflicts = []

    banned = [
        "proves causality",
        "live Meituan backend access",
        "true Bayesian posterior",
        "updates neural network weights",
        "fully comparable"
    ]

    for claim in claims:
        low = claim.lower()
        for item in banned:
            if item.lower() in low:
                unsupported_claims.append(claim)

    if question_type == "strategic_recommendation":
        for claim in claims:
            if "roi" in claim.lower():
                definition_conflicts.append("Activity cost ratio should not be called ROI.")

    return {
        "status": "fail" if unsupported_claims or definition_conflicts else "pass",
        "unsupported_claims": unsupported_claims,
        "definition_conflicts": definition_conflicts
    }


def build_belief_update(question_type: str) -> dict[str, Any]:
    if question_type == "causal_diagnostic":
        return {
            "belief_id": "store_a_march_april_increase_not_search_only",
            "claim": "Store A's March-to-April increases in transaction amount and transaction orders should not be attributed to search exposure alone.",
            "confidence": 0.82,
            "status": "active",
            "validity_conditions": ["Store A Demo 1 context.", "Available month-over-month evidence only."],
            "limitations": [
                "No randomized experiment.",
                "Evidence scope is limited to committed local project files.",
                "No complete competitor-side evidence."
            ]
        }

    if question_type == "comparability_judgment":
        return {
            "belief_id": "stores_b_f_same_period_not_directly_comparable",
            "claim": "Stores B-F can be organized for same-period diagnostic review, but should not be treated as directly comparable without pairwise gates.",
            "confidence": 0.86,
            "status": "active",
            "validity_conditions": ["Demo 2 March 2026 B-F context."],
            "limitations": [
                "Pairwise quantitative gates are not defined in the current contract.",
                "Region type remains weak context.",
                "Three monthly B-F reporting windows are available; they do not by themselves establish stable pairwise comparability."
            ]
        }

    if question_type == "strategic_recommendation":
        return {
            "belief_id": "promotion_changes_require_multi_factor_check",
            "claim": (
                "The current evidence supports a bounded "
                "promotion-review checklist, not an automatic "
                "promotion change."
            ),
            "confidence": 0.80,
            "status": "active",
            "validity_conditions": ["Retail operations decision-support questions."],
            "limitations": [
                "Margin fields are absent from the current evidence.",
                "Competitor data may be incomplete.",
                "One reporting window is insufficient for robust action attribution."
            ]
        }

    return {
        "belief_id": "rac_should_layer_above_existing_memory",
        "claim": "RAC operates as a review layer above the existing typed memory system while leaving existing endpoints unchanged.",
        "confidence": 0.84,
        "status": "active",
        "validity_conditions": ["Current project architecture stage."],
        "limitations": [
            "The current RAC path is limited to committed local evidence."
        ]
    }


def write_final_report(state: dict[str, Any]) -> str:
    factor_by_id = {factor["factor_id"]: factor for factor in state["factors"]}
    lines = []

    lines.append("# Answer")
    lines.append("")
    lines.append("## 1. Direct Answer")
    lines.append("")
    lines.append(state["belief_update"]["claim"])
    lines.append("")
    lines.append("This is a deterministic mock result. It confirms that the current fixed fixture can generate the expected artifacts end-to-end, but it does not claim live retrieval or autonomous world modeling.")
    lines.append("")
    lines.append("## 2. Question Type")
    lines.append("")
    lines.append(f"- Question type: {state['question_type']}")
    lines.append(f"- Domain: {state['domain']}")
    lines.append("")
    lines.append("## 3. Relevant Factors Considered")
    lines.append("")
    lines.append("| Factor | Weight | Evidence Status | Why It Matters |")
    lines.append("|---|---:|---|---|")

    for row in state["factor_weights"]:
        factor = factor_by_id[row["factor_id"]]
        lines.append(f"| {factor['factor_id']} | {row['weight']:.2f} | {row['evidence_status']} | {row['weight_reason']} |")

    lines.append("")
    lines.append("## 4. Evidence Used")
    lines.append("")
    lines.append("| Evidence | Source | Supports | Limitations |")
    lines.append("|---|---|---|---|")

    for evidence in state["evidence_packets"]:
        limitation_text = join_sentence_fragments(
            evidence["limitations"]
        )
        lines.append(
            f"| {evidence['evidence_id']} | "
            f"{evidence['source_path']} | "
            f"{evidence['claim_supported']} | "
            f"{limitation_text} |"
        )

    lines.append("")
    lines.append("## 5. Competing Hypotheses")
    lines.append("")
    lines.append(
        "Hypothesis confidence values are deterministic scenario-template values "
        "assigned by `generate_hypotheses(question_type)` in `rac/src/mock_pipeline.py`. "
        "They are not learned probabilities, calibrated likelihoods, or direct calculations "
        "from observed Meituan metric tables."
    )
    lines.append("")
    lines.append("| Hypothesis | Confidence | Status | Weakness |")
    lines.append("|---|---:|---|---|")

    for h in state["hypotheses"]:
        weakness_text = join_sentence_fragments(
            h["weaknesses"]
        )
        lines.append(
            f"| {h['claim']} | {h['confidence']:.2f} | "
            f"{h['status']} | {weakness_text} |"
        )

    lines.append("")
    lines.append("## 6. Critic Findings")
    lines.append("")

    for finding in state["critic_findings"]:
        lines.append(f"- [{finding['severity']}] {finding['issue']} Recommendation: {finding['recommendation']}")

    lines.append("")
    lines.append("## 7. Final Judgment")
    lines.append("")
    lines.append(state["belief_update"]["claim"])
    lines.append("")
    lines.append("The conclusion is conservative because this mock pipeline uses structured placeholder evidence and does not perform live retrieval.")
    lines.append("")
    lines.append("## 8. Scenario-Template Confidence")
    lines.append("")
    lines.append(f"{state['belief_update']['confidence']:.2f}")
    lines.append("")
    lines.append("How this value is assigned:")
    lines.append("")
    lines.append("- Source: `build_belief_update(question_type)` in `rac/src/mock_pipeline.py`.")
    lines.append("- Rule: deterministic case-template assignment by question type.")
    lines.append("- It is not calculated from evidence-packet counts, factor weights, or observed metric tables.")
    lines.append("- It is not learned from historical data.")
    lines.append("- It is not a calibrated probability or Bayesian posterior.")
    lines.append("- It is not a causal confidence score or business-success probability.")
    lines.append("- It is kept only to show how the deterministic mock scaffold carries a review-state value.")
    lines.append("- Grounded reports use a formula-based `Evidence-Coverage Score` instead.")
    lines.append("")
    lines.append("## 9. What Cannot Be Concluded")
    lines.append("")

    for limitation in state["belief_update"]["limitations"]:
        lines.append(f"- {limitation}")

    lines.append("")
    lines.append("## 10. Review-State Update")
    lines.append("")
    lines.append(f"- review_state_id: {state['belief_update']['belief_id']}")
    lines.append(f"- status: {state['belief_update']['status']}")
    lines.append("- validity_conditions:")

    for condition in state["belief_update"]["validity_conditions"]:
        lines.append(f"  - {condition}")

    lines.append("")
    return "\n".join(lines)


def run_mock_pipeline(question: str) -> dict[str, Any]:
    analysis = analyze_question(question)
    factors = expand_factors(analysis["question_type"])
    factor_weights = weight_factors(factors)
    evidence_packets = route_evidence(analysis["question_type"], factors)
    hypotheses = generate_hypotheses(analysis["question_type"])
    critic_findings = critique(analysis["question_type"])

    fact_check_result = fact_check(analysis["question_type"], [h["claim"] for h in hypotheses])
    belief_update = build_belief_update(analysis["question_type"])

    state = {
        "question": question,
        "question_type": analysis["question_type"],
        "domain": analysis["domain"],
        "factors": factors,
        "factor_weights": factor_weights,
        "evidence_packets": evidence_packets,
        "hypotheses": hypotheses,
        "critic_findings": critic_findings,
        "fact_check": fact_check_result,
        "belief_update": belief_update,
        "final_report": ""
    }

    state["final_report"] = write_final_report(state)
    validate_cognition_state(
        state,
        root=Path(__file__).resolve().parents[2],
    )
    return state


def save_state_outputs(state: dict[str, Any], output_dir: Path, name: str | None = None) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = name or slugify(state["question"])

    json_path = output_dir / f"{slug}.json"
    md_path = output_dir / f"{slug}.md"

    json_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(state["final_report"], encoding="utf-8")

    return {"json": str(json_path), "markdown": str(md_path)}
