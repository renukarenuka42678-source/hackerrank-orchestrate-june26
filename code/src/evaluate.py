"""
Evaluation script.
Compares evaluation/sample_output.csv against dataset/sample_claims.csv (ground truth).
Produces evaluation/evaluation_report.md
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
EVAL_DIR = BASE_DIR / "evaluation"

SCORED_FIELDS = [
    "evidence_standard_met",
    "claim_status",
    "issue_type",
    "object_part",
    "valid_image",
    "severity",
]

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def accuracy(preds, gts, field):
    correct = sum(1 for p, g in zip(preds, gts) if p.get(field, "").strip().lower() == g.get(field, "").strip().lower())
    return correct / len(preds) if preds else 0

def jaccard(pred_str, gt_str):
    """Jaccard similarity for semicolon-separated flag sets."""
    pred = set(x.strip() for x in pred_str.split(";") if x.strip())
    gt   = set(x.strip() for x in gt_str.split(";") if x.strip())
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    return len(pred & gt) / len(pred | gt)

def evaluate():
    gt_path   = DATASET_DIR / "sample_claims.csv"
    pred_path = EVAL_DIR / "sample_output.csv"

    if not pred_path.exists():
        print("sample_output.csv not found. Run pipeline.py first.")
        return

    gt    = load_csv(gt_path)
    preds = load_csv(pred_path)

    n = min(len(gt), len(preds))
    gt    = gt[:n]
    preds = preds[:n]

    results = {}
    for field in SCORED_FIELDS:
        results[field] = accuracy(preds, gt, field)

    # Risk flags Jaccard
    flag_scores = [jaccard(p.get("risk_flags",""), g.get("risk_flags","")) for p,g in zip(preds, gt)]
    results["risk_flags_jaccard"] = sum(flag_scores) / len(flag_scores) if flag_scores else 0

    # Supporting image IDs Jaccard
    sid_scores  = [jaccard(p.get("supporting_image_ids","none"), g.get("supporting_image_ids","none")) for p,g in zip(preds, gt)]
    results["supporting_image_ids_jaccard"] = sum(sid_scores) / len(sid_scores) if sid_scores else 0

    overall = sum(results.values()) / len(results)

    # Per-object breakdown
    by_object = defaultdict(lambda: defaultdict(list))
    for p, g in zip(preds, gt):
        obj = g.get("claim_object","unknown")
        for field in SCORED_FIELDS:
            match = int(p.get(field,"").strip().lower() == g.get(field,"").strip().lower())
            by_object[obj][field].append(match)

    report_lines = [
        "# Evaluation Report — Multi-Modal Evidence Review",
        "",
        f"Evaluated {n} sample claims against ground truth.",
        "",
        "## Per-Field Accuracy",
        "",
        "| Field | Accuracy |",
        "|---|---|",
    ]
    for field, acc in results.items():
        report_lines.append(f"| {field} | {acc:.2%} |")
    report_lines += ["", f"**Overall mean score: {overall:.2%}**", ""]

    report_lines += [
        "## Per-Object Breakdown",
        "",
        "| Object | Field | Accuracy |",
        "|---|---|---|",
    ]
    for obj, fields in sorted(by_object.items()):
        for field, vals in sorted(fields.items()):
            avg = sum(vals)/len(vals) if vals else 0
            report_lines.append(f"| {obj} | {field} | {avg:.2%} |")
    report_lines.append("")

    report_lines += [
        "## Operational Analysis",
        "",
        "### Model calls",
        "",
        "- **Sample set**: 1 API call per claim = N_sample calls",
        "- **Test set**: 1 API call per claim = N_test calls",
        "- Each call includes all images for that claim in a single multimodal request.",
        "- No redundant repeated calls; images are not cached externally but kept in-process.",
        "",
        "### Token usage estimates",
        "",
        "| Component | Estimate |",
        "|---|---|",
        "| System prompt | ~350 tokens (input) |",
        "| User context text per claim | ~200–400 tokens |",
        "| Per image (thumbnail) | ~1,000–2,000 tokens (vision) |",
        "| Output JSON | ~200–300 tokens |",
        "| **Per-claim total (2 images avg)** | **~3,000–5,000 tokens** |",
        "",
        "For 100 test claims: **~300,000–500,000 tokens**.",
        "",
        "### Cost estimate",
        "",
        "Using claude-sonnet-4-6 pricing (as of June 2026):",
        "",
        "| | Input | Output |",
        "|---|---|---|",
        "| Price per MTok | $3.00 | $15.00 |",
        "| 100 claims (500K input, 30K output) | $1.50 | $0.45 |",
        "| **Total estimate** | | **~$2–5 for full test set** |",
        "",
        "### Latency",
        "",
        "- ~3–8 seconds per claim (network + inference).",
        "- 100 claims: ~5–15 minutes sequential.",
        "- Parallelism (e.g. 5 workers): ~1–3 minutes.",
        "",
        "### TPM / RPM considerations",
        "",
        "- Throttle at 40 calls/minute (configurable via `CALLS_PER_MINUTE`).",
        "- Automatic sleep of 60s every 40 calls.",
        "- Exponential-backoff on `RateLimitError` with up to 3 retries.",
        "- Each claim is one request; no micro-batching needed (images inline).",
        "- Caching: system prompt is constant — Anthropic prompt caching could reduce input cost by ~90% on the system prompt portion if enabled.",
        "",
        "### Error handling",
        "",
        "- JSON parse failures retry up to 3 times.",
        "- API errors (5xx, timeout) also retry with delay.",
        "- A fallback row is written for any claim that still fails, flagged with `manual_review_required`.",
        "",
    ]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EVAL_DIR / "evaluation_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"✓ Evaluation report → {report_path}")
    print(f"  Overall mean accuracy: {overall:.2%}")
    for field, acc in results.items():
        print(f"  {field}: {acc:.2%}")


if __name__ == "__main__":
    evaluate()
