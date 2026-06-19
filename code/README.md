# Multi-Modal Evidence Review — Solution

## Overview

This system evaluates insurance damage claims by combining:
- **Visual evidence** (one or more images) — primary source of truth
- **Claim conversation** — defines what to check
- **User history** — adds risk context only
- **Evidence requirements** — minimum checklist per object × issue family

It calls `claude-sonnet-4-6` with a structured JSON prompt and enforces allowed enum values on every output field.

---

## Repo structure

```
multimodal-evidence-review/
├── dataset/
│   ├── sample_claims.csv          # labeled examples (ground truth)
│   ├── claims.csv                 # test input (no labels)
│   ├── user_history.csv
│   ├── evidence_requirements.csv
│   └── images/
│       ├── sample/                # images for sample claims
│       └── test/                  # images for test claims
├── src/
│   ├── pipeline.py                # main processing pipeline
│   └── evaluate.py                # evaluation + report generator
├── evaluation/
│   ├── sample_output.csv          # predictions on sample set
│   └── evaluation_report.md       # accuracy + operational analysis
├── output.csv                     # final predictions on claims.csv
└── README.md
```

---

## How to run

### Prerequisites

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-..."
```

### Full run (sample eval + test output)

```bash
python src/pipeline.py
```

This will:
1. Process `dataset/sample_claims.csv` → `evaluation/sample_output.csv`
2. Process `dataset/claims.csv` → `output.csv`

### Evaluation only (after pipeline run)

```bash
python src/evaluate.py
```

Produces `evaluation/evaluation_report.md` with per-field accuracy scores.

---

## Design decisions

### One call per claim
All images for a claim are sent in a single multimodal API call. This minimises latency and cost compared to separate image-analysis calls.

### Structured JSON output
The model is instructed to return pure JSON. A post-processing sanitiser enforces:
- All enum fields (claim_status, issue_type, severity, etc.)
- Semicolon-separated list fields (risk_flags, supporting_image_ids)
- Boolean fields as lowercase strings

### Risk flag escalation
User history is injected as text context. The model may add `user_history_risk` or `manual_review_required` flags based on past rejection / manual review rates. History alone never overrides visual evidence.

### Retry & rate-limit strategy
- 3 retries on JSON parse error or API error
- Backoff: 5 s normal, 10 s on RateLimitError
- Throttle: sleep 60 s every 40 calls (configurable via `CALLS_PER_MINUTE`)

### Fallback rows
If a claim still fails after all retries, a fallback row is written with `claim_status=not_enough_information` and `manual_review_required` flag so the CSV always has the same row count.

---

## Output columns

| Column | Description |
|---|---|
| user_id | User who submitted the claim |
| image_paths | Raw paths from input |
| user_claim | Raw transcript from input |
| claim_object | car / laptop / package |
| evidence_standard_met | true / false |
| evidence_standard_met_reason | Short reason |
| risk_flags | Semicolon-separated flags |
| issue_type | Detected issue |
| object_part | Relevant part |
| claim_status | supported / contradicted / not_enough_information |
| claim_status_justification | Image-grounded explanation |
| supporting_image_ids | Semicolon-separated image IDs |
| valid_image | true / false |
| severity | none / low / medium / high / unknown |
