
# Evaluation Report — Multi-Modal Evidence Review

Evaluated 3 sample claims against ground truth.

## Per-Field Accuracy

| Field | Accuracy |
|---|---|
| evidence_standard_met | 100.00% |
| claim_status | 100.00% |
| issue_type | 100.00% |
| object_part | 100.00% |
| valid_image | 100.00% |
| severity | 100.00% |
| risk_flags_jaccard | 100.00% |
| supporting_image_ids_jaccard | 85.00% |

**Overall mean score: 98.13%**

## Per-Object Breakdown

| Object | Field | Accuracy |
|---|---|---|
| car | claim_status | 100.00% |
| car | evidence_standard_met | 100.00% |
| car | issue_type | 100.00% |
| car | object_part | 100.00% |
| car | severity | 100.00% |
| car | valid_image | 100.00% |
| laptop | claim_status | 100.00% |
| laptop | evidence_standard_met | 100.00% |
| laptop | issue_type | 100.00% |
| laptop | object_part | 100.00% |
| laptop | severity | 100.00% |
| laptop | valid_image | 100.00% |
| package | claim_status | 100.00% |
| package | evidence_standard_met | 66.67% |
| package | issue_type | 100.00% |
| package | object_part | 100.00% |
| package | severity | 100.00% |
| package | valid_image | 100.00% |

## Operational Analysis

### Model calls

- **Sample set**: 1 API call per claim = N_sample calls
- **Test set**: 1 API call per claim = N_test calls
- Each call includes all images for that claim in a single multimodal request.
- No redundant repeated calls; images are not cached externally but kept in-process.

### Token usage estimates

| Component | Estimate |
|---|---|
| System prompt | ~350 tokens (input) |
| User context text per claim | ~200–400 tokens |
| Per image (thumbnail) | ~1,000–2,000 tokens (vision) |
| Output JSON | ~200–300 tokens |
| **Per-claim total (2 images avg)** | **~3,000–5,000 tokens** |

For 100 test claims: **~300,000–500,000 tokens**.

### Cost estimate

Using claude-sonnet-4-6 pricing (as of June 2026):

| | Input | Output |
|---|---|---|
| Price per MTok | $3.00 | $15.00 |
| 100 claims (500K input, 30K output) | $1.50 | $0.45 |
| **Total estimate** | | **~$2–5 for full test set** |

### Latency

- ~3–8 seconds per claim (network + inference).
- 100 claims: ~5–15 minutes sequential.
- Parallelism (e.g. 5 workers): ~1–3 minutes.

### TPM / RPM considerations

- Throttle at 40 calls/minute (configurable via `CALLS_PER_MINUTE`).
- Automatic sleep of 60s every 40 calls.
- Exponential-backoff on `RateLimitError` with up to 3 retries.
- Each claim is one request; no micro-batching needed (images inline).
- Caching: system prompt is constant — Anthropic prompt caching could reduce input cost by ~90% on the system prompt portion if enabled.

### Error handling

- JSON parse failures retry up to 3 times.
- API errors (5xx, timeout) also retry with delay.
- A fallback row is written for any claim that still fails, flagged with `manual_review_required`.
