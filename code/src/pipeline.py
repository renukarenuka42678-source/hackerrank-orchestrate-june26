"""
Multi-Modal Evidence Review Pipeline
Processes damage claims using images, conversation, and user history
via the Anthropic API (claude-sonnet-4-6).
"""

import anthropic
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1000
BASE_DIR = Path(__file__).parent.parent  # project root
DATASET_DIR = BASE_DIR / "dataset"
IMAGE_BASE_DIR = BASE_DIR  # image paths in CSV are relative to project root

# Rate-limit / retry config
MAX_RETRIES = 3
RETRY_DELAY = 5          # seconds between retries
CALLS_PER_MINUTE = 40    # conservative; tune for your tier

# Allowed enum values (mirrors the spec)
CLAIM_STATUS_VALUES    = {"supported", "contradicted", "not_enough_information"}
ISSUE_TYPE_VALUES      = {
    "dent","scratch","crack","glass_shatter","broken_part","missing_part",
    "torn_packaging","crushed_packaging","water_damage","stain","none","unknown"
}
SEVERITY_VALUES        = {"none","low","medium","high","unknown"}
RISK_FLAG_VALUES       = {
    "none","blurry_image","cropped_or_obstructed","low_light_or_glare",
    "wrong_angle","wrong_object","wrong_object_part","damage_not_visible",
    "claim_mismatch","possible_manipulation","non_original_image",
    "text_instruction_present","user_history_risk","manual_review_required"
}

CAR_PARTS = {
    "front_bumper","rear_bumper","door","hood","windshield","side_mirror",
    "headlight","taillight","fender","quarter_panel","body","unknown"
}
LAPTOP_PARTS = {
    "screen","keyboard","trackpad","hinge","lid","corner","port","base","body","unknown"
}
PACKAGE_PARTS = {
    "box","package_corner","package_side","seal","label","contents","item","unknown"
}
PART_MAP = {"car": CAR_PARTS, "laptop": LAPTOP_PARTS, "package": PACKAGE_PARTS}

OUTPUT_COLUMNS = [
    "user_id","image_paths","user_claim","claim_object",
    "evidence_standard_met","evidence_standard_met_reason",
    "risk_flags","issue_type","object_part","claim_status",
    "claim_status_justification","supporting_image_ids","valid_image","severity"
]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
client = anthropic.Anthropic()   # picks up ANTHROPIC_API_KEY from env

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

def encode_image(image_path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for a local image file."""
    full = IMAGE_BASE_DIR / image_path
    ext = Path(image_path).suffix.lower().lstrip(".")
    media = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png",  "gif": "image/gif",
        "webp": "image/webp"
    }.get(ext, "image/jpeg")
    with open(full, "rb") as f:
        return base64.b64encode(f.read()).decode(), media

def parse_image_paths(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(";") if p.strip()]

def image_id_from_path(path: str) -> str:
    return Path(path).stem

def clamp(value: str, allowed: set, default: str) -> str:
    return value if value in allowed else default

def clamp_list(values: list[str], allowed: set) -> list[str]:
    return [v for v in values if v in allowed] or ["none"]

# ─────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────
def load_user_history(path: Path) -> dict[str, dict]:
    rows = load_csv(path)
    return {r["user_id"]: r for r in rows}

def load_evidence_requirements(path: Path) -> list[dict]:
    return load_csv(path)

def get_requirements_for(claim_object: str, evidence_reqs: list[dict]) -> list[dict]:
    return [
        r for r in evidence_reqs
        if r["claim_object"] in (claim_object, "all")
    ]

# ─────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a damage-claim review AI.
You will receive:
- A user's damage claim conversation
- One or more images of the object
- The claim object type (car, laptop, package)
- Evidence requirements for this claim type
- The user's claim history summary

Your job is to evaluate whether the submitted images support, contradict, or provide insufficient information about the claim.

You MUST respond with a single valid JSON object and nothing else. No markdown, no preamble.
The JSON must have exactly these keys:
{
  "evidence_standard_met": true | false,
  "evidence_standard_met_reason": "<string>",
  "risk_flags": ["<flag>", ...],
  "issue_type": "<issue_type>",
  "object_part": "<object_part>",
  "claim_status": "supported" | "contradicted" | "not_enough_information",
  "claim_status_justification": "<string>",
  "supporting_image_ids": ["<id>", ...],
  "valid_image": true | false,
  "severity": "none" | "low" | "medium" | "high" | "unknown"
}

Rules:
- Images are the PRIMARY source of truth.
- User history can only increase risk_flags, never override clear visual evidence.
- supporting_image_ids: list the image IDs (filename without extension) that support your decision.
  Use ["none"] if no image is sufficient.
- risk_flags: use ["none"] when there are no flags.
- Be concise and image-grounded in justifications.
"""

def build_user_message(
    claim: dict,
    user_hist: dict | None,
    evidence_reqs: list[dict],
    image_paths: list[str],
) -> list[dict]:
    """Build the multipart content list for the API call."""

    content: list[dict] = []

    # Text block: context
    req_text = "\n".join(
        f"- [{r['requirement_id']}] {r['applies_to']}: {r['minimum_image_evidence']}"
        for r in evidence_reqs
    )
    hist_text = (
        f"Past claims: {user_hist['past_claim_count']}, "
        f"Accepted: {user_hist['accept_claim']}, "
        f"Rejected: {user_hist['rejected_claim']}, "
        f"Manual review: {user_hist['manual_review_claim']}, "
        f"Last 90 days: {user_hist['last_90_days_claim_count']}, "
        f"Flags: {user_hist['history_flags']}, "
        f"Summary: {user_hist['history_summary']}"
    ) if user_hist else "No history available."

    context_text = f"""CLAIM OBJECT: {claim['claim_object']}
CLAIM CONVERSATION:
{claim['user_claim']}

EVIDENCE REQUIREMENTS:
{req_text}

USER HISTORY:
{hist_text}

IMAGE IDs in submission order: {", ".join(image_id_from_path(p) for p in image_paths)}

Now evaluate the images below and respond with JSON only."""

    content.append({"type": "text", "text": context_text})

    # Image blocks
    for path in image_paths:
        try:
            b64, media_type = encode_image(path)
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64}
            })
        except FileNotFoundError:
            content.append({
                "type": "text",
                "text": f"[Image not found: {path}]"
            })

    return content

# ─────────────────────────────────────────────
# API call with retry
# ─────────────────────────────────────────────
def call_claude(content: list[dict]) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}]
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON parse error on attempt {attempt}: {e}", file=sys.stderr)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY)
        except anthropic.RateLimitError:
            print(f"  [WARN] Rate limit hit, sleeping {RETRY_DELAY*2}s (attempt {attempt})", file=sys.stderr)
            time.sleep(RETRY_DELAY * 2)
        except Exception as e:
            print(f"  [WARN] API error on attempt {attempt}: {e}", file=sys.stderr)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY)

# ─────────────────────────────────────────────
# Result sanitizer
# ─────────────────────────────────────────────
def sanitize(result: dict, claim: dict, image_paths: list[str]) -> dict:
    """Enforce allowed values and coerce types."""
    obj = claim["claim_object"]
    valid_parts = PART_MAP.get(obj, set()) | {"unknown"}

    risk_flags_raw = result.get("risk_flags", ["none"])
    if isinstance(risk_flags_raw, str):
        risk_flags_raw = [r.strip() for r in risk_flags_raw.split(";")]
    risk_flags = clamp_list(risk_flags_raw, RISK_FLAG_VALUES)

    sup_ids_raw = result.get("supporting_image_ids", ["none"])
    if isinstance(sup_ids_raw, str):
        sup_ids_raw = [s.strip() for s in sup_ids_raw.split(";")]
    # keep only valid IDs or "none"
    valid_ids = {image_id_from_path(p) for p in image_paths} | {"none"}
    sup_ids = [i for i in sup_ids_raw if i in valid_ids] or ["none"]

    return {
        "evidence_standard_met":        str(result.get("evidence_standard_met", False)).lower(),
        "evidence_standard_met_reason": result.get("evidence_standard_met_reason", ""),
        "risk_flags":                   ";".join(risk_flags),
        "issue_type":                   clamp(result.get("issue_type", "unknown"), ISSUE_TYPE_VALUES, "unknown"),
        "object_part":                  clamp(result.get("object_part", "unknown"), valid_parts, "unknown"),
        "claim_status":                 clamp(result.get("claim_status", "not_enough_information"), CLAIM_STATUS_VALUES, "not_enough_information"),
        "claim_status_justification":   result.get("claim_status_justification", ""),
        "supporting_image_ids":         ";".join(sup_ids),
        "valid_image":                  str(result.get("valid_image", False)).lower(),
        "severity":                     clamp(result.get("severity", "unknown"), SEVERITY_VALUES, "unknown"),
    }

# ─────────────────────────────────────────────
# Process a single claim
# ─────────────────────────────────────────────
def process_claim(
    claim: dict,
    user_history_map: dict,
    evidence_reqs: list[dict],
    call_counter: list[int],
) -> dict:
    uid = claim["user_id"]
    image_paths = parse_image_paths(claim["image_paths"])
    user_hist = user_history_map.get(uid)
    reqs = get_requirements_for(claim["claim_object"], evidence_reqs)

    content = build_user_message(claim, user_hist, reqs, image_paths)

    print(f"  Calling API for user={uid}, images={len(image_paths)}", file=sys.stderr)
    raw_result = call_claude(content)
    call_counter[0] += 1

    sanitized = sanitize(raw_result, claim, image_paths)

    # Rate-limit throttle (simple token-bucket approximation)
    if call_counter[0] % CALLS_PER_MINUTE == 0:
        print(f"  [THROTTLE] {call_counter[0]} calls made, sleeping 60s to respect RPM", file=sys.stderr)
        time.sleep(60)

    return {
        "user_id":      uid,
        "image_paths":  claim["image_paths"],
        "user_claim":   claim["user_claim"],
        "claim_object": claim["claim_object"],
        **sanitized,
    }

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run(input_csv: str, output_csv: str, label: str = "test"):
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Processing {label} set: {input_csv}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    claims       = load_csv(Path(input_csv))
    user_hist    = load_user_history(DATASET_DIR / "user_history.csv")
    evidence_req = load_evidence_requirements(DATASET_DIR / "evidence_requirements.csv")

    results      = []
    call_counter = [0]
    errors       = []

    for i, claim in enumerate(claims, 1):
        print(f"\n[{i}/{len(claims)}] claim_object={claim['claim_object']} user={claim['user_id']}", file=sys.stderr)
        try:
            row = process_claim(claim, user_hist, evidence_req, call_counter)
        except Exception as e:
            print(f"  [ERROR] {e}", file=sys.stderr)
            errors.append((i, str(e)))
            # Fallback row so output always has same count
            image_paths = parse_image_paths(claim.get("image_paths", ""))
            row = {
                "user_id":                      claim.get("user_id", ""),
                "image_paths":                  claim.get("image_paths", ""),
                "user_claim":                   claim.get("user_claim", ""),
                "claim_object":                 claim.get("claim_object", ""),
                "evidence_standard_met":        "false",
                "evidence_standard_met_reason": "Processing error",
                "risk_flags":                   "manual_review_required",
                "issue_type":                   "unknown",
                "object_part":                  "unknown",
                "claim_status":                 "not_enough_information",
                "claim_status_justification":   f"Error during processing: {e}",
                "supporting_image_ids":         "none",
                "valid_image":                  "false",
                "severity":                     "unknown",
            }
        results.append(row)

    save_csv(results, Path(output_csv))
    print(f"\n✓ Saved {len(results)} rows → {output_csv}", file=sys.stderr)
    print(f"  Total API calls: {call_counter[0]}", file=sys.stderr)
    if errors:
        print(f"  Errors: {errors}", file=sys.stderr)

    return call_counter[0], len(results)


if __name__ == "__main__":
    # 1) Evaluate on sample set
    sample_calls, sample_n = run(
        str(DATASET_DIR / "sample_claims.csv"),
        str(BASE_DIR / "evaluation" / "sample_output.csv"),
        label="sample"
    )

    # 2) Run on test set → final output.csv
    test_calls, test_n = run(
        str(DATASET_DIR / "claims.csv"),
        str(BASE_DIR / "output.csv"),
        label="test"
    )

    print(f"\nDone. Sample calls={sample_calls}, Test calls={test_calls}")
