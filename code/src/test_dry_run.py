"""
Dry-run test: validates pipeline logic (CSV I/O, image loading,
prompt building, sanitization) without making real API calls.
"""

import sys
import json
sys.path.insert(0, str(__file__.replace("test_dry_run.py", "")))

from pathlib import Path
BASE = Path(__file__).parent.parent

# ── patch sys.path so we can import pipeline ──────────────────────
import importlib.util, types

# Stub out anthropic so import doesn't fail without the package
stub = types.ModuleType("anthropic")
class FakeClient:
    class messages:
        @staticmethod
        def create(**kw):
            raise NotImplementedError("stub")
stub.Anthropic = lambda: FakeClient()
stub.RateLimitError = Exception
sys.modules["anthropic"] = stub

spec = importlib.util.spec_from_file_location("pipeline", BASE / "src" / "pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)

# ── tests ─────────────────────────────────────────────────────────
def test_load_csvs():
    uh = pl.load_user_history(BASE / "dataset" / "user_history.csv")
    assert "U001" in uh, "U001 missing from user_history"
    er = pl.load_evidence_requirements(BASE / "dataset" / "evidence_requirements.csv")
    assert len(er) > 0, "evidence_requirements empty"
    claims = pl.load_csv(BASE / "dataset" / "claims.csv")
    assert len(claims) == 5, f"Expected 5 claims, got {len(claims)}"
    print("✓ CSV loading")

def test_parse_image_paths():
    raw = "images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg"
    paths = pl.parse_image_paths(raw)
    assert paths == ["images/test/case_001/img_1.jpg", "images/test/case_001/img_2.jpg"]
    assert pl.image_id_from_path("images/test/case_001/img_1.jpg") == "img_1"
    print("✓ Image path parsing")

def test_encode_image():
    path = "images/test/case_001/img_1.jpg"
    b64, media = pl.encode_image(path)
    assert media == "image/jpeg"
    assert len(b64) > 100
    import base64
    data = base64.b64decode(b64)
    assert data[:3] == b'\xff\xd8\xff', "Not a valid JPEG"
    print("✓ Image encoding")

def test_build_user_message():
    claims = pl.load_csv(BASE / "dataset" / "claims.csv")
    uh     = pl.load_user_history(BASE / "dataset" / "user_history.csv")
    er     = pl.load_evidence_requirements(BASE / "dataset" / "evidence_requirements.csv")
    claim  = claims[0]
    paths  = pl.parse_image_paths(claim["image_paths"])
    reqs   = pl.get_requirements_for(claim["claim_object"], er)
    content = pl.build_user_message(claim, uh.get(claim["user_id"]), reqs, paths)
    assert any(b["type"] == "text" for b in content)
    assert any(b["type"] == "image" for b in content)
    print(f"✓ Message building ({len(content)} blocks for {len(paths)} images)")

def test_sanitize():
    claims = pl.load_csv(BASE / "dataset" / "claims.csv")
    claim  = claims[0]
    paths  = pl.parse_image_paths(claim["image_paths"])

    raw = {
        "evidence_standard_met": True,
        "evidence_standard_met_reason": "Two clear images showing dent",
        "risk_flags": ["blurry_image", "INVALID_FLAG"],
        "issue_type": "dent",
        "object_part": "front_bumper",
        "claim_status": "supported",
        "claim_status_justification": "img_1 shows dent",
        "supporting_image_ids": ["img_1", "img_NONEXISTENT"],
        "valid_image": True,
        "severity": "medium",
    }
    s = pl.sanitize(raw, claim, paths)
    assert s["claim_status"] == "supported"
    assert s["issue_type"] == "dent"
    assert "INVALID_FLAG" not in s["risk_flags"]
    assert "img_NONEXISTENT" not in s["supporting_image_ids"]
    assert s["evidence_standard_met"] == "true"
    print("✓ Sanitization (invalid enum values dropped)")

def test_clamp():
    assert pl.clamp("supported", pl.CLAIM_STATUS_VALUES, "not_enough_information") == "supported"
    assert pl.clamp("GARBAGE",   pl.CLAIM_STATUS_VALUES, "not_enough_information") == "not_enough_information"
    print("✓ Enum clamping")

def test_get_requirements():
    er   = pl.load_evidence_requirements(BASE / "dataset" / "evidence_requirements.csv")
    reqs = pl.get_requirements_for("car", er)
    objs = {r["claim_object"] for r in reqs}
    assert "car" in objs or "all" in objs
    assert "package" not in objs
    print(f"✓ Evidence requirement filtering ({len(reqs)} rules for car)")

def test_output_structure():
    """Simulate a full row and verify all OUTPUT_COLUMNS present."""
    claim = {
        "user_id": "U001",
        "image_paths": "images/test/case_001/img_1.jpg",
        "user_claim": "My front bumper is dented.",
        "claim_object": "car",
    }
    sanitized = {
        "evidence_standard_met": "true",
        "evidence_standard_met_reason": "Clear image of bumper",
        "risk_flags": "none",
        "issue_type": "dent",
        "object_part": "front_bumper",
        "claim_status": "supported",
        "claim_status_justification": "img_1 shows dent on front bumper",
        "supporting_image_ids": "img_1",
        "valid_image": "true",
        "severity": "medium",
    }
    row = {"user_id": claim["user_id"], "image_paths": claim["image_paths"],
           "user_claim": claim["user_claim"], "claim_object": claim["claim_object"],
           **sanitized}
    for col in pl.OUTPUT_COLUMNS:
        assert col in row, f"Missing column: {col}"
    print("✓ Output row has all required columns")

if __name__ == "__main__":
    tests = [
        test_load_csvs,
        test_parse_image_paths,
        test_encode_image,
        test_build_user_message,
        test_sanitize,
        test_clamp,
        test_get_requirements,
        test_output_structure,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__}: {e}")
    print(f"\n{'='*40}")
    print(f"Passed {passed}/{len(tests)} tests")
    sys.exit(0 if passed == len(tests) else 1)
