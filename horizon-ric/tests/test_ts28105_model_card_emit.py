"""3GPP TS 28.105 — AI/ML model description card emission test.

For every shipped checkpoint .pt, there must be a sibling .md file
("model card") that records the TS 28.105 mandatory identification
metadata: name, version, training_data, evaluation_metrics,
parameter_count, file_size, sha256.

The recorded sha256 must match the actual sha256 of the .pt artefact —
this is what makes the card auditable instead of decorative.

Cards without a .pt sibling document procedures (e.g. the TD-MPC planner
is parameter-free) and are skipped.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "checkpoints"


def _list_card_pt_pairs() -> list[tuple[Path, Path]]:
    pairs = []
    for md in sorted(CKPT.glob("*.md")):
        pt = md.with_suffix(".pt")
        if pt.is_file():
            pairs.append((md, pt))
    assert pairs, f"no checkpoint cards found under {CKPT}"
    return pairs


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Patterns we accept for each TS 28.105 mandatory field. Cards in this
# repo use a mix of styles ("- **SHA-256:**", "- **sha256:**",
# "Training Data", "Dataset", "## Metrics", "## Performance" …) so we
# match the union.
_FIELD_REGEX: dict[str, re.Pattern[str]] = {
    "name": re.compile(
        r"\*\*(name|model name)[:]?\*\*\s*[:=]?\s*[`\"]?([^\n`\"]+)", re.I
    ),
    "version": re.compile(
        r"\*\*version[:]?\*\*\s*[:=]?\s*[`\"]?(v?\d[\w\.\-]*)", re.I
    ),
    "training_data": re.compile(
        r"(##+\s*(training data|training run|training dataset|dataset|data)\b"
        r"|^Dataset:|^Trained:|\*\*training[_\s]?data\*\*"
        r"|\*\*train(ing)? samples\*\*|\*\*source\*\*)",
        re.I | re.M,
    ),
    "evaluation_metrics": re.compile(
        r"(##+\s*(metrics|performance|evaluation|results)\b"
        r"|val[_\s](loss|mse|r\^?2|acc)|final\s+(train\s+)?(mse|loss)"
        r"|\*\*ece\*\*|\bbrier\b|\bMSE\b|\bECE\b)",
        re.I | re.M,
    ),
    "parameter_count": re.compile(
        r"\*\*(parameter[_\s]?count|parameters?)[:]?\*\*\s*[:=]?\s*([\d,]+)", re.I
    ),
    "file_size": re.compile(
        r"\*\*(file[_\s]?size(\s*\(bytes\))?)[:]?\*\*\s*[:=]?\s*([\d,]+)", re.I
    ),
    "sha256": re.compile(
        r"\*\*sha-?256[:]?\*\*\s*[:=]?\s*[`\"]?([0-9a-fA-F]{64})", re.I
    ),
    # Devil-A Finding #3 closure: TS 28.105 §7.4 mandatory fields.
    "inferenceType": re.compile(
        r"\*\*inferenceType[:]?\*\*\s*[:=]?\s*[`\"]?"
        r"(real-time|batch|stream|near-real-time)",
        re.I,
    ),
    "expectedRunTimeContext": re.compile(
        r"\*\*expectedRunTimeContext[:]?\*\*\s*[:=]?\s*[`\"]?[^\n]+",
        re.I,
    ),
    "lastTrainingDataset": re.compile(
        r"\*\*lastTrainingDataset[:]?\*\*\s*[:=]?\s*[^\n]*sha256[^\n]*"
        r"[`\"]?([0-9a-fA-F]{64})",
        re.I,
    ),
    "evaluationReport": re.compile(
        r"\*\*evaluationReport[:]?\*\*\s*[:=]?\s*[^\n]*sha256[^\n]*"
        r"[`\"]?([0-9a-fA-F]{64})",
        re.I,
    ),
}


# The four TS 28.105 §7.4 spec-mandatory fields (Finding #3).
_TS28105_74_REQUIRED_FIELDS = (
    "inferenceType",
    "expectedRunTimeContext",
    "lastTrainingDataset",
    "evaluationReport",
)


@pytest.mark.parametrize("md,pt", _list_card_pt_pairs(), ids=lambda x: x.name)
def test_card_has_all_ts28105_mandatory_fields(md: Path, pt: Path) -> None:
    text = md.read_text()
    missing = [k for k, rx in _FIELD_REGEX.items() if not rx.search(text)]
    assert not missing, (
        f"{md.name}: missing TS 28.105 mandatory field(s) {missing}. "
        f"Required by 3GPP TS 28.105 §7 (Model Description Card)."
    )


_CANONICAL_BEGIN = "<!-- TS28105-IDENTIFICATION-BEGIN -->"
_CANONICAL_END = "<!-- TS28105-IDENTIFICATION-END -->"


def _canonical_block(text: str) -> str:
    """Extract the canonical TS 28.105 identification block when present;
    cards may also include legacy free-form sha/size lines from earlier
    revisions, but the canonical block is the one we audit against.
    """
    if _CANONICAL_BEGIN in text and _CANONICAL_END in text:
        a = text.index(_CANONICAL_BEGIN)
        b = text.index(_CANONICAL_END) + len(_CANONICAL_END)
        return text[a:b]
    return text


@pytest.mark.parametrize("md,pt", _list_card_pt_pairs(), ids=lambda x: x.name)
def test_card_sha256_matches_artifact(md: Path, pt: Path) -> None:
    block = _canonical_block(md.read_text())
    m = _FIELD_REGEX["sha256"].search(block)
    assert m is not None, f"{md.name}: no sha256 in canonical block"
    declared = m.group(1).lower()
    actual = _sha256_file(pt)
    assert declared == actual, (
        f"{md.name}: declared sha256 {declared} != actual {actual} of "
        f"{pt.name}. Card is stale or artefact has been tampered with."
    )


@pytest.mark.parametrize("md,pt", _list_card_pt_pairs(), ids=lambda x: x.name)
def test_card_file_size_matches_artifact(md: Path, pt: Path) -> None:
    block = _canonical_block(md.read_text())
    m = _FIELD_REGEX["file_size"].search(block)
    assert m is not None, f"{md.name}: no file_size in canonical block"
    declared = int(m.group(3).replace(",", ""))
    actual = pt.stat().st_size
    assert declared == actual, (
        f"{md.name}: declared file_size {declared} != actual {actual} bytes."
    )


def test_at_least_one_card_present() -> None:
    pairs = _list_card_pt_pairs()
    assert len(pairs) >= 5, f"expected >=5 model cards, found {len(pairs)}"


# ─── TS 28.105 §7.4 spec-mandatory fields (Devil-A Finding #3) ─────────


@pytest.mark.parametrize("md,pt", _list_card_pt_pairs(), ids=lambda x: x.name)
def test_card_has_ts28105_74_mandatory_fields(md: Path, pt: Path) -> None:
    """Every card must carry the four §7.4 spec-mandatory fields:

    * ``inferenceType`` (real-time / batch / stream)
    * ``expectedRunTimeContext`` (compute / latency budgets)
    * ``lastTrainingDataset`` (URI + SHA-256)
    * ``evaluationReport``     (URI + SHA-256)
    """
    block = _canonical_block(md.read_text())
    missing = [
        f for f in _TS28105_74_REQUIRED_FIELDS if not _FIELD_REGEX[f].search(block)
    ]
    assert not missing, (
        f"{md.name}: missing TS 28.105 §7.4 field(s) {missing}. "
        f"Required by Devil-A Finding #3 closure."
    )


@pytest.mark.parametrize("md,pt", _list_card_pt_pairs(), ids=lambda x: x.name)
def test_card_inference_type_is_recognised(md: Path, pt: Path) -> None:
    """``inferenceType`` value must be one of the recognised vocab tokens."""
    block = _canonical_block(md.read_text())
    m = _FIELD_REGEX["inferenceType"].search(block)
    assert m is not None, f"{md.name}: no inferenceType in canonical block"
    val = m.group(1).lower()
    assert val in {"real-time", "batch", "stream", "near-real-time"}, (
        f"{md.name}: inferenceType {val!r} not in TS 28.105 §7.4 vocab."
    )


def test_emitter_helper_round_trips() -> None:
    """The emitter ``compose_canonical_block`` must produce a block that
    passes the same field regexes the audit test asserts."""
    from horizon_ric.observability.model_card import compose_canonical_block

    block = compose_canonical_block(
        checkpoint_name="sla_head_v9.9_test",
        version="9.9_test",
        artefact_path=Path("checkpoints/sla_head_v9.9_test.pt"),
        sha256="0" * 64,
        file_size=12345,
        parameter_count=42,
    )
    for field in _TS28105_74_REQUIRED_FIELDS:
        assert _FIELD_REGEX[field].search(block), (
            f"emitter output missing field {field!r}; "
            f"compose_canonical_block must emit all §7.4 fields"
        )
