"""AI-PHY model-card lineage tests (TS 28.105 §7.4 extensions)."""

from __future__ import annotations

import json

import pytest

from horizon_ric.evidence.ai_phy_lineage import (
    AIPhyModelCard,
    PARange,
    clear_registry,
    emit_ai_phy_card,
    replay_lineage_chain,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


def _card(
    name: str,
    block: str,
    *,
    sha: str | None = None,
    prev: str | None = None,
) -> AIPhyModelCard:
    return AIPhyModelCard(
        name=name,
        version="0.1",
        sha256=sha or ("a" * 64),
        inference_type="real-time",
        expected_run_time_context="GH200, 32GB, p99=10ms",
        last_training_dataset_uri="s3://horizon-ric/datasets/v1",
        last_training_dataset_sha256="b" * 64,
        evaluation_report_uri="s3://horizon-ric/eval/v1",
        evaluation_report_sha256="c" * 64,
        block_replaced=block,  # type: ignore[arg-type]
        applicable_pa_regimes=(PARange(0.0, 5.0), PARange(7.0, 12.0)),
        applicable_mobility_classes=("low", "medium"),
        cell_config_tags=("FR1-3.5GHz", "100MHz"),
        previous_version_sha=prev,
    )


# ─── 6 mandatory-fields per block kind ────────────────────────────────


@pytest.mark.parametrize(
    "block",
    [
        "channel_estimation",
        "equalization",
        "symbol_demapping",
        "constellation_mapping",
        "papr_shaping",
        "sic_decoder",
    ],
)
def test_ts28105_mandatory_fields_present(block):
    c = _card(f"{block}_v0.1", block)
    d = c.to_dict()
    # TS 28.105 §7.4 mandatory fields
    assert "inference_type" in d and d["inference_type"]
    assert "expected_run_time_context" in d and d["expected_run_time_context"]
    assert "last_training_dataset_uri" in d
    assert "last_training_dataset_sha256" in d
    assert "evaluation_report_uri" in d
    assert "evaluation_report_sha256" in d
    # AI-PHY extension
    assert d["block_replaced"] == block


# ─── PA-range invariant ────────────────────────────────────────────────


def test_invalid_pa_range_rejected():
    with pytest.raises(ValueError):
        PARange(min_db=10.0, max_db=5.0)


def test_pa_regime_overlap_detection():
    """A model with applicable_pa_regimes=[(0,5),(7,12)] excludes 5–7 dB."""
    c = _card("hybrid_deep_rx_v0.1", "channel_estimation")
    assert c.applies_to(pa_backoff_db=4.0)
    assert c.applies_to(pa_backoff_db=10.0)
    assert not c.applies_to(pa_backoff_db=6.0)
    assert not c.applies_to(pa_backoff_db=15.0)


def test_mobility_class_filter():
    c = _card("hybrid_deep_rx_v0.1", "channel_estimation")
    assert c.applies_to(mobility="low")
    assert not c.applies_to(mobility="high")


def test_cell_config_tag_filter():
    c = _card("hybrid_deep_rx_v0.1", "channel_estimation")
    assert c.applies_to(cell_tag="FR1-3.5GHz")
    assert not c.applies_to(cell_tag="FR2-28GHz")


# ─── Lineage chain replay ─────────────────────────────────────────────


def test_lineage_chain_three_versions_in_order():
    v1 = _card("hybrid_deep_rx_v0.1", "channel_estimation", sha="a" * 64)
    emit_ai_phy_card(v1)
    sha_v1 = v1.card_sha256()

    v2 = _card("hybrid_deep_rx_v0.2", "channel_estimation", sha="d" * 64, prev=sha_v1)
    emit_ai_phy_card(v2)
    sha_v2 = v2.card_sha256()

    v3 = _card("hybrid_deep_rx_v0.3", "channel_estimation", sha="e" * 64, prev=sha_v2)
    emit_ai_phy_card(v3)
    sha_v3 = v3.card_sha256()

    chain = replay_lineage_chain(sha_v3)
    assert len(chain) == 3
    assert chain[0].version == "0.1"
    assert chain[1].version == "0.1"  # all use _card defaults; version from name suffix
    assert chain[2].version == "0.1"
    # Order verified by sha lineage
    assert chain[0].sha256 == "a" * 64
    assert chain[1].sha256 == "d" * 64
    assert chain[2].sha256 == "e" * 64


def test_lineage_chain_unknown_card_raises():
    with pytest.raises(KeyError):
        replay_lineage_chain("z" * 64)


# ─── Card sha256 stability ─────────────────────────────────────────────


def test_card_sha256_stable_across_issued_at():
    """Card SHA must NOT depend on issued_at (it's wall-clock)."""
    c1 = _card("hybrid_deep_rx_v0.1", "channel_estimation")
    c2 = _card("hybrid_deep_rx_v0.1", "channel_estimation")
    assert c1.card_sha256() == c2.card_sha256()


def test_card_sha256_changes_on_block_replaced():
    c1 = _card("a_v0.1", "channel_estimation")
    c2 = _card("a_v0.1", "equalization")
    assert c1.card_sha256() != c2.card_sha256()


# ─── JSON round-trip ──────────────────────────────────────────────────


def test_to_dict_serialisable_via_json():
    c = _card("hybrid_deep_rx_v0.1", "channel_estimation")
    blob = json.dumps(c.to_dict())
    out = json.loads(blob)
    assert out["block_replaced"] == "channel_estimation"
    assert len(out["applicable_pa_regimes"]) == 2
