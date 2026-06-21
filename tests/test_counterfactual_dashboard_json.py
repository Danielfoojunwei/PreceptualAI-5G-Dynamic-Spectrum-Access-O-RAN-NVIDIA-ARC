"""Lightweight schema test for the counterfactual Grafana dashboard.

The dashboard JSON in `deploy/grafana/dashboards/horizon-counterfactual.json`
is operator-facing — it must be importable into a vanilla Grafana 10+
instance with no manual edits. This test guards three concrete claims
made in the user guide:

  1. The file is parseable JSON (would otherwise fail at Grafana import).
  2. It declares ``schemaVersion >= 36`` (Grafana 10's import floor;
     anything older silently coerces panel options and emits warnings).
  3. It carries the five panels referenced in
     ``docs/COUNTERFACTUAL_USER_GUIDE.md`` (Row 14): decisions/min,
     rejected-alts histogram, top reasons, envelope bytes, audit verify
     p99.

If a future panel ever drops below five, the user guide is silently
out-of-sync. The test makes that loud.
"""

from __future__ import annotations

import json
from pathlib import Path

DASHBOARD = (
    Path(__file__).parent.parent
    / "deploy"
    / "grafana"
    / "dashboards"
    / "horizon-counterfactual.json"
)


def _load() -> dict:
    assert DASHBOARD.exists(), f"missing dashboard JSON: {DASHBOARD}"
    with DASHBOARD.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_dashboard_json_is_parseable() -> None:
    data = _load()
    assert isinstance(data, dict)
    assert "panels" in data
    assert isinstance(data["panels"], list)


def test_dashboard_has_at_least_five_panels() -> None:
    data = _load()
    assert len(data["panels"]) >= 5, (
        f"expected ≥ 5 panels, got {len(data['panels'])}"
    )


def test_dashboard_schema_version_is_modern() -> None:
    data = _load()
    schema_version = data.get("schemaVersion")
    assert isinstance(schema_version, int), (
        f"schemaVersion must be int, got {type(schema_version)}"
    )
    assert schema_version >= 36, (
        f"schemaVersion must be ≥ 36 for Grafana 10+; got {schema_version}"
    )


def test_dashboard_titles_cover_required_panels() -> None:
    """Every panel referenced in COUNTERFACTUAL_USER_GUIDE.md Row 14
    must be present by title prefix. Loose substring match so titles
    can be tweaked for clarity without breaking the test."""
    data = _load()
    titles = [p.get("title", "") for p in data["panels"]]
    required_substrings = [
        "Decisions per minute",
        "Rejected alternatives",
        "Top rejection reasons",
        "envelope size",
        "Audit chain verify latency",
    ]
    for needle in required_substrings:
        assert any(needle.lower() in t.lower() for t in titles), (
            f"required panel substring not found: {needle!r}; "
            f"have titles: {titles}"
        )


def test_dashboard_targets_are_prometheus_expressions() -> None:
    """Every panel target must carry an ``expr`` string — Grafana renders
    nothing for a panel whose target lacks an expression."""
    data = _load()
    for panel in data["panels"]:
        targets = panel.get("targets", [])
        assert targets, f"panel {panel.get('title')!r} has no targets"
        for t in targets:
            assert t.get("expr"), (
                f"panel {panel.get('title')!r} target missing expr: {t}"
            )
