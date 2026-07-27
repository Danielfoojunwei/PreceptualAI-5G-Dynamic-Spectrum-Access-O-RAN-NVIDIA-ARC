"""KafkaSource.from_dict config mapping + the loud missing-dependency path.

No broker is faked and none is required: only the config mapping and the
real import behaviour (``aiokafka`` absent → loud, actionable error) are
tested here.
"""

from __future__ import annotations

import importlib.util

import pytest

from horizon_ric.io.connector import ConnectorConfigError, ConnectorState
from horizon_ric.io.connectors.kafka_connector import (
    KafkaDependencyError,
    KafkaSource,
)

_AIOKAFKA_INSTALLED = importlib.util.find_spec("aiokafka") is not None


def test_from_dict_maps_runner_yaml_source_block():
    src = KafkaSource.from_dict(
        {
            "type": "kafka",  # routing key must be dropped, not rejected
            "bootstrap_servers": "broker-1:9092,broker-2:9092",
            "topic": "horizon.telemetry",
            "group_id": "horizon-rapp",
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": "PLAIN",
            "sasl_plain_username": "svc",
            "sasl_plain_password": "secret",
        }
    )
    assert src.cfg.bootstrap_servers == "broker-1:9092,broker-2:9092"
    assert src.cfg.topic == "horizon.telemetry"
    assert src.cfg.group_id == "horizon-rapp"
    assert src.cfg.security_protocol == "SASL_SSL"
    assert src.cfg.sasl_mechanism == "PLAIN"
    assert src.cfg.sasl_plain_username == "svc"
    assert src.cfg.sasl_plain_password == "secret"
    assert src.cfg.name == "daemon-kafka-source"  # stable default source_id
    assert src.state == ConnectorState.INIT  # not connected yet


def test_from_dict_minimal_block_and_name_override():
    src = KafkaSource.from_dict(
        {
            "bootstrap_servers": "localhost:9092",
            "topic": "t",
            "group_id": "g",
            "name": "edge-kafka",
        }
    )
    assert src.cfg.name == "edge-kafka"
    assert src.cfg.security_protocol == "PLAINTEXT"


@pytest.mark.parametrize(
    "cfg,missing",
    [
        ({"topic": "t", "group_id": "g"}, "bootstrap_servers"),
        ({"bootstrap_servers": "b:9092", "group_id": "g"}, "topic"),
    ],
)
def test_from_dict_missing_required_keys_fails_loudly(cfg, missing):
    with pytest.raises(ConnectorConfigError, match=missing):
        KafkaSource.from_dict(cfg)


def test_from_dict_unknown_key_fails_loudly():
    with pytest.raises(ConnectorConfigError, match="bootstrap_serverz"):
        KafkaSource.from_dict(
            {
                "bootstrap_serverz": "typo:9092",  # extra='forbid' catches typos
                "topic": "t",
                "group_id": "g",
            }
        )


@pytest.mark.skipif(
    _AIOKAFKA_INSTALLED, reason="aiokafka installed — missing-dep path not reachable"
)
@pytest.mark.asyncio
async def test_connect_without_aiokafka_fails_loudly():
    src = KafkaSource.from_dict(
        {"bootstrap_servers": "localhost:9092", "topic": "t", "group_id": "g"}
    )
    with pytest.raises(KafkaDependencyError, match=r"horizon-ric\[kafka\]"):
        await src.connect()
    # State never silently flips to STARTED.
    assert src.state == ConnectorState.INIT


@pytest.mark.asyncio
async def test_stream_before_connect_is_refused():
    src = KafkaSource.from_dict(
        {"bootstrap_servers": "localhost:9092", "topic": "t", "group_id": "g"}
    )
    from horizon_ric.io.connector import ConnectorIOError

    with pytest.raises(ConnectorIOError, match="not connected"):
        async for _ in src.stream():  # pragma: no cover — raises immediately
            pass
