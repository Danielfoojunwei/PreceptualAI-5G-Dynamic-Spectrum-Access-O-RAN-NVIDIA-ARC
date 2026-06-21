"""Kafka connector — relies on optional `confluent-kafka` (or `aiokafka`).

The implementation defers the import so the module can be loaded even if
the underlying client isn't installed; the registry then skips
registration. Once `aiokafka` is on the path, both source and sink work.
"""

from __future__ import annotations

from typing import AsyncIterator

from pydantic import Field

from horizon_ric.io.connector import (
    ConnectorConfig,
    ConnectorIOError,
    ConnectorState,
    Sink,
    Source,
)
from horizon_ric.io.schemas import (
    AuditRecord,
    FeatureFrame,
    PolicyAction,
    TelemetryEvent,
)

try:  # pragma: no cover — optional
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Kafka connector requires `aiokafka`. Install with `pip install aiokafka`. "
        "The connector registry skips Kafka registration when the dependency is absent."
    ) from exc


class _KafkaConfig(ConnectorConfig):
    bootstrap_servers: str = Field(..., description="comma-separated brokers")
    topic: str
    group_id: str | None = None
    """Required for consumer; ignored for producer."""
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: str | None = None
    sasl_plain_username: str | None = None
    sasl_plain_password: str | None = None


class KafkaSource(Source):
    Config = _KafkaConfig
    cfg: _KafkaConfig

    async def connect(self) -> None:  # pragma: no cover — IO
        if not self.cfg.group_id:
            raise ConnectorIOError("KafkaSource requires group_id")
        kwargs = {
            "bootstrap_servers": self.cfg.bootstrap_servers,
            "group_id": self.cfg.group_id,
            "auto_offset_reset": "latest",
            "enable_auto_commit": True,
        }
        if self.cfg.security_protocol != "PLAINTEXT":
            kwargs.update(
                security_protocol=self.cfg.security_protocol,
                sasl_mechanism=self.cfg.sasl_mechanism,
                sasl_plain_username=self.cfg.sasl_plain_username,
                sasl_plain_password=self.cfg.sasl_plain_password,
            )
        self._consumer = AIOKafkaConsumer(self.cfg.topic, **kwargs)
        await self._consumer.start()
        self._state = ConnectorState.STARTED

    async def close(self) -> None:  # pragma: no cover
        if self._state == ConnectorState.STARTED:
            await self._consumer.stop()
        self._state = ConnectorState.STOPPED

    async def stream(self) -> AsyncIterator[TelemetryEvent]:  # pragma: no cover
        async for msg in self._consumer:
            try:
                yield TelemetryEvent.model_validate_json(msg.value)
            except Exception as exc:
                raise ConnectorIOError(
                    f"failed to parse Kafka message: {exc}"
                ) from exc


class KafkaSink(Sink):
    Config = _KafkaConfig
    cfg: _KafkaConfig

    async def connect(self) -> None:  # pragma: no cover
        kwargs = {"bootstrap_servers": self.cfg.bootstrap_servers}
        if self.cfg.security_protocol != "PLAINTEXT":
            kwargs.update(
                security_protocol=self.cfg.security_protocol,
                sasl_mechanism=self.cfg.sasl_mechanism,
                sasl_plain_username=self.cfg.sasl_plain_username,
                sasl_plain_password=self.cfg.sasl_plain_password,
            )
        self._producer = AIOKafkaProducer(**kwargs)
        await self._producer.start()
        self._state = ConnectorState.STARTED

    async def close(self) -> None:  # pragma: no cover
        if self._state == ConnectorState.STARTED:
            await self._producer.stop()
        self._state = ConnectorState.STOPPED

    async def write(
        self,
        message: TelemetryEvent | FeatureFrame | PolicyAction | AuditRecord,
    ) -> None:  # pragma: no cover
        await self._producer.send_and_wait(
            self.cfg.topic,
            message.model_dump_json().encode("utf-8"),
        )


__all__ = ["KafkaSink", "KafkaSource"]
