"""Kafka connector — relies on optional `aiokafka` (install extra: `kafka`).

The implementation defers the aiokafka import to ``connect()`` time so the
module can be loaded (and configs validated) even if the underlying client
isn't installed. Connecting without ``aiokafka`` on the path raises a loud
:class:`KafkaDependencyError` telling the operator exactly what to install —
never a silent no-op.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from pydantic import Field, ValidationError

from horizon_ric.io.connector import (
    ConnectorConfig,
    ConnectorConfigError,
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


class KafkaDependencyError(ConnectorIOError):
    """Raised when `aiokafka` is required but not installed."""


def _import_aiokafka() -> Any:
    """Import and return the ``aiokafka`` module, or fail loudly."""
    try:
        import aiokafka
    except ImportError as exc:
        raise KafkaDependencyError(
            "Kafka connector requires the optional `aiokafka` dependency. "
            "Install it with `pip install 'horizon-ric[kafka]'` "
            "(or `pip install aiokafka`)."
        ) from exc
    return aiokafka


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

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "KafkaSource":
        """Build a KafkaSource from the daemon's YAML ``source:`` block.

        Maps the runner-facing keys (``bootstrap_servers``, ``topic``,
        ``group_id``, optional SASL/TLS knobs) onto :class:`_KafkaConfig`.
        The routing key ``type`` is dropped; ``name`` defaults to
        ``daemon-kafka-source``. Invalid or missing keys raise a loud
        :class:`~horizon_ric.io.connector.ConnectorConfigError` naming the
        offending fields — never a silent fallback.
        """
        data = {k: v for k, v in dict(cfg).items() if k != "type"}
        data.setdefault("name", "daemon-kafka-source")
        try:
            config = _KafkaConfig(**data)
        except ValidationError as exc:
            raise ConnectorConfigError(
                f"invalid Kafka source config: {exc}"
            ) from exc
        return cls(config)

    async def connect(self) -> None:  # pragma: no cover — IO
        aiokafka = _import_aiokafka()
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
        self._consumer = aiokafka.AIOKafkaConsumer(self.cfg.topic, **kwargs)
        await self._consumer.start()
        self._state = ConnectorState.STARTED

    async def close(self) -> None:  # pragma: no cover
        if self._state == ConnectorState.STARTED:
            await self._consumer.stop()
        self._state = ConnectorState.STOPPED

    async def stream(self) -> AsyncIterator[TelemetryEvent]:  # pragma: no cover
        if self._state != ConnectorState.STARTED:
            raise ConnectorIOError(
                "KafkaSource not connected — call `await connect()` first"
            )
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
        aiokafka = _import_aiokafka()
        kwargs = {"bootstrap_servers": self.cfg.bootstrap_servers}
        if self.cfg.security_protocol != "PLAINTEXT":
            kwargs.update(
                security_protocol=self.cfg.security_protocol,
                sasl_mechanism=self.cfg.sasl_mechanism,
                sasl_plain_username=self.cfg.sasl_plain_username,
                sasl_plain_password=self.cfg.sasl_plain_password,
            )
        self._producer = aiokafka.AIOKafkaProducer(**kwargs)
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


__all__ = ["KafkaDependencyError", "KafkaSink", "KafkaSource"]
