"""Connector registry — name → class plug-in lookup.

Two ways to register a connector:

1. **In-process** — call `register_source("kafka", KafkaSource)` at import
   time. Simplest path for in-tree connectors.
2. **Plugin entry-point** — declare in `pyproject.toml`::

       [project.entry-points."horizon_ric.connectors"]
       kafka = "my_pkg.kafka:KafkaSource"

   At first import of `horizon_ric.io.registry`, all entry-points under
   the `horizon_ric.connectors` group are auto-loaded. External packages
   ship connectors without touching this codebase.

Both methods produce the same lookup behaviour. Calling
`get_source("kafka", config_dict)` instantiates the registered class with
its declared config model.
"""

from __future__ import annotations

import logging
from importlib import metadata
from typing import Any, ClassVar

from horizon_ric.io.connector import (
    ConnectorConfig,
    ConnectorConfigError,
    Sink,
    Source,
)

logger = logging.getLogger(__name__)


class ConnectorRegistry:
    """Two-key registry: ('source' | 'sink', name) → class."""

    _ENTRY_POINT_GROUP: ClassVar[str] = "horizon_ric.connectors"

    def __init__(self):
        self._sources: dict[str, type[Source]] = {}
        self._sinks: dict[str, type[Sink]] = {}
        self._loaded_entry_points = False

    # ── registration ────────────────────────────────────────────────────

    def register_source(self, name: str, cls: type[Source]) -> None:
        if not issubclass(cls, Source):
            raise TypeError(f"{cls!r} is not a Source subclass")
        if name in self._sources and self._sources[name] is not cls:
            raise ValueError(
                f"source {name!r} already registered to {self._sources[name]!r}"
            )
        self._sources[name] = cls
        logger.debug("registered source %s -> %s", name, cls)

    def register_sink(self, name: str, cls: type[Sink]) -> None:
        if not issubclass(cls, Sink):
            raise TypeError(f"{cls!r} is not a Sink subclass")
        if name in self._sinks and self._sinks[name] is not cls:
            raise ValueError(
                f"sink {name!r} already registered to {self._sinks[name]!r}"
            )
        self._sinks[name] = cls
        logger.debug("registered sink %s -> %s", name, cls)

    # ── instantiation ───────────────────────────────────────────────────

    def get_source(self, name: str, config: ConnectorConfig | dict[str, Any]) -> Source:
        self._maybe_load_entry_points()
        cls = self._sources.get(name)
        if cls is None:
            raise ConnectorConfigError(
                f"unknown source {name!r}; registered: {sorted(self._sources)}"
            )
        cfg = self._coerce_config(cls, config)
        return cls(cfg)

    def get_sink(self, name: str, config: ConnectorConfig | dict[str, Any]) -> Sink:
        self._maybe_load_entry_points()
        cls = self._sinks.get(name)
        if cls is None:
            raise ConnectorConfigError(
                f"unknown sink {name!r}; registered: {sorted(self._sinks)}"
            )
        cfg = self._coerce_config(cls, config)
        return cls(cfg)

    # ── introspection ───────────────────────────────────────────────────

    def list_connectors(self) -> dict[str, list[str]]:
        self._maybe_load_entry_points()
        return {
            "sources": sorted(self._sources),
            "sinks": sorted(self._sinks),
        }

    # ── helpers ─────────────────────────────────────────────────────────

    def _coerce_config(
        self, cls: type, config: ConnectorConfig | dict[str, Any]
    ) -> ConnectorConfig:
        # Each connector class may declare a `Config` attribute that
        # subclasses ConnectorConfig with extra fields. Use it if present.
        config_cls: type[ConnectorConfig] = getattr(cls, "Config", ConnectorConfig)
        if isinstance(config, ConnectorConfig):
            if not isinstance(config, config_cls):
                raise ConnectorConfigError(
                    f"{cls.__name__} expects {config_cls.__name__}, got "
                    f"{type(config).__name__}"
                )
            return config
        try:
            return config_cls.model_validate(config)
        except Exception as exc:
            raise ConnectorConfigError(
                f"invalid config for {cls.__name__}: {exc}"
            ) from exc

    def _maybe_load_entry_points(self) -> None:
        if self._loaded_entry_points:
            return
        self._loaded_entry_points = True
        try:
            eps = metadata.entry_points(group=self._ENTRY_POINT_GROUP)
        except Exception:  # pragma: no cover
            return
        for ep in eps:
            try:
                cls = ep.load()
            except Exception as exc:  # pragma: no cover
                logger.warning("failed to load entry point %s: %s", ep.name, exc)
                continue
            if isinstance(cls, type) and issubclass(cls, Source):
                self.register_source(ep.name, cls)
            elif isinstance(cls, type) and issubclass(cls, Sink):
                self.register_sink(ep.name, cls)
            else:
                logger.warning("entry point %s is not Source or Sink", ep.name)


# Module-level singleton.
registry = ConnectorRegistry()


def register_source(name: str, cls: type[Source]) -> None:
    registry.register_source(name, cls)


def register_sink(name: str, cls: type[Sink]) -> None:
    registry.register_sink(name, cls)


def get_source(name: str, config: ConnectorConfig | dict[str, Any]) -> Source:
    return registry.get_source(name, config)


def get_sink(name: str, config: ConnectorConfig | dict[str, Any]) -> Sink:
    return registry.get_sink(name, config)


def list_connectors() -> dict[str, list[str]]:
    return registry.list_connectors()


__all__ = [
    "ConnectorRegistry",
    "get_sink",
    "get_source",
    "list_connectors",
    "register_sink",
    "register_source",
    "registry",
]
