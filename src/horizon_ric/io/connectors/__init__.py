"""Reference connector implementations.

In-tree connectors registered with the global registry on import.

    file        FileSource / FileSink — JSONL on disk; zero deps.
    http        HttpRestSource / HttpRestSink — async httpx polling/post.
    kafka       KafkaSource / KafkaSink — optional `confluent-kafka` dep.
    grpc        GrpcStreamSource / GrpcStreamSink — optional `grpcio` dep.

External packages register their own via the
`horizon_ric.connectors` entry-point group; see `io.registry`.
"""

from horizon_ric.io.connectors.file_connector import FileSink, FileSource
from horizon_ric.io.connectors.http_connector import HttpRestSink, HttpRestSource
from horizon_ric.io.registry import register_sink, register_source

# Auto-register on import. External packages can override via entry-points.
register_source("file", FileSource)
register_sink("file", FileSink)
register_source("http", HttpRestSource)
register_sink("http", HttpRestSink)

# Optional connectors — registered only if their underlying deps are present.
try:  # pragma: no cover
    from horizon_ric.io.connectors.kafka_connector import KafkaSink, KafkaSource

    register_source("kafka", KafkaSource)
    register_sink("kafka", KafkaSink)
except ImportError:  # pragma: no cover
    pass

try:  # pragma: no cover
    from horizon_ric.io.connectors.grpc_connector import (
        GrpcStreamSink,
        GrpcStreamSource,
    )

    register_source("grpc", GrpcStreamSource)
    register_sink("grpc", GrpcStreamSink)
except ImportError:  # pragma: no cover
    pass


__all__ = [
    "FileSink",
    "FileSource",
    "HttpRestSink",
    "HttpRestSource",
]
