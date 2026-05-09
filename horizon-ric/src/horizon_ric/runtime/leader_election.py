"""Lease-based leader election for the PreceptualAI HPA deployment.

Closes Devil-B finding #12: when the Helm chart scales to N>1 replicas,
all replicas would otherwise emit the same A1 policies and double-PUT
into the Near-RT RIC. The expected fix is one of:

  * ``coordination.k8s.io/Lease`` (this module's primary path) — a
    Kubernetes Lease object held by the leader and renewed every
    ``renew_period`` seconds. Followers watch it; if the holder fails
    to renew within ``lease_duration`` they take over.
  * A file-based lease (``LeaderElector(backend=FileLeaseBackend(...))``) for
    single-node tests where the kubernetes API server is unavailable.

Wire format follows
[K8s leader election](https://kubernetes.io/docs/concepts/architecture/leases/):
``Lease.spec`` carries ``holderIdentity``, ``leaseDurationSeconds``,
``acquireTime``, ``renewTime``, ``leaseTransitions``.

Usage::

    elector = LeaderElector(
        identity=os.environ["POD_NAME"],
        lease_name="horizon-rapp-leader",
        lease_namespace="horizon",
        backend=KubernetesLeaseBackend.from_in_cluster(),
        on_started_leading=lambda: lifecycle.start_emitting(),
        on_stopped_leading=lambda: lifecycle.stop_emitting(),
    )
    await elector.run()

Only the leader emits A1 policies; followers are warm standbys. The
elector wakes every ``renew_period`` and either renews (if leader) or
attempts to acquire (if not leader and the current lease has expired).
"""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import structlog

logger = structlog.get_logger(__name__)


DEFAULT_LEASE_DURATION_S: float = 15.0
DEFAULT_RENEW_PERIOD_S: float = 5.0
DEFAULT_RETRY_PERIOD_S: float = 2.0


@dataclass
class LeaseRecord:
    """In-memory representation of a Kubernetes Lease object's spec."""

    holder_identity: Optional[str] = None
    lease_duration_seconds: int = int(DEFAULT_LEASE_DURATION_S)
    acquire_time: Optional[datetime] = None
    renew_time: Optional[datetime] = None
    lease_transitions: int = 0
    # Resource version for optimistic-concurrency updates.
    resource_version: Optional[str] = None

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if self.renew_time is None:
            return True
        return now > self.renew_time + timedelta(
            seconds=self.lease_duration_seconds
        )


class LeaseBackend(ABC):
    """Pluggable storage for the leader lease.

    Two implementations: ``KubernetesLeaseBackend`` (the production
    backend, talks to ``coordination.k8s.io/v1``) and ``FileLeaseBackend``
    (single-node testing). Tests use ``FakeLeaseBackend``.
    """

    @abstractmethod
    def get(self) -> Optional[LeaseRecord]:
        """Return the current lease, or None if it does not exist."""

    @abstractmethod
    def create(self, record: LeaseRecord) -> LeaseRecord:
        """Atomically create the lease. Raises if it already exists."""

    @abstractmethod
    def update(
        self, record: LeaseRecord, *, expected_resource_version: Optional[str]
    ) -> LeaseRecord:
        """Optimistic-concurrency update. Raises if the resource version
        does not match (someone else won the race)."""


class LeaseConflictError(RuntimeError):
    """Raised when an optimistic-concurrency update collides with another writer."""


class FakeLeaseBackend(LeaseBackend):
    """Thread-safe in-memory backend for unit tests.

    The fake implements optimistic-concurrency on a monotonically-
    increasing resource version. Tests share one instance across
    several `LeaderElector` instances to simulate multiple replicas.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._record: Optional[LeaseRecord] = None
        self._rv = 0

    def get(self) -> Optional[LeaseRecord]:
        with self._lock:
            if self._record is None:
                return None
            return LeaseRecord(
                holder_identity=self._record.holder_identity,
                lease_duration_seconds=self._record.lease_duration_seconds,
                acquire_time=self._record.acquire_time,
                renew_time=self._record.renew_time,
                lease_transitions=self._record.lease_transitions,
                resource_version=self._record.resource_version,
            )

    def create(self, record: LeaseRecord) -> LeaseRecord:
        with self._lock:
            if self._record is not None:
                raise LeaseConflictError("lease already exists")
            self._rv += 1
            record.resource_version = str(self._rv)
            self._record = record
            return self.get()  # type: ignore[return-value]

    def update(
        self,
        record: LeaseRecord,
        *,
        expected_resource_version: Optional[str],
    ) -> LeaseRecord:
        with self._lock:
            if self._record is None:
                raise LeaseConflictError("lease does not exist")
            if self._record.resource_version != expected_resource_version:
                raise LeaseConflictError(
                    f"resource version mismatch: have "
                    f"{self._record.resource_version!r}, "
                    f"expected {expected_resource_version!r}"
                )
            self._rv += 1
            record.resource_version = str(self._rv)
            self._record = record
            return self.get()  # type: ignore[return-value]


class FileLeaseBackend(LeaseBackend):
    """Filesystem-backed lease for single-node testing.

    Concurrency is enforced with ``fcntl.flock`` on a sidecar lockfile.
    NOT cluster-aware: only safe when all electors share the same
    filesystem (same node). Documented honestly per Devil-B fix #12 so
    operators don't run this in a real multi-node cluster.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lockpath = self._path.with_suffix(self._path.suffix + ".lock")

    def _read_locked(self) -> tuple[Optional[LeaseRecord], int]:
        import fcntl
        import json

        with open(self._lockpath, "a+") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            if not self._path.exists() or self._path.stat().st_size == 0:
                return None, 0
            data = json.loads(self._path.read_text())
            rv = int(data.get("rv", 0))
            spec = data.get("spec", {})
            rec = LeaseRecord(
                holder_identity=spec.get("holder_identity"),
                lease_duration_seconds=int(
                    spec.get("lease_duration_seconds", DEFAULT_LEASE_DURATION_S)
                ),
                acquire_time=(
                    datetime.fromisoformat(spec["acquire_time"])
                    if spec.get("acquire_time")
                    else None
                ),
                renew_time=(
                    datetime.fromisoformat(spec["renew_time"])
                    if spec.get("renew_time")
                    else None
                ),
                lease_transitions=int(spec.get("lease_transitions", 0)),
                resource_version=str(rv),
            )
            return rec, rv

    def _write_locked(self, record: LeaseRecord, rv: int) -> None:
        import fcntl

        with open(self._lockpath, "a+") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            data = {
                "rv": rv,
                "spec": {
                    "holder_identity": record.holder_identity,
                    "lease_duration_seconds": record.lease_duration_seconds,
                    "acquire_time": (
                        record.acquire_time.isoformat()
                        if record.acquire_time
                        else None
                    ),
                    "renew_time": (
                        record.renew_time.isoformat()
                        if record.renew_time
                        else None
                    ),
                    "lease_transitions": record.lease_transitions,
                },
            }
            self._path.write_text(__import__("json").dumps(data, sort_keys=True))

    def get(self) -> Optional[LeaseRecord]:
        rec, _ = self._read_locked()
        return rec

    def create(self, record: LeaseRecord) -> LeaseRecord:
        existing, _ = self._read_locked()
        if existing is not None:
            raise LeaseConflictError("lease already exists")
        record.resource_version = "1"
        self._write_locked(record, 1)
        return record

    def update(
        self,
        record: LeaseRecord,
        *,
        expected_resource_version: Optional[str],
    ) -> LeaseRecord:
        existing, rv = self._read_locked()
        if existing is None:
            raise LeaseConflictError("lease does not exist")
        if str(rv) != str(expected_resource_version):
            raise LeaseConflictError(
                f"resource version mismatch: have {rv}, "
                f"expected {expected_resource_version!r}"
            )
        new_rv = rv + 1
        record.resource_version = str(new_rv)
        self._write_locked(record, new_rv)
        return record


class KubernetesLeaseBackend(LeaseBackend):  # pragma: no cover - exercised in cluster
    """Production backend talking to ``coordination.k8s.io/v1``.

    Imports ``kubernetes`` lazily so unit tests that use ``FakeLeaseBackend``
    don't pay the import cost or the missing-package risk on aarch64.
    """

    def __init__(self, name: str, namespace: str, *, in_cluster: bool = True) -> None:
        from kubernetes import client, config  # type: ignore

        if in_cluster:
            config.load_incluster_config()
        else:
            config.load_kube_config()
        self._api = client.CoordinationV1Api()
        self._models = client
        self._name = name
        self._namespace = namespace

    @classmethod
    def from_in_cluster(
        cls, name: str = "horizon-rapp-leader", namespace: str = "horizon"
    ) -> "KubernetesLeaseBackend":
        return cls(name, namespace, in_cluster=True)

    def _to_record(self, lease: Any) -> LeaseRecord:
        spec = lease.spec
        return LeaseRecord(
            holder_identity=spec.holder_identity,
            lease_duration_seconds=spec.lease_duration_seconds
            or int(DEFAULT_LEASE_DURATION_S),
            acquire_time=spec.acquire_time,
            renew_time=spec.renew_time,
            lease_transitions=spec.lease_transitions or 0,
            resource_version=lease.metadata.resource_version,
        )

    def _to_lease(self, record: LeaseRecord) -> Any:
        return self._models.V1Lease(
            metadata=self._models.V1ObjectMeta(
                name=self._name,
                namespace=self._namespace,
                resource_version=record.resource_version,
            ),
            spec=self._models.V1LeaseSpec(
                holder_identity=record.holder_identity,
                lease_duration_seconds=int(record.lease_duration_seconds),
                acquire_time=record.acquire_time,
                renew_time=record.renew_time,
                lease_transitions=record.lease_transitions,
            ),
        )

    def get(self) -> Optional[LeaseRecord]:
        from kubernetes.client.exceptions import ApiException  # type: ignore

        try:
            lease = self._api.read_namespaced_lease(self._name, self._namespace)
            return self._to_record(lease)
        except ApiException as exc:
            if exc.status == 404:
                return None
            raise

    def create(self, record: LeaseRecord) -> LeaseRecord:
        from kubernetes.client.exceptions import ApiException  # type: ignore

        try:
            lease = self._api.create_namespaced_lease(
                self._namespace, self._to_lease(record)
            )
            return self._to_record(lease)
        except ApiException as exc:
            if exc.status == 409:
                raise LeaseConflictError("lease already exists") from exc
            raise

    def update(
        self,
        record: LeaseRecord,
        *,
        expected_resource_version: Optional[str],
    ) -> LeaseRecord:
        from kubernetes.client.exceptions import ApiException  # type: ignore

        record.resource_version = expected_resource_version
        try:
            lease = self._api.replace_namespaced_lease(
                self._name, self._namespace, self._to_lease(record)
            )
            return self._to_record(lease)
        except ApiException as exc:
            if exc.status == 409:
                raise LeaseConflictError("conflict on lease update") from exc
            raise


@dataclass
class LeaderElector:
    """Acquires and renews a Kubernetes Lease so exactly one replica leads.

    The behaviour matches the Go ``client-go/tools/leaderelection``
    contract, simplified for our single-purpose use case:

      * Followers periodically inspect the lease; if it has expired,
        they attempt to overwrite it with their own ``holder_identity``.
      * The leader renews the lease every ``renew_period`` seconds; if a
        renew fails, it stops emitting (calls ``on_stopped_leading``)
        and retries to acquire on the next tick.

    ``run()`` is a long-lived coroutine; cancel it on shutdown.
    """

    identity: str
    backend: LeaseBackend
    lease_duration_s: float = DEFAULT_LEASE_DURATION_S
    renew_period_s: float = DEFAULT_RENEW_PERIOD_S
    retry_period_s: float = DEFAULT_RETRY_PERIOD_S
    on_started_leading: Optional[Callable[[], Awaitable[None] | None]] = None
    on_stopped_leading: Optional[Callable[[], Awaitable[None] | None]] = None
    on_new_leader: Optional[Callable[[str], Awaitable[None] | None]] = None
    # Test injection: monotonic clock and sleep coroutine.
    _now: Callable[[], datetime] = field(
        default_factory=lambda: lambda: datetime.now(timezone.utc)
    )
    _sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    _is_leader: bool = field(default=False, init=False)
    _last_observed_leader: Optional[str] = field(default=None, init=False)
    _stop: asyncio.Event = field(default_factory=asyncio.Event, init=False)

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    def stop(self) -> None:
        self._stop.set()

    async def _fire(
        self, cb: Optional[Callable[..., Any]], *args: Any
    ) -> None:
        if cb is None:
            return
        try:
            res = cb(*args)
            if asyncio.iscoroutine(res):
                await res
        except Exception:  # noqa: BLE001
            logger.exception("leader.callback_failed")

    def _new_record(
        self, *, transitions_from: Optional[LeaseRecord]
    ) -> LeaseRecord:
        now = self._now()
        return LeaseRecord(
            holder_identity=self.identity,
            lease_duration_seconds=int(self.lease_duration_s),
            acquire_time=now,
            renew_time=now,
            lease_transitions=(transitions_from.lease_transitions + 1)
            if transitions_from is not None
            and transitions_from.holder_identity != self.identity
            else (
                transitions_from.lease_transitions
                if transitions_from is not None
                else 0
            ),
            resource_version=None,
        )

    async def try_acquire_or_renew(self) -> bool:
        """Single-shot: returns True if we end the call as the leader.

        This is the loop body extracted as a public method so tests can
        drive it manually without spinning up a real asyncio loop.
        """
        existing = self.backend.get()
        now = self._now()
        # Track who currently leads so observers can fire on transition.
        new_leader = existing.holder_identity if existing is not None else None
        if new_leader != self._last_observed_leader:
            self._last_observed_leader = new_leader
            if new_leader is not None:
                await self._fire(self.on_new_leader, new_leader)

        # Case 1: lease does not exist → create it.
        if existing is None:
            try:
                self.backend.create(self._new_record(transitions_from=None))
                return await self._become_leader()
            except LeaseConflictError:
                logger.info("leader.create_lost_race", identity=self.identity)
                return await self._become_follower()

        # Case 2: we already hold it → renew.
        if existing.holder_identity == self.identity:
            renewed = LeaseRecord(
                holder_identity=self.identity,
                lease_duration_seconds=int(self.lease_duration_s),
                acquire_time=existing.acquire_time,
                renew_time=now,
                lease_transitions=existing.lease_transitions,
                resource_version=existing.resource_version,
            )
            try:
                self.backend.update(
                    renewed,
                    expected_resource_version=existing.resource_version,
                )
                self._is_leader = True
                return True
            except LeaseConflictError:
                # Someone else updated the lease — we lost it.
                logger.warning(
                    "leader.renew_failed",
                    identity=self.identity,
                    reason="resource-version-conflict",
                )
                return await self._become_follower()

        # Case 3: someone else holds it → only steal if expired.
        if not existing.is_expired(now=now):
            return await self._become_follower()

        # Lease expired → attempt to acquire.
        new = self._new_record(transitions_from=existing)
        try:
            self.backend.update(
                new, expected_resource_version=existing.resource_version
            )
            return await self._become_leader()
        except LeaseConflictError:
            return await self._become_follower()

    async def _become_leader(self) -> bool:
        was_leader = self._is_leader
        self._is_leader = True
        if not was_leader:
            logger.info("leader.acquired", identity=self.identity)
            await self._fire(self.on_started_leading)
        return True

    async def _become_follower(self) -> bool:
        was_leader = self._is_leader
        self._is_leader = False
        if was_leader:
            logger.info("leader.lost", identity=self.identity)
            await self._fire(self.on_stopped_leading)
        return False

    async def run(self) -> None:
        """Long-lived loop: tick at ``renew_period`` until ``stop()``."""
        logger.info(
            "leader.elector_start",
            identity=self.identity,
            lease_duration_s=self.lease_duration_s,
            renew_period_s=self.renew_period_s,
        )
        try:
            while not self._stop.is_set():
                try:
                    await self.try_acquire_or_renew()
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "leader.tick_error", identity=self.identity
                    )
                    sleep_s = self.retry_period_s
                else:
                    sleep_s = (
                        self.renew_period_s if self._is_leader else self.retry_period_s
                    )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=sleep_s)
                    break
                except asyncio.TimeoutError:
                    pass
        finally:
            if self._is_leader:
                await self._become_follower()


__all__ = [
    "DEFAULT_LEASE_DURATION_S",
    "DEFAULT_RENEW_PERIOD_S",
    "DEFAULT_RETRY_PERIOD_S",
    "FakeLeaseBackend",
    "FileLeaseBackend",
    "KubernetesLeaseBackend",
    "LeaderElector",
    "LeaseBackend",
    "LeaseConflictError",
    "LeaseRecord",
]
