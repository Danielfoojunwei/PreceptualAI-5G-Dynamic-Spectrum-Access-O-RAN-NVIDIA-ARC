"""rApp daemon entrypoint for the chaos test.

Boots a real `HorizonRAppLifecycle` against the env-configured SMO,
adds a UDP listener that feeds incoming packets into a real
`BackpressureQueue` (so the chaos harness can flood it), and runs
`run_forever()` until SIGTERM.

This is a real daemon — no mocks, no test doubles. The chaos harness
launches it as a subprocess and validates that it survives real
failure injection.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from horizon_ric.rapp.a1_adapter import A1AdapterConfig
from horizon_ric.rapp.lifecycle import HorizonRAppLifecycle
from horizon_ric.rapp.r1_adapter import R1AdapterConfig
from horizon_ric.runtime.backpressure import BackpressureQueue
from horizon_ric.runtime.graceful_degradation import FailureMode


class _UDPProtocol(asyncio.DatagramProtocol):
    """Drop incoming UDP packets into the backpressure queue."""

    def __init__(self, queue: BackpressureQueue[bytes]) -> None:
        self._q = queue

    def datagram_received(self, data: bytes, _addr) -> None:
        # Use put_nowait via the queue (oldest policy: never blocks).
        try:
            self._q.put_nowait(data)
        except Exception:
            # Even an error here must not crash the daemon.
            pass


async def _drain_queue(queue: BackpressureQueue[bytes]) -> None:
    """Background consumer that drains the queue at a fixed rate."""
    while True:
        try:
            await queue.get()
            # Simulate slow processing so the queue can build up.
            await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Defensive: never let a consumer blow up the daemon.
            await asyncio.sleep(0.01)


async def _run() -> int:
    queue = BackpressureQueue[bytes](
        name="chaos.tick", max_depth=128, drop_policy="oldest"
    )

    # Bind the UDP listener for the chaos flood test.
    udp_port = int(os.environ.get("HORIZON_CHAOS_TICK_PORT", "0"))
    if udp_port:
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(
            lambda: _UDPProtocol(queue),
            local_addr=("127.0.0.1", udp_port),
        )

    drain_task = asyncio.create_task(_drain_queue(queue))

    async def _periodic_emit(lc):
        """Try to emit an A1 policy every 200 ms.

        Each emit goes through the real circuit breaker and the real
        SMO. When the SMO is down (chaos), the failures trip the
        breaker — exercising graceful degradation for real.
        """
        while True:
            try:
                await asyncio.sleep(0.2)
                if lc.state.value != "running":
                    continue
                try:
                    await lc.a1.emit_policy(
                        "horizon.qos.priority",
                        {
                            "scope": {"slice_id": "chaos"},
                            "qos_objectives": {"priority": 5},
                        },
                    )
                except Exception:
                    # Failures (CB open, 5xx, timeout) are expected during chaos.
                    pass
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(0.2)

    r1 = R1AdapterConfig(
        smo_base_url=os.environ.get("HORIZON_SMO_URL", "http://127.0.0.1:0"),
        rapp_id=os.environ.get("HORIZON_RAPP_ID", "horizon-chaos"),
    )
    a1 = A1AdapterConfig(
        near_rt_ric_base_url=os.environ.get(
            "HORIZON_NEAR_RT_RIC_URL", "http://127.0.0.1:0"
        ),
        rapp_id=r1.rapp_id,
    )
    lifecycle = HorizonRAppLifecycle(
        r1_config=r1,
        a1_config=a1,
        health_host="127.0.0.1",
        health_port=int(os.environ.get("HORIZON_HEALTH_PORT", "8081")),
        state_path=Path(
            os.environ.get(
                "HORIZON_STATE_PATH", "/tmp/horizon-chaos-state.json"
            )
        ),
        # Fast cadences so the chaos test can observe their effects in 60 s.
        checkpoint_interval_s=2.0,
        watchdog_interval_s=2.0,
    )

    # When R1/A1 fail, transition the degradation controller. This is
    # what production code does — we install a tiny background monitor
    # that flips to KEEP_LAST_GOOD if the R1 breaker opens.
    async def _degradation_monitor():
        while True:
            try:
                if lifecycle.r1.circuit_breaker.state == "open":
                    lifecycle.degradation.enter(
                        FailureMode.SMO_UNREACHABLE, reason="r1_breaker_open"
                    )
                else:
                    if not lifecycle.degradation.is_serving():
                        lifecycle.degradation.recover(reason="r1_breaker_closed")
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(0.5)

    monitor_task = asyncio.create_task(_degradation_monitor())
    emit_task = asyncio.create_task(_periodic_emit(lifecycle))

    try:
        await lifecycle.run_forever()
    finally:
        drain_task.cancel()
        monitor_task.cancel()
        emit_task.cancel()
        for t in (drain_task, monitor_task, emit_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
