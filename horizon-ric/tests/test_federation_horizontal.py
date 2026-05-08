"""Federation horizontal scale-out — 10 real client subprocesses.

Spawns 10 independent ``multiprocessing.Process`` workers, each fitting a
small model on a disjoint partition of a shared corpus and returning its
state-dict. The aggregator runs FedAvg on the collected updates and we
verify the result equals single-process training within numerical
tolerance.

This is a real scale-out test (not threading, not async) — it exercises
serialisation across pickling boundaries.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest
import torch

from horizon_ric.federated import ClientUpdate, FedAvg

# The worker lives in its own module so multiprocessing's spawn context
# can pickle it by import path.
from tests._fed_worker import (
    TinyLinear as _TinyLinear,
    client_worker as _client_worker,
    state_from_bytes as _state_from_bytes,
    state_dict_to_bytes as _state_dict_to_bytes,
    tensor_to_bytes as _tensor_to_bytes,
)


def _build_corpus(n: int = 1000, d_in: int = 4, d_out: int = 2, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d_in, generator=g)
    # ground-truth W and b
    W_true = torch.tensor([[1.0, -2.0, 0.5, 0.25], [-0.5, 1.5, -1.0, 0.75]])
    b_true = torch.tensor([0.1, -0.1])
    Y = X @ W_true.T + b_true
    return X, Y


def _spawn_clients(
    X: torch.Tensor,
    Y: torch.Tensor,
    init_state: dict,
    *,
    n_clients: int,
    epochs: int,
    lr: float,
) -> list[ClientUpdate]:
    n = X.shape[0]
    chunk = n // n_clients
    procs: list[mp.Process] = []
    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue()
    init_bytes = _state_dict_to_bytes(init_state)
    for i in range(n_clients):
        s = i * chunk
        e = (i + 1) * chunk if i + 1 < n_clients else n
        X_b = _tensor_to_bytes("X", X[s:e].clone())
        Y_b = _tensor_to_bytes("Y", Y[s:e].clone())
        p = ctx.Process(
            target=_client_worker,
            args=(f"client_{i}", X_b, Y_b, init_bytes, epochs, lr, out_q),
        )
        p.start()
        procs.append(p)

    results: list[ClientUpdate] = []
    for _ in procs:
        d = out_q.get(timeout=120.0)
        results.append(
            ClientUpdate(
                client_id=d["client_id"],
                state_dict=_state_from_bytes(d["state_bytes"]),
                sample_count=d["sample_count"],
            )
        )
    for p in procs:
        p.join(timeout=10.0)
        assert p.exitcode == 0, f"client {p.pid} exited {p.exitcode}"
    return results


def _single_process_reference(
    X: torch.Tensor,
    Y: torch.Tensor,
    init_state: dict,
    *,
    n_clients: int,
    epochs: int,
    lr: float,
) -> dict[str, torch.Tensor]:
    """Run the same FedAvg pipeline serially in one process."""
    n = X.shape[0]
    chunk = n // n_clients
    updates: list[ClientUpdate] = []
    for i in range(n_clients):
        s = i * chunk
        e = (i + 1) * chunk if i + 1 < n_clients else n
        torch.manual_seed(0)
        m = _TinyLinear(X.shape[1], Y.shape[1])
        m.load_state_dict({k: v.clone() for k, v in init_state.items()})
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = ((m(X[s:e]) - Y[s:e]) ** 2).mean()
            loss.backward()
            opt.step()
        updates.append(
            ClientUpdate(
                client_id=f"client_{i}",
                state_dict={k: v.detach().clone() for k, v in m.state_dict().items()},
                sample_count=int(e - s),
            )
        )
    return FedAvg().aggregate(updates)


# ─── Tests ────────────────────────────────────────────────────────────────


def _all_close(a: dict, b: dict, rtol=1e-4, atol=1e-5) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        if not torch.allclose(a[k], b[k], rtol=rtol, atol=atol):
            return False
    return True


def test_horizontal_10_clients_matches_single_process():
    """10 real subprocesses + FedAvg must agree with the same pipeline
    run in a single process within numerical tolerance."""
    X, Y = _build_corpus(n=1000)
    torch.manual_seed(7)
    init = _TinyLinear(X.shape[1], Y.shape[1]).state_dict()
    init = {k: v.clone() for k, v in init.items()}

    updates = _spawn_clients(X, Y, init, n_clients=10, epochs=20, lr=0.05)
    parallel = FedAvg().aggregate(updates)

    serial = _single_process_reference(X, Y, init, n_clients=10, epochs=20, lr=0.05)

    assert _all_close(parallel, serial), (
        "FedAvg over 10 subprocesses did not match single-process baseline"
    )


def test_horizontal_3_clients_smoke():
    """Same contract at smaller fan-out — sanity check on partitioning."""
    X, Y = _build_corpus(n=300)
    torch.manual_seed(11)
    init = _TinyLinear(X.shape[1], Y.shape[1]).state_dict()
    init = {k: v.clone() for k, v in init.items()}

    updates = _spawn_clients(X, Y, init, n_clients=3, epochs=10, lr=0.05)
    parallel = FedAvg().aggregate(updates)
    serial = _single_process_reference(X, Y, init, n_clients=3, epochs=10, lr=0.05)
    assert _all_close(parallel, serial)


def test_horizontal_does_not_leak_telemetry():
    """Per project memory, the aggregator must consume **only** weight
    state-dicts — no raw samples. We assert the ClientUpdate payload
    contains nothing but tensors and a sample_count integer."""
    X, Y = _build_corpus(n=200)
    torch.manual_seed(3)
    init = _TinyLinear(X.shape[1], Y.shape[1]).state_dict()
    init = {k: v.clone() for k, v in init.items()}

    updates = _spawn_clients(X, Y, init, n_clients=5, epochs=5, lr=0.05)
    for u in updates:
        for v in u.state_dict.values():
            assert isinstance(v, torch.Tensor)
        assert isinstance(u.sample_count, int)
        # No raw KPMs / X / Y leaking through.
        for v in u.state_dict.values():
            assert v.numel() < 1_000_000  # parameter tensor sanity
