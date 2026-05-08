"""Top-level (importable) worker module for the federation horizontal test.

Lives outside the test module so that ``multiprocessing`` with the
``spawn`` context can pickle the worker callable by reference. Tensor
state-dicts are serialised to bytes before going through the queue to
sidestep torch's ``rebuild_storage_fd`` shared-memory path (which fails
on the GB10's MP authkey handshake under pytest).
"""

from __future__ import annotations

import io
import pickle

import torch


class TinyLinear(torch.nn.Module):
    def __init__(self, d_in: int = 4, d_out: int = 2):
        super().__init__()
        self.fc = torch.nn.Linear(d_in, d_out, bias=True)

    def forward(self, x):
        return self.fc(x)


def _state_to_bytes(state):
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu().clone() for k, v in state.items()}, buf)
    return buf.getvalue()


def state_from_bytes(b: bytes):
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)


def client_worker(
    client_id: str,
    X_bytes: bytes,
    Y_bytes: bytes,
    init_bytes: bytes,
    epochs: int,
    lr: float,
    out_q,
):
    """Fit a fresh model on a partition. All tensors are passed and
    returned as serialised bytes to avoid torch shared-memory pickling."""
    torch.manual_seed(0)
    X_part = state_from_bytes(X_bytes)["X"]
    Y_part = state_from_bytes(Y_bytes)["Y"]
    init_state = state_from_bytes(init_bytes)

    model = TinyLinear(X_part.shape[1], Y_part.shape[1])
    model.load_state_dict({k: v.clone() for k, v in init_state.items()})
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(X_part) - Y_part) ** 2).mean()
        loss.backward()
        opt.step()
    out_q.put(
        {
            "client_id": client_id,
            "state_bytes": _state_to_bytes(model.state_dict()),
            "sample_count": int(X_part.shape[0]),
        }
    )


def tensor_to_bytes(name: str, t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save({name: t.detach().cpu().clone()}, buf)
    return buf.getvalue()


def state_dict_to_bytes(state) -> bytes:
    return _state_to_bytes(state)
