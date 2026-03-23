"""Tests for the experience replay buffer."""

import numpy as np
import torch

from spectrai.core.replay_buffer import ReplayBuffer


class TestPushAndSample:
    """Basic push/sample round-trip."""

    def test_push_and_sample(self, device):
        state_shape = (8, 12)
        buf = ReplayBuffer(capacity=50, state_shape=state_shape, device=device)

        # Push 20 transitions
        for i in range(20):
            s = np.random.randn(*state_shape).astype(np.float32)
            a = int(np.random.randint(0, 4))
            r = float(np.random.randn())
            ns = np.random.randn(*state_shape).astype(np.float32)
            d = bool(i % 10 == 0)
            buf.push(s, a, r, ns, d)

        assert len(buf) == 20

        batch = buf.sample(8)
        assert set(batch.keys()) == {"states", "actions", "rewards", "next_states", "dones"}
        assert batch["states"].shape == (8, *state_shape)
        assert batch["actions"].shape == (8,)


class TestCapacityOverflow:
    """Ring buffer should wrap around when capacity is exceeded."""

    def test_capacity_overflow(self, device):
        state_shape = (4, 6)
        capacity = 10
        buf = ReplayBuffer(capacity=capacity, state_shape=state_shape, device=device)

        # Push more than capacity
        for i in range(25):
            s = np.full(state_shape, float(i), dtype=np.float32)
            buf.push(s, 0, 0.0, s, False)

        # Size should be capped at capacity
        assert len(buf) == capacity

        # The oldest entries (0..14) should have been overwritten.
        # The buffer should contain entries 15..24.
        # Sample everything and verify no value < 15 remains.
        batch = buf.sample(capacity)
        min_val = batch["states"].min().item()
        assert min_val >= 15.0, f"Expected all values >= 15, got min {min_val}"


class TestSampleShape:
    """Verify tensor shapes and dtypes in sampled batches."""

    def test_sample_shape(self, device):
        state_shape = (8, 12)
        buf = ReplayBuffer(capacity=100, state_shape=state_shape, device=device)

        for _ in range(30):
            s = np.random.randn(*state_shape).astype(np.float32)
            buf.push(s, 1, 0.5, s, False)

        batch = buf.sample(16)

        assert batch["states"].shape == (16, 8, 12)
        assert batch["next_states"].shape == (16, 8, 12)
        assert batch["actions"].shape == (16,)
        assert batch["rewards"].shape == (16,)
        assert batch["dones"].shape == (16,)

        assert batch["states"].dtype == torch.float32
        assert batch["actions"].dtype == torch.int64
        assert batch["rewards"].dtype == torch.float32
        assert batch["dones"].dtype == torch.float32
