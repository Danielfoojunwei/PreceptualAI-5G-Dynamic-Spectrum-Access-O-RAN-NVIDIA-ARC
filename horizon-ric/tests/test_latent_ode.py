"""Latent-ODE encoder + ODE decoder tests."""

import pytest
import torch

from horizon_ric.core.latent_ode import (
    LatentODEConfig,
    LatentODEPosterior,
    ODESolverHead,
    reparameterize,
)


class TestLatentODEPosterior:
    def test_irregular_observations_shape(self):
        cfg = LatentODEConfig(obs_dim=4, latent_dim=8, hidden_dim=16)
        enc = LatentODEPosterior(cfg)
        obs = [
            (0.0, torch.randn(2, 4)),
            (0.5, torch.randn(2, 4)),
            (1.7, torch.randn(2, 4)),
        ]
        mu, logvar = enc(obs)
        assert mu.shape == (2, 8)
        assert logvar.shape == (2, 8)

    def test_grad_flows(self):
        cfg = LatentODEConfig(obs_dim=2, latent_dim=4, hidden_dim=8)
        enc = LatentODEPosterior(cfg)
        x1 = torch.randn(3, 2, requires_grad=True)
        x2 = torch.randn(3, 2)
        mu, _ = enc([(0.0, x1), (1.0, x2)])
        mu.sum().backward()
        assert x1.grad is not None
        assert x1.grad.abs().sum().item() > 0

    def test_rejects_wrong_dim(self):
        cfg = LatentODEConfig(obs_dim=4)
        enc = LatentODEPosterior(cfg)
        with pytest.raises(ValueError):
            enc([(0.0, torch.zeros(2, 9))])  # wrong obs_dim

    def test_rejects_descending_times(self):
        cfg = LatentODEConfig(obs_dim=4)
        enc = LatentODEPosterior(cfg)
        with pytest.raises(ValueError):
            enc([(1.0, torch.zeros(2, 4)), (0.5, torch.zeros(2, 4))])

    def test_rejects_empty(self):
        cfg = LatentODEConfig(obs_dim=4)
        enc = LatentODEPosterior(cfg)
        with pytest.raises(ValueError):
            enc([])


class TestODESolverHead:
    def test_trajectory_shape(self):
        cfg = LatentODEConfig(obs_dim=4, latent_dim=8, hidden_dim=16)
        dec = ODESolverHead(cfg)
        z0 = torch.randn(3, 8)
        queries = torch.tensor([0.0, 0.5, 1.0, 2.0])
        traj = dec(z0, queries)
        assert traj.shape == (3, 4, 8)

    def test_zero_query_returns_z0(self):
        cfg = LatentODEConfig(obs_dim=4, latent_dim=8, hidden_dim=16)
        dec = ODESolverHead(cfg)
        z0 = torch.randn(2, 8)
        traj = dec(z0, torch.tensor([0.0]))
        # Integration over zero interval is the identity.
        assert torch.allclose(traj[:, 0, :], z0, atol=1e-6)

    def test_empty_queries(self):
        cfg = LatentODEConfig(latent_dim=8, hidden_dim=16, obs_dim=4)
        dec = ODESolverHead(cfg)
        z0 = torch.randn(2, 8)
        traj = dec(z0, torch.zeros(0))
        assert traj.shape == (2, 0, 8)

    def test_rejects_descending_queries(self):
        cfg = LatentODEConfig(latent_dim=4, obs_dim=2)
        dec = ODESolverHead(cfg)
        with pytest.raises(ValueError):
            dec(torch.zeros(1, 4), torch.tensor([0.0, 1.0, 0.5]))

    def test_rejects_wrong_z0(self):
        cfg = LatentODEConfig(latent_dim=4, obs_dim=2)
        dec = ODESolverHead(cfg)
        with pytest.raises(ValueError):
            dec(torch.zeros(4), torch.tensor([0.0, 1.0]))


class TestReparameterize:
    def test_shapes_match(self):
        mu = torch.randn(4, 8)
        logvar = torch.randn(4, 8)
        z = reparameterize(mu, logvar)
        assert z.shape == mu.shape

    def test_zero_logvar_no_noise(self):
        mu = torch.randn(2, 4)
        # logvar = -inf → sigma = 0 → z = mu (modulo finite arithmetic).
        logvar = torch.full_like(mu, -20.0)
        z = reparameterize(mu, logvar)
        assert torch.allclose(z, mu, atol=1e-3)
