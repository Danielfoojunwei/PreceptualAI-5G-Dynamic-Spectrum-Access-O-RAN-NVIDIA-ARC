"""Tests for ONNX model export."""

import os
import tempfile

import numpy as np
import pytest
import torch

from spectrai.core.actor import LTCActor
from spectrai.core.ltc_encoder import LTCEncoder
from spectrai.export.onnx_export import export_actor_onnx


INPUT_DIM = 12
HIDDEN_DIM = 32
LATENT_DIM = 32
NUM_ACTIONS = 4
SEQ_LEN = 8


class _FakeAgent:
    """Minimal stand-in exposing an .actor attribute for export."""

    def __init__(self, device: torch.device):
        encoder = LTCEncoder(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            latent_dim=LATENT_DIM,
            num_layers=1,
        ).to(device)
        self.actor = LTCActor(encoder, NUM_ACTIONS).to(device)


@pytest.fixture
def fake_agent(device):
    return _FakeAgent(device)


class TestExportCreatesFile:
    """ONNX export should produce a valid file on disk."""

    def test_export_creates_file(self, fake_agent):
        onnx = pytest.importorskip("onnx")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "actor.onnx")
            result_path = export_actor_onnx(
                agent=fake_agent,
                output_path=path,
                sequence_length=SEQ_LEN,
                input_dim=INPUT_DIM,
                validate=False,
            )

            assert os.path.isfile(result_path)
            assert os.path.getsize(result_path) > 0

            # Validate the ONNX model structure
            model = onnx.load(result_path)
            onnx.checker.check_model(model)


class TestExportOutputMatchesPytorch:
    """ONNX inference output should match PyTorch output within tolerance."""

    def test_export_output_matches_pytorch(self, fake_agent):
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "actor.onnx")
            export_actor_onnx(
                agent=fake_agent,
                output_path=path,
                sequence_length=SEQ_LEN,
                input_dim=INPUT_DIM,
                validate=False,
            )

            # PyTorch reference output
            dummy = torch.randn(1, SEQ_LEN, INPUT_DIM)
            fake_agent.actor.eval()
            with torch.no_grad():
                pt_output = fake_agent.actor(dummy).cpu().numpy()

            # ONNX Runtime output
            session = ort.InferenceSession(path)
            input_name = session.get_inputs()[0].name
            ort_inputs = {input_name: dummy.cpu().numpy()}
            ort_output = session.run(None, ort_inputs)[0]

            np.testing.assert_allclose(
                pt_output,
                ort_output,
                rtol=1e-4,
                atol=1e-5,
                err_msg="ONNX output diverges from PyTorch output",
            )
