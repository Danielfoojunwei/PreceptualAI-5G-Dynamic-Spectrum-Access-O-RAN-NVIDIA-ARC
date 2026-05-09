"""Edge agent CLI: real round-trip on a checkpointed risk head."""

import io
import json
from pathlib import Path

import torch

from horizon_ric.agent.inference import EdgeAgentConfig, main, run
from horizon_ric.heads.sla_risk import SLARiskConfig, SLARiskHead


def _save_head_checkpoint(path: Path, latent_dim: int = 32) -> None:
    head = SLARiskHead(SLARiskConfig(latent_dim=latent_dim, hidden_dim=16))
    torch.save(head.state_dict(), path)


class TestMainErrorPaths:
    def test_main_no_checkpoint_returns_2(self, capsys):
        rc = main([])  # no --checkpoint, no env
        assert rc == 2
        captured = capsys.readouterr()
        assert "checkpoint" in captured.err.lower()

    def test_main_missing_file_returns_2(self, tmp_path: Path, capsys):
        rc = main(["--checkpoint", str(tmp_path / "does_not_exist.pt")])
        assert rc == 2
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()


class TestInferenceLoop:
    def test_run_processes_real_telemetry(self, tmp_path: Path):
        ckpt = tmp_path / "head.pt"
        _save_head_checkpoint(ckpt, latent_dim=32)

        # Build telemetry stream: 3 valid records + 1 garbage line
        records = [
            json.dumps({"latent": [0.0] * 32}),
            json.dumps({"latent": [0.1] * 32}),
            "{not valid json",
            json.dumps({"latent": [-0.5] * 32}),
        ]
        stream = io.StringIO("\n".join(records) + "\n")
        sink = io.StringIO()

        cfg = EdgeAgentConfig(checkpoint_path=ckpt, latent_dim=32, hidden_dim=16)
        n = run(cfg, stream, sink)

        assert n == 3
        out_lines = [l for l in sink.getvalue().splitlines() if l.strip()]
        assert len(out_lines) == 3
        for line in out_lines:
            obj = json.loads(line)
            assert "h_30s" in obj
            assert "h_60s" in obj
            assert "h_300s" in obj
            assert 0.0 <= obj["h_30s"] <= 1.0

    def test_run_skips_wrong_latent_shape(self, tmp_path: Path):
        ckpt = tmp_path / "head.pt"
        _save_head_checkpoint(ckpt, latent_dim=32)

        # latent of length 16, mismatched
        stream = io.StringIO(json.dumps({"latent": [0.0] * 16}) + "\n")
        sink = io.StringIO()
        cfg = EdgeAgentConfig(checkpoint_path=ckpt, latent_dim=32, hidden_dim=16)
        n = run(cfg, stream, sink)
        assert n == 0
        assert sink.getvalue() == ""

    def test_max_iterations_caps_loop(self, tmp_path: Path):
        ckpt = tmp_path / "head.pt"
        _save_head_checkpoint(ckpt, latent_dim=32)

        many = "\n".join(json.dumps({"latent": [0.0] * 32}) for _ in range(50))
        stream = io.StringIO(many + "\n")
        sink = io.StringIO()

        cfg = EdgeAgentConfig(
            checkpoint_path=ckpt, latent_dim=32, hidden_dim=16, max_iterations=5
        )
        n = run(cfg, stream, sink)
        assert n == 5
