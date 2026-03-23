"""
ONNX export for the LTC Actor network.

Exports the actor's forward pass (state -> action probabilities) to ONNX
format for deployment on NVIDIA TensorRT / ARC inference pipelines.
"""

from pathlib import Path

import torch

from spectrai.core.actor import LTCActor


def export_actor_to_onnx(
    actor: LTCActor,
    input_shape: tuple,
    output_path: str,
    opset_version: int = 17,
) -> str:
    """
    Export an LTCActor to ONNX format.

    Args:
        actor:         Trained LTCActor instance.
        input_shape:   (sequence_length, input_dim) — single observation shape.
        output_path:   Destination file path for the .onnx model.
        opset_version: ONNX opset version (default 17).

    Returns:
        Absolute path to the exported ONNX file.
    """
    actor.eval()
    device = next(actor.parameters()).device

    # Dummy input: batch of 1
    dummy = torch.randn(1, *input_shape, device=device)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        actor,
        (dummy,),
        str(path),
        input_names=["state"],
        output_names=["action_probs"],
        dynamic_axes={
            "state": {0: "batch_size"},
            "action_probs": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    return str(path.resolve())
