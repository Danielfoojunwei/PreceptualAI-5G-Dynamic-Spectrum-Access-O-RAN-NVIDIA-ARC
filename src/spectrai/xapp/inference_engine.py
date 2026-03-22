"""
Unified inference engine with automatic backend selection.

Supports a fallback chain:  TensorRT -> ONNX Runtime -> PyTorch.
Tracks rolling average latency for monitoring.
"""

from __future__ import annotations

import collections
import logging
import time
from pathlib import Path
from typing import Deque, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class InferenceEngineError(Exception):
    """Raised when no usable inference backend is available."""


class InferenceEngine:
    """
    Unified inference engine for SpectrAI.

    Attempts to load models in order of decreasing performance:
        1. TensorRT engine (``.trt`` / ``.engine``)
        2. ONNX Runtime (``.onnx``)
        3. PyTorch checkpoint (``.pt`` / ``.pth``)

    All three paths expose the same ``predict()`` / ``predict_batch()`` API.
    """

    LATENCY_WINDOW = 100  # rolling window size for latency tracking

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        sequence_length: int = 16,
        input_dim: int = 30,
        num_actions: int = 10,
        # PyTorch-specific (only needed for .pt loading)
        hidden_dim: int = 128,
        latent_dim: int = 128,
        num_layers: int = 2,
    ):
        self._model_path = Path(model_path)
        self._device = device
        self._sequence_length = sequence_length
        self._input_dim = input_dim
        self._num_actions = num_actions
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim
        self._num_layers = num_layers

        self._backend: Optional[str] = None
        self._session: object = None  # ort.InferenceSession or trt context
        self._torch_model: object = None

        # Latency tracking
        self._latencies: Deque[float] = collections.deque(
            maxlen=self.LATENCY_WINDOW
        )

        self._load()

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Try each backend in priority order."""
        suffix = self._model_path.suffix.lower()

        if suffix in (".trt", ".engine"):
            self._try_tensorrt()
        elif suffix == ".onnx":
            self._try_onnx()
        elif suffix in (".pt", ".pth"):
            self._try_pytorch()
        else:
            # Unknown extension — try all
            if not self._try_tensorrt():
                if not self._try_onnx():
                    if not self._try_pytorch():
                        raise InferenceEngineError(
                            f"No backend could load {self._model_path}"
                        )

    def _try_tensorrt(self) -> bool:
        """Attempt to load as a TensorRT engine."""
        try:
            import tensorrt as trt
            try:
                import pycuda.driver as cuda
                import pycuda.autoinit
            except ImportError:
                logger.debug("pycuda not available — skipping TensorRT.")
                return False

            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(str(self._model_path), "rb") as f:
                engine_data = f.read()
            engine = runtime.deserialize_cuda_engine(engine_data)
            if engine is None:
                return False

            context = engine.create_execution_context()

            # Allocate buffers
            input_shape = (1, self._sequence_length, self._input_dim)
            context.set_input_shape("observation", input_shape)

            self._trt_engine = engine
            self._trt_context = context
            self._session = context
            self._backend = "tensorrt"
            logger.info("Loaded TensorRT engine from %s", self._model_path)
            return True
        except Exception as exc:
            logger.debug("TensorRT load failed: %s", exc)
            return False

    def _try_onnx(self) -> bool:
        """Attempt to load as an ONNX model."""
        try:
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cpu":
                providers = ["CPUExecutionProvider"]

            sess = ort.InferenceSession(
                str(self._model_path), providers=providers
            )
            self._session = sess
            self._backend = "onnx"
            logger.info("Loaded ONNX model from %s", self._model_path)
            return True
        except Exception as exc:
            logger.debug("ONNX load failed: %s", exc)
            return False

    def _try_pytorch(self) -> bool:
        """Attempt to load as a PyTorch checkpoint."""
        try:
            import torch
            from spectrai.core.ltc_encoder import LTCEncoder
            from spectrai.core.actor import LTCActor

            encoder = LTCEncoder(
                input_dim=self._input_dim,
                hidden_dim=self._hidden_dim,
                latent_dim=self._latent_dim,
                num_layers=self._num_layers,
            )
            actor = LTCActor(encoder, self._num_actions)

            ckpt = torch.load(
                str(self._model_path),
                map_location=self._device,
                weights_only=False,
            )

            # Support both full agent checkpoints and bare actor state_dicts
            if isinstance(ckpt, dict) and "actor" in ckpt:
                actor.load_state_dict(ckpt["actor"])
            else:
                actor.load_state_dict(ckpt)

            actor.eval()
            if self._device != "cpu":
                actor = actor.to(self._device)

            self._torch_model = actor
            self._torch_device = torch.device(self._device)
            self._backend = "pytorch"
            logger.info("Loaded PyTorch model from %s", self._model_path)
            return True
        except Exception as exc:
            logger.debug("PyTorch load failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, observation: np.ndarray) -> int:
        """
        Predict a single action from an observation.

        Args:
            observation: Array of shape ``(sequence_length, input_dim)``
                         or ``(1, sequence_length, input_dim)``.

        Returns:
            Selected action index (int).
        """
        if observation.ndim == 2:
            observation = observation[np.newaxis, ...]
        probs = self._infer(observation)
        return int(np.argmax(probs[0]))

    def predict_batch(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict actions for a batch of observations.

        Args:
            observations: Array of shape ``(batch, sequence_length, input_dim)``.

        Returns:
            Array of action indices, shape ``(batch,)``.
        """
        probs = self._infer(observations)
        return np.argmax(probs, axis=-1)

    def _infer(self, observations: np.ndarray) -> np.ndarray:
        """Run inference through the active backend, tracking latency."""
        obs = observations.astype(np.float32)
        t0 = time.perf_counter()

        if self._backend == "tensorrt":
            result = self._infer_tensorrt(obs)
        elif self._backend == "onnx":
            result = self._infer_onnx(obs)
        elif self._backend == "pytorch":
            result = self._infer_pytorch(obs)
        else:
            raise InferenceEngineError("No backend loaded.")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._latencies.append(elapsed_ms)

        return result

    def _infer_onnx(self, obs: np.ndarray) -> np.ndarray:
        """Run inference via ONNX Runtime."""
        outputs = self._session.run(None, {"observation": obs})
        return outputs[0]

    def _infer_pytorch(self, obs: np.ndarray) -> np.ndarray:
        """Run inference via PyTorch."""
        import torch

        with torch.no_grad():
            tensor = torch.from_numpy(obs).to(self._torch_device)
            probs = self._torch_model(tensor)
            return probs.cpu().numpy()

    def _infer_tensorrt(self, obs: np.ndarray) -> np.ndarray:
        """Run inference via TensorRT."""
        import pycuda.driver as cuda

        batch_size = obs.shape[0]
        input_shape = (batch_size, self._sequence_length, self._input_dim)
        self._trt_context.set_input_shape("observation", input_shape)

        h_input = np.ascontiguousarray(obs)
        h_output = np.empty((batch_size, self._num_actions), dtype=np.float32)

        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)

        stream = cuda.Stream()
        cuda.memcpy_htod_async(d_input, h_input, stream)

        self._trt_context.set_tensor_address("observation", int(d_input))
        self._trt_context.set_tensor_address("action_probs", int(d_output))
        self._trt_context.execute_async_v3(stream_handle=stream.handle)

        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        d_input.free()
        d_output.free()

        return h_output

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backend(self) -> Optional[str]:
        """Active backend name: 'tensorrt', 'onnx', or 'pytorch'."""
        return self._backend

    @property
    def latency_ms(self) -> float:
        """Rolling average inference latency in milliseconds."""
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def last_latency_ms(self) -> float:
        """Most recent inference latency in milliseconds."""
        if not self._latencies:
            return 0.0
        return self._latencies[-1]
