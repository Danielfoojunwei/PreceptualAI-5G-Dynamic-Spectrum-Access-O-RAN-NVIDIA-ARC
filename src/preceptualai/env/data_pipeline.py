"""
Unified 5G Data Pipeline for PreceptualAI O-RAN xApp.

Normalizes diverse real 5G dataset formats into a common tensor format
compatible with the SpectrumEnv observation space.

Supported real data sources:
  - UCC MISL 5G Dataset (CSV: RSRP, RSRQ, SNR, CQI, RSSI)
  - Colosseum O-RAN COMMAG (CSV: rsrp, dl_snr, dl_mcs, dl_brate, ul_mcs)

All data is loaded from real measurement campaigns — no synthetic generation.
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


class Unified5GDataPipeline:
    """
    Unified data loader for multiple real 5G dataset sources.

    Normalizes CSV data into (T, num_features) numpy arrays with
    consistent feature ordering and z-score normalization.
    """

    # Standard feature set (output order)
    STANDARD_FEATURES = ["rsrp", "rsrq", "snr", "cqi", "rssi"]

    # Column mappings per dataset
    UCC_MISL_COLUMNS = {
        "RSRP": "rsrp", "RSRQ": "rsrq", "SNR": "snr",
        "CQI": "cqi", "RSSI": "rssi",
    }

    COLOSSEUM_COLUMNS = {
        "rsrp": "rsrp", "dl_snr": "snr",
        "dl_mcs": "cqi",  # Map MCS to CQI slot
        "dl_brate": "rssi",  # Map bitrate to RSSI slot
        "ul_mcs": "rsrq",  # Map UL MCS to RSRQ slot
    }

    def __init__(
        self,
        data_dirs: Optional[Dict[str, str]] = None,
        max_traces_per_source: int = 200,
        min_trace_length: int = 50,
    ):
        """
        Args:
            data_dirs: Dict mapping source name to directory path.
                       e.g. {"ucc_misl": "data/ucc_misl/extracted/5G-production-dataset",
                              "colosseum": "data/openran_gym/colosseum"}
            max_traces_per_source: Max traces to load per source (memory control).
            min_trace_length: Skip traces shorter than this.
        """
        self.data_dirs = data_dirs or {}
        self.max_traces = max_traces_per_source
        self.min_length = min_trace_length

        self._traces: List[np.ndarray] = []
        self._source_labels: List[str] = []
        self._stats: Dict[str, Dict[str, float]] = {}

    def load_all(self) -> List[np.ndarray]:
        """Load and normalize all datasets."""
        all_values = {f: [] for f in self.STANDARD_FEATURES}

        # First pass: collect values for normalization
        raw_traces = []
        for source, path in self.data_dirs.items():
            # TelecomTS can load from HuggingFace (no local path needed)
            if source == "telecomts":
                traces = self._load_telecomts(path)
            elif not os.path.exists(path):
                print(f"[DataPipeline] Skipping {source}: {path} not found")
                continue
            elif source == "ucc_misl":
                traces = self._load_ucc_misl(path)
            elif source == "colosseum":
                traces = self._load_colosseum(path)
            else:
                print(f"[DataPipeline] Unknown source: {source}")
                continue

            for trace, label in traces:
                raw_traces.append((trace, label))
                for i, feat in enumerate(self.STANDARD_FEATURES):
                    col = trace[:, i]
                    valid = col[~np.isnan(col)]
                    if len(valid) > 0:
                        all_values[feat].extend(valid.tolist())

        # Compute normalization stats
        for feat in self.STANDARD_FEATURES:
            vals = np.array(all_values[feat]) if all_values[feat] else np.array([0.0])
            self._stats[feat] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()) if vals.std() > 0 else 1.0,
            }

        # Second pass: normalize
        self._traces = []
        self._source_labels = []
        for trace, label in raw_traces:
            normalized = np.zeros_like(trace)
            for i, feat in enumerate(self.STANDARD_FEATURES):
                normalized[:, i] = (trace[:, i] - self._stats[feat]["mean"]) / self._stats[feat]["std"]
            # Replace NaN with 0
            normalized = np.nan_to_num(normalized, nan=0.0)
            self._traces.append(normalized.astype(np.float32))
            self._source_labels.append(label)

        total_rows = sum(len(t) for t in self._traces)
        print(f"[DataPipeline] Loaded {len(self._traces)} traces, "
              f"{total_rows:,} measurements from {len(self.data_dirs)} sources")
        return self._traces

    def to_gpu_tensor(self, device: torch.device) -> torch.Tensor:
        """Concatenate all traces into a single GPU tensor."""
        if not self._traces:
            self.load_all()
        combined = np.concatenate(self._traces, axis=0)
        return torch.from_numpy(combined).to(device)

    def _load_ucc_misl(self, path: str) -> List[Tuple[np.ndarray, str]]:
        """Load UCC MISL 5G dataset CSVs."""
        csv_files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
        csv_files = [f for f in csv_files if "MACOSX" not in f][:self.max_traces]

        traces = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                features = np.full((len(df), len(self.STANDARD_FEATURES)), np.nan, dtype=np.float32)

                for src_col, dst_feat in self.UCC_MISL_COLUMNS.items():
                    if src_col in df.columns:
                        idx = self.STANDARD_FEATURES.index(dst_feat)
                        vals = pd.to_numeric(df[src_col], errors="coerce").values
                        features[:, idx] = vals.astype(np.float32)

                if len(features) >= self.min_length:
                    traces.append((features, f"ucc_misl:{os.path.basename(f)}"))
            except Exception:
                continue

        print(f"[DataPipeline] UCC MISL: {len(traces)} traces from {path}")
        return traces

    def _load_colosseum(self, path: str) -> List[Tuple[np.ndarray, str]]:
        """Load Colosseum O-RAN COMMAG dataset CSVs."""
        csv_files = sorted(glob.glob(os.path.join(path, "**", "ue*.csv"), recursive=True))
        csv_files = csv_files[:self.max_traces]

        traces = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                features = np.full((len(df), len(self.STANDARD_FEATURES)), np.nan, dtype=np.float32)

                for src_col, dst_feat in self.COLOSSEUM_COLUMNS.items():
                    if src_col in df.columns:
                        idx = self.STANDARD_FEATURES.index(dst_feat)
                        vals = pd.to_numeric(df[src_col], errors="coerce").values
                        features[:, idx] = vals.astype(np.float32)

                if len(features) >= self.min_length:
                    traces.append((features, f"colosseum:{os.path.basename(f)}"))
            except Exception:
                continue

        print(f"[DataPipeline] Colosseum: {len(traces)} traces from {path}")
        return traces

    # Column mapping for TelecomTS (HuggingFace: AliMaatouk/TelecomTS)
    # Real 5G testbed data at 10 Hz with 18 KPIs per 128-step trace
    TELECOMTS_COLUMNS = {
        "RSRP": "rsrp",
        "UL_SNR": "snr",
        "DL_MCS": "cqi",       # MCS → CQI slot (correlated)
        "PRB_Utilization_DL": "rssi",  # DL utilization → RSSI slot
        "UL_BLER": "rsrq",     # UL BLER → RSRQ slot (quality metric)
    }

    def _load_telecomts(self, path: str) -> List[Tuple[np.ndarray, str]]:
        """
        Load TelecomTS dataset from HuggingFace (AliMaatouk/TelecomTS).

        This is a REAL 5G testbed dataset with 32,000 samples, each containing
        128-timestep arrays of KPIs at 10 Hz (100ms resolution).

        Args:
            path: Either "huggingface" to download, or local cache directory.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("[DataPipeline] TelecomTS: 'datasets' package not installed, skipping")
            return []

        try:
            if path == "huggingface" or path == "AliMaatouk/TelecomTS":
                ds = load_dataset("AliMaatouk/TelecomTS", split="train")
            else:
                ds = load_dataset("AliMaatouk/TelecomTS", split="train",
                                  cache_dir=path)
        except Exception as e:
            print(f"[DataPipeline] TelecomTS: failed to load: {e}")
            return []

        traces = []
        import json
        max_samples = min(len(ds), self.max_traces)
        for i in range(max_samples):
            try:
                sample = ds[i]
                kpis = sample["KPIs"]
                if isinstance(kpis, str):
                    kpis = json.loads(kpis)

                # Each KPI is a list of 128 floats
                trace_len = len(kpis.get("RSRP", []))
                if trace_len < self.min_length:
                    continue

                features = np.full(
                    (trace_len, len(self.STANDARD_FEATURES)), np.nan, dtype=np.float32
                )
                for src_col, dst_feat in self.TELECOMTS_COLUMNS.items():
                    if src_col in kpis:
                        idx = self.STANDARD_FEATURES.index(dst_feat)
                        vals = kpis[src_col]
                        if isinstance(vals, list):
                            features[:len(vals), idx] = np.array(vals, dtype=np.float32)[:trace_len]

                # Extract anomaly label for metadata
                labels = sample.get("labels", {})
                if isinstance(labels, str):
                    labels = json.loads(labels)
                zone = labels.get("zone", "?")
                app  = labels.get("application", "?")
                label = f"telecomts:s{i}_z{zone}_{app}"

                traces.append((features, label))
            except Exception:
                continue

        print(f"[DataPipeline] TelecomTS: {len(traces)} traces (real 5G testbed, 10Hz)")
        return traces

    @property
    def num_features(self) -> int:
        return len(self.STANDARD_FEATURES)

    @property
    def normalization_stats(self) -> Dict[str, Dict[str, float]]:
        return self._stats
