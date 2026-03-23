#!/usr/bin/env python3
"""
SpectrAI xApp gRPC Client — queries the running xApp for spectrum decisions.

Usage:
    # Start server first:
    python -m spectrai.xapp.server --config xapp_config.yaml &

    # Then run client:
    python scripts/xapp_client.py
    python scripts/xapp_client.py --host localhost --port 50051 --num_requests 100
"""

import argparse
import sys
import time

import grpc
import numpy as np

# Add proto path
sys.path.insert(0, "src")
from spectrai.proto import spectrai_pb2, spectrai_pb2_grpc


def run_health_check(stub):
    """Check xApp health status."""
    print("\n=== Health Check ===")
    request = spectrai_pb2.HealthRequest()
    response = stub.GetHealth(request)
    print(f"  Status:  {spectrai_pb2.HealthStatus.ServingStatus.Name(response.status)}")
    print(f"  Model:   {response.model_path}")
    print(f"  Backend: {response.backend}")
    print(f"  Version: {response.version}")
    print(f"  Uptime:  {response.uptime_seconds:.1f}s")
    return response


def run_predictions(stub, num_requests, sequence_length, num_channels, num_features):
    """Send prediction requests and measure latency."""
    print(f"\n=== Predictions ({num_requests} requests) ===")

    input_dim = num_channels * num_features
    latencies = []
    actions = []

    for i in range(num_requests):
        # Generate observation (simulating real 5G channel measurements)
        observation = np.random.randn(sequence_length * input_dim).astype(np.float32).tolist()

        request = spectrai_pb2.PredictRequest(
            observation=observation,
            sequence_length=sequence_length,
            input_dim=num_channels * num_features,
        )

        t0 = time.perf_counter()
        response = stub.Predict(request)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        actions.append(response.action)

        if i < 5 or i == num_requests - 1:
            probs = list(response.action_probs)
            top_prob = max(probs)
            print(f"  [{i+1:>4d}] Channel {response.action} "
                  f"(conf={top_prob:.3f}, latency={latency:.2f}ms, "
                  f"server_latency={response.latency_ms:.2f}ms)")

    latencies = np.array(latencies)
    actions = np.array(actions)

    print("\n  --- Latency Stats ---")
    print(f"  Mean:   {latencies.mean():.2f}ms")
    print(f"  P50:    {np.percentile(latencies, 50):.2f}ms")
    print(f"  P95:    {np.percentile(latencies, 95):.2f}ms")
    print(f"  P99:    {np.percentile(latencies, 99):.2f}ms")
    print(f"  Min:    {latencies.min():.2f}ms")
    print(f"  Max:    {latencies.max():.2f}ms")

    channel_counts = np.bincount(actions, minlength=num_channels)
    print("\n  --- Channel Distribution ---")
    for ch, count in enumerate(channel_counts):
        bar = "#" * int(count / num_requests * 50)
        print(f"  Ch {ch}: {count:>4d} ({count/num_requests*100:>5.1f}%) {bar}")

    return latencies, actions


def run_metrics(stub):
    """Get xApp metrics."""
    print("\n=== Metrics ===")
    request = spectrai_pb2.MetricsRequest()
    response = stub.GetMetrics(request)
    print(f"  Predictions:  {response.predictions_total}")
    print(f"  Collisions:   {response.collisions_total}")
    print(f"  Successes:    {response.successes_total}")
    print(f"  Mean latency: {response.inference_latency_mean_ms:.3f}ms")
    print(f"  P50 latency:  {response.inference_latency_p50_ms:.3f}ms")
    print(f"  P99 latency:  {response.inference_latency_p99_ms:.3f}ms")
    if response.prometheus_text:
        print(f"  Prometheus:   {len(response.prometheus_text)} bytes")
    return response


def main():
    parser = argparse.ArgumentParser(description="SpectrAI xApp gRPC Client")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--num_requests", type=int, default=100)
    parser.add_argument("--num_channels", type=int, default=10)
    parser.add_argument("--num_features", type=int, default=5)
    parser.add_argument("--sequence_length", type=int, default=16)
    args = parser.parse_args()

    target = f"{args.host}:{args.port}"
    print(f"SpectrAI xApp Client — connecting to {target}")

    channel = grpc.insecure_channel(target)
    stub = spectrai_pb2_grpc.SpectrAIStub(channel)

    # 1. Health check
    health = run_health_check(stub)

    # 2. Predictions
    latencies, actions = run_predictions(
        stub, args.num_requests,
        args.sequence_length, args.num_channels, args.num_features,
    )

    # 3. Metrics
    run_metrics(stub)

    print("\n=== Summary ===")
    print("  xApp Status:        SERVING")
    print(f"  Backend:            {health.backend}")
    print(f"  Requests Sent:      {args.num_requests}")
    print(f"  Mean Latency:       {latencies.mean():.2f}ms")
    print(f"  P99 Latency:        {np.percentile(latencies, 99):.2f}ms")
    print("  Near-RT RIC Budget: 10ms")
    print(f"  Status:             {'PASS' if np.percentile(latencies, 99) < 10 else 'REVIEW'}")

    channel.close()


if __name__ == "__main__":
    main()
