"""
Throughput benchmark for inference server

Measures requests/second and tokens/second under load.
Useful for comparing different batch sizes and optimization strategies.
"""

import asyncio
import httpx
import time
import statistics
from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkResult:
    total_requests: int
    total_time_sec: float
    requests_per_sec: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_batch_size: float


async def benchmark_throughput(
    num_requests: int = 100,
    concurrent_workers: int = 10,
    prompt: str = "The future of AI is"
) -> BenchmarkResult:
    """
    Sustained throughput test with multiple concurrent workers.

    Args:
        num_requests: Total number of requests to send
        concurrent_workers: Number of concurrent clients
        prompt: Prompt to use for generation
    """
    print(f"\nRunning benchmark:")
    print(f"  Total requests: {num_requests}")
    print(f"  Concurrent workers: {concurrent_workers}")
    print(f"  Prompt: '{prompt}'\n")

    latencies = []
    batch_sizes = []
    completed = 0

    async def worker(client: httpx.AsyncClient, worker_id: int, num_tasks: int):
        """Single worker sending requests"""
        nonlocal completed

        for i in range(num_tasks):
            start = time.perf_counter()

            try:
                response = await client.post(
                    "http://localhost:8000/generate",
                    json={
                        "prompt": prompt,
                        "max_new_tokens": 50,
                        "temperature": 0.7,
                        "request_id": f"worker_{worker_id}_req_{i}"
                    },
                    timeout=30.0
                )

                result = response.json()
                latency = (time.perf_counter() - start) * 1000

                latencies.append(latency)
                batch_sizes.append(result["batch_size"])
                completed += 1

                # Progress indicator
                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{num_requests} requests completed")

            except Exception as e:
                print(f"  Error in worker {worker_id}: {e}")

    # Distribute work across workers
    tasks_per_worker = num_requests // concurrent_workers
    remainder = num_requests % concurrent_workers

    start_time = time.perf_counter()

    async with httpx.AsyncClient() as client:
        workers = []
        for worker_id in range(concurrent_workers):
            # Give extra tasks to first workers if there's a remainder
            num_tasks = tasks_per_worker + (1 if worker_id < remainder else 0)
            workers.append(worker(client, worker_id, num_tasks))

        await asyncio.gather(*workers)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Calculate statistics
    latencies.sort()
    result = BenchmarkResult(
        total_requests=len(latencies),
        total_time_sec=total_time,
        requests_per_sec=len(latencies) / total_time,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=latencies[len(latencies) // 2],
        p95_latency_ms=latencies[int(len(latencies) * 0.95)],
        p99_latency_ms=latencies[int(len(latencies) * 0.99)],
        avg_batch_size=statistics.mean(batch_sizes)
    )

    return result


def print_benchmark_results(result: BenchmarkResult):
    """Pretty print benchmark results"""
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"\nThroughput:")
    print(f"  Requests/sec:        {result.requests_per_sec:.2f}")
    print(f"  Total requests:      {result.total_requests}")
    print(f"  Total time:          {result.total_time_sec:.2f}s")

    print(f"\nLatency (ms):")
    print(f"  Average:             {result.avg_latency_ms:.2f}")
    print(f"  p50 (median):        {result.p50_latency_ms:.2f}")
    print(f"  p95:                 {result.p95_latency_ms:.2f}")
    print(f"  p99:                 {result.p99_latency_ms:.2f}")

    print(f"\nBatching:")
    print(f"  Avg batch size:      {result.avg_batch_size:.2f}")

    print(f"\n{'='*80}\n")


async def compare_concurrency_levels():
    """
    Compare throughput at different concurrency levels.

    This helps understand:
    - How batching scales with load
    - Optimal concurrency for max throughput
    - Where bottlenecks appear
    """
    print("\n" + "="*80)
    print("CONCURRENCY COMPARISON")
    print("="*80)

    concurrency_levels = [1, 5, 10, 20]
    num_requests = 50

    results = []

    for concurrency in concurrency_levels:
        print(f"\n--- Testing with {concurrency} concurrent workers ---")
        result = await benchmark_throughput(
            num_requests=num_requests,
            concurrent_workers=concurrency
        )
        results.append((concurrency, result))

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Concurrency':<15} {'Req/sec':<15} {'Avg Latency':<15} {'Avg Batch Size':<15}")
    print("-" * 80)

    for concurrency, result in results:
        print(f"{concurrency:<15} {result.requests_per_sec:<15.2f} "
              f"{result.avg_latency_ms:<15.2f} {result.avg_batch_size:<15.2f}")

    print("\nKey observations:")
    print("  - Higher concurrency → larger batches → better throughput")
    print("  - But latency increases (requests wait in queue)")
    print("  - This is the latency-throughput tradeoff")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark inference server")
    parser.add_argument(
        "--mode",
        choices=["single", "compare"],
        default="single",
        help="Benchmark mode"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests to send"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent workers"
    )

    args = parser.parse_args()

    if args.mode == "compare":
        asyncio.run(compare_concurrency_levels())
    else:
        result = asyncio.run(benchmark_throughput(
            num_requests=args.num_requests,
            concurrent_workers=args.concurrency
        ))
        print_benchmark_results(result)
