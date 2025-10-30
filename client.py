"""
Test client to demonstrate batching behavior

This script sends concurrent requests to observe:
1. How batching improves throughput
2. Individual request latency vs total throughput
3. Batching effectiveness
"""

import asyncio
import httpx
import time
from typing import List
import argparse


async def send_request(client: httpx.AsyncClient, prompt: str, request_id: int):
    """Send a single generation request"""
    start_time = time.perf_counter()

    response = await client.post(
        "http://localhost:8000/generate",
        json={
            "prompt": prompt,
            "max_new_tokens": 50,
            "temperature": 0.7,
            "request_id": f"req_{request_id}"
        },
        timeout=30.0
    )

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    result = response.json()
    return {
        "request_id": request_id,
        "end_to_end_latency_ms": latency_ms,
        "server_processing_ms": result["generation_time_ms"],
        "batch_size": result["batch_size"],
        "generated_text": result["generated_text"][:100] + "..."  # Truncate for display
    }


async def run_concurrent_requests(num_requests: int, delay_ms: float = 0):
    """
    Send multiple requests concurrently to test batching.

    Args:
        num_requests: Number of requests to send
        delay_ms: Delay between request starts (0 = all at once)
    """
    print(f"\n{'='*80}")
    print(f"Sending {num_requests} concurrent requests (delay={delay_ms}ms between starts)")
    print(f"{'='*80}\n")

    # Sample prompts
    prompts = [
        "Once upon a time in a distant galaxy",
        "The future of artificial intelligence is",
        "In the world of quantum computing",
        "The most interesting thing about neural networks is",
        "When considering GPU optimization, we must think about",
        "Machine learning models are becoming",
        "The key to understanding transformers is",
        "Modern computer architecture relies on"
    ]

    async with httpx.AsyncClient() as client:
        start_time = time.perf_counter()

        # Create tasks
        tasks = []
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            tasks.append(send_request(client, prompt, i))

            # Optional delay between request starts
            if delay_ms > 0 and i < num_requests - 1:
                await asyncio.sleep(delay_ms / 1000.0)

        # Wait for all to complete
        results = await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

    # Analysis
    print("\nResults:")
    print(f"{'Request ID':<12} {'E2E Latency':<15} {'Server Time':<15} {'Batch Size':<12}")
    print("-" * 80)

    batch_sizes = []
    e2e_latencies = []

    for r in results:
        print(f"{r['request_id']:<12} {r['end_to_end_latency_ms']:<15.2f} "
              f"{r['server_processing_ms']:<15.2f} {r['batch_size']:<12}")
        batch_sizes.append(r['batch_size'])
        e2e_latencies.append(r['end_to_end_latency_ms'])

    # Summary statistics
    print(f"\n{'='*80}")
    print("Summary:")
    print(f"  Total time: {total_time_ms:.2f}ms")
    print(f"  Throughput: {num_requests / (total_time_ms / 1000):.2f} requests/sec")
    print(f"  Average E2E latency: {sum(e2e_latencies) / len(e2e_latencies):.2f}ms")
    print(f"  Average batch size: {sum(batch_sizes) / len(batch_sizes):.2f}")
    print(f"  Max batch size seen: {max(batch_sizes)}")
    print(f"\nKey insight:")
    if max(batch_sizes) > 1:
        print(f"  ✓ Batching is working! Requests were grouped together.")
        print(f"  ✓ Each request in a batch shares the GPU compute cost.")
    else:
        print(f"  ⚠ No batching occurred. Try sending requests faster or closer together.")
    print(f"{'='*80}\n")


async def benchmark_batching_vs_sequential():
    """
    Compare batched processing vs sequential processing.

    This demonstrates the throughput benefit of batching.
    """
    num_requests = 8

    print("\n" + "="*80)
    print("BENCHMARK: Batched vs Sequential Processing")
    print("="*80)

    # Test 1: All requests at once (should batch)
    print("\nTest 1: All requests sent simultaneously")
    await run_concurrent_requests(num_requests, delay_ms=0)

    # Test 2: Spread out requests (partial batching)
    print("\nTest 2: Requests spread over 100ms")
    await run_concurrent_requests(num_requests, delay_ms=15)

    # Test 3: Sequential (no batching expected)
    print("\nTest 3: Sequential requests (200ms apart)")
    await run_concurrent_requests(num_requests, delay_ms=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLM inference server")
    parser.add_argument(
        "--mode",
        choices=["concurrent", "benchmark"],
        default="concurrent",
        help="Test mode: concurrent (single test) or benchmark (compare scenarios)"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=8,
        help="Number of concurrent requests to send"
    )
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=0,
        help="Delay in ms between request starts"
    )

    args = parser.parse_args()

    if args.mode == "benchmark":
        asyncio.run(benchmark_batching_vs_sequential())
    else:
        asyncio.run(run_concurrent_requests(args.num_requests, args.delay_ms))
