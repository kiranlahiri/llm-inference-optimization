"""
GPU Profiling Script for LLM Inference

This script runs the inference engine with PyTorch profiler enabled
to collect detailed GPU performance metrics.
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def profile_inference(batch_size=8, prompt="The future of AI is", max_new_tokens=50):
    """
    Profile a single batch inference to understand GPU performance.

    Args:
        batch_size: Number of requests to process in batch
        prompt: Input text prompt
        max_new_tokens: Number of tokens to generate
    """

    print("=" * 80)
    print("GPU PROFILING - LLM INFERENCE")
    print("=" * 80)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print(f"Batch size: {batch_size}")
    print(f"Prompt: '{prompt}'")
    print(f"Max new tokens: {max_new_tokens}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.to(device)
    model.eval()

    model.half()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully")

    # Prepare batch
    prompts = [prompt] * batch_size
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)


    print(f"\nInput shape: {inputs['input_ids'].shape}")

    # Warmup run (first run includes CUDA initialization overhead)
    print("\nRunning warmup...")
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    if device == "cuda":
        torch.cuda.synchronize()

    print("Warmup complete")

    # Profile the actual inference
    print("\n" + "=" * 80)
    print("PROFILING INFERENCE...")
    print("=" * 80)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device == "cuda" else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # Set to True for detailed stack traces (more overhead)
        with_flops=True,   # Estimate FLOPs
    ) as prof:
        with record_function("model_inference"):
            start_time = time.perf_counter()

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )

            if device == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000

    print("\n" + "=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)

    print(f"\nTotal inference time: {total_time_ms:.2f}ms")
    print(f"Time per request: {total_time_ms / batch_size:.2f}ms")

    if device == "cuda":
        print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"VRAM reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Top operations by CUDA time
    print("\n" + "-" * 80)
    print("TOP 15 OPERATIONS BY GPU TIME")
    print("-" * 80)

    if device == "cuda":
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=15,
            max_src_column_width=50
        ))
    else:
        print(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=15,
            max_src_column_width=50
        ))

    # Top operations by CPU time
    print("\n" + "-" * 80)
    print("TOP 10 OPERATIONS BY CPU TIME")
    print("-" * 80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=10,
        max_src_column_width=50
    ))

    # Memory operations
    if device == "cuda":
        print("\n" + "-" * 80)
        print("TOP 10 MEMORY-INTENSIVE OPERATIONS")
        print("-" * 80)
        print(prof.key_averages().table(
            sort_by="cuda_memory_usage",
            row_limit=10,
            max_src_column_width=50
        ))

    # Export detailed trace (can be viewed in chrome://tracing)
    trace_file = "profile_trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"\n✓ Detailed trace exported to: {trace_file}")
    print(f"  View it at: chrome://tracing (load the JSON file)")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    events = prof.key_averages()

    if device == "cuda":
        total_cuda_time = sum([e.cuda_time_total for e in events])
        total_cpu_time = sum([e.cpu_time_total for e in events])

        print(f"Total CUDA time: {total_cuda_time / 1000:.2f}ms")
        print(f"Total CPU time: {total_cpu_time / 1000:.2f}ms")

        # Find key operation categories
        matmul_time = sum([e.cuda_time_total for e in events if 'matmul' in e.key.lower() or 'gemm' in e.key.lower()])
        softmax_time = sum([e.cuda_time_total for e in events if 'softmax' in e.key.lower()])
        layernorm_time = sum([e.cuda_time_total for e in events if 'norm' in e.key.lower()])

        print(f"\nOperation breakdown:")
        print(f"  Matrix operations: {matmul_time / 1000:.2f}ms ({matmul_time / total_cuda_time * 100:.1f}%)")
        print(f"  Softmax (attention): {softmax_time / 1000:.2f}ms ({softmax_time / total_cuda_time * 100:.1f}%)")
        print(f"  Layer normalization: {layernorm_time / 1000:.2f}ms ({layernorm_time / total_cuda_time * 100:.1f}%)")

    print("\n" + "=" * 80)
    print("Next steps:")
    print("  1. Look for operations taking most GPU time")
    print("  2. Check if using FP32 (should see 'float' in operations)")
    print("  3. Try FP16 to utilize T4 Tensor Cores")
    print("  4. Examine memory usage patterns")
    print("=" * 80)


if __name__ == "__main__":
    # Profile with different batch sizes
    batch_sizes = [1, 4, 8]

    for bs in batch_sizes:
        profile_inference(batch_size=bs)
        print("\n\n")
