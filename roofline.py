"""
Roofline Analysis for GPT-2 Inference on T4

The roofline model is the fundamental framework for understanding whether an
operation is memory-bound or compute-bound. It asks one question:

    Given how much data this operation moves, how fast *could* it run
    if the hardware were perfectly efficient?

If the answer is "faster than it actually runs", you have a compute problem.
If the answer is "about as fast as it actually runs", you have a memory problem.
The ridge point is where the two limits meet.

T4 specs:
    Memory bandwidth:  320 GB/s
    Compute (FP16 TC): 65 TFLOPS (Tensor Cores)
    Compute (FP32):    8.1 TFLOPS
    Ridge point (FP16): 65e12 / 320e9 = ~203 FLOPs/byte
    Ridge point (FP32): 8.1e12 / 320e9 = ~25 FLOPs/byte
"""

import argparse
import torch
import time
import math
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from typing import List, Optional

# ── T4 hardware limits ────────────────────────────────────────────────────────
T4_MEMORY_BANDWIDTH_GBS = 320.0          # GB/s
T4_TFLOPS_FP16_TENSOR   = 65.0          # TFLOPS (Tensor Cores, FP16)
T4_TFLOPS_FP32          = 8.1           # TFLOPS (CUDA cores, FP32)

# Ridge points: the arithmetic intensity (FLOPs/byte) where compute and memory
# limits are equal. Below this = memory-bound. Above = compute-bound.
RIDGE_POINT_FP16 = (T4_TFLOPS_FP16_TENSOR * 1e12) / (T4_MEMORY_BANDWIDTH_GBS * 1e9)
RIDGE_POINT_FP32 = (T4_TFLOPS_FP32 * 1e12) / (T4_MEMORY_BANDWIDTH_GBS * 1e9)


@dataclass
class RooflineResult:
    batch_size: int
    prompt_tokens: int
    max_new_tokens: int
    total_time_ms: float
    tokens_per_sec: float
    ms_per_token: float

    # From profiler
    addmm_flops: float       # Total MFLOPs for aten::addmm
    addmm_cuda_ms: float     # CUDA time for aten::addmm
    mm_flops: float          # Total MFLOPs for aten::mm (lm_head)
    mm_cuda_ms: float
    cat_cuda_ms: float       # KV cache concatenation cost
    bmm_flops: float         # Attention score computation FLOPs
    bmm_cuda_ms: float

    # Derived
    achieved_tflops_addmm: float       # TFLOPS actually achieved
    arithmetic_intensity_addmm: float  # FLOPs/byte
    bandwidth_util_pct: float          # % of T4's 320 GB/s


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    model.half()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def build_prompt(tokenizer, prompt: Optional[str] = None, prompt_tokens: Optional[int] = None):
    """
    Build a prompt for controlled benchmarking.

    - Default: use one short fixed prompt for a stable decode-focused baseline.
    - Optional: synthesize a prompt with an approximate token length to study
      context-length sensitivity without changing the workload semantics too much.
    """
    if prompt_tokens is not None:
        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be > 0")

        seed_ids = tokenizer.encode("The future of AI is ", add_special_tokens=False)
        repeats = math.ceil(prompt_tokens / len(seed_ids))
        prompt_ids = (seed_ids * repeats)[:prompt_tokens]
        return tokenizer.decode(prompt_ids)

    if prompt is not None:
        return prompt

    return "The future of AI is"


def run_profiled_inference(
    model,
    tokenizer,
    device,
    batch_size,
    max_new_tokens=50,
    prompt: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
):
    prompt = build_prompt(tokenizer, prompt=prompt, prompt_tokens=prompt_tokens)
    prompts = [prompt] * batch_size

    inputs = tokenizer(
        prompts, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    input_tokens = inputs["input_ids"].shape[1]

    # Warmup
    with torch.no_grad():
        _ = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed + profiled run
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_flops=True,
    ) as prof:
        with record_function("model_inference"):
            start = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id
                )
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

    total_ms = (end - start) * 1000
    tokens_generated = max_new_tokens * batch_size

    # ── Extract profiler stats ────────────────────────────────────────────────
    events = {e.key: e for e in prof.key_averages()}

    def get(key, attr, default=0.0):
        e = events.get(key)
        return getattr(e, attr, default) if e else default

    addmm_flops = get("aten::addmm", "flops") / 1e6    # MFLOPs
    addmm_ms    = get("aten::addmm", "cuda_time_total") / 1e3  # ms
    mm_flops    = get("aten::mm",    "flops") / 1e6
    mm_ms       = get("aten::mm",    "cuda_time_total") / 1e3
    cat_ms      = get("aten::cat",   "cuda_time_total") / 1e3
    bmm_flops   = get("aten::bmm",   "flops") / 1e6
    bmm_ms      = get("aten::bmm",   "cuda_time_total") / 1e3

    # ── Arithmetic intensity for the linear (addmm) operations ───────────────
    #
    # For a batched linear layer: input [B, K] × weight [K, N] → output [B, N]
    # FLOPs  = 2 * B * K * N
    # Bytes  = 2 * (B*K + K*N + B*N)   (FP16 = 2 bytes/element)
    #
    # At B=1: bytes ≈ 2*K*N (weight dominates), AI ≈ 1 FLOP/byte
    # At B=8: bytes ≈ 2*K*N (weight still dominates), AI ≈ 8 FLOPs/byte
    # The weight matrix bytes don't change with batch size — that's why batching
    # helps: you amortize the same weight load over more computation.
    #
    # Estimated AI for the dominant projection (c_attn: K=768, N=2304):
    K, N = 768, 2304
    ai_addmm = (2 * batch_size * K * N) / (2 * (batch_size * K + K * N + batch_size * N))

    # Achieved throughput
    achieved_tflops = (addmm_flops * 1e6) / (addmm_ms * 1e-3) / 1e12 if addmm_ms > 0 else 0

    # Bandwidth utilization estimate:
    # If AI ≈ 1 FLOP/byte (batch=1), then bytes moved ≈ FLOPs.
    # More precisely: bytes = FLOPs / AI
    bytes_moved_gb = (addmm_flops * 1e6) / ai_addmm / 1e9
    bw_util_pct = (bytes_moved_gb / (addmm_ms * 1e-3)) / T4_MEMORY_BANDWIDTH_GBS * 100

    return RooflineResult(
        batch_size=batch_size,
        prompt_tokens=input_tokens,
        max_new_tokens=max_new_tokens,
        total_time_ms=total_ms,
        tokens_per_sec=tokens_generated / (total_ms / 1000),
        ms_per_token=total_ms / max_new_tokens,
        addmm_flops=addmm_flops,
        addmm_cuda_ms=addmm_ms,
        mm_flops=mm_flops,
        mm_cuda_ms=mm_ms,
        cat_cuda_ms=cat_ms,
        bmm_flops=bmm_flops,
        bmm_cuda_ms=bmm_ms,
        achieved_tflops_addmm=achieved_tflops,
        arithmetic_intensity_addmm=ai_addmm,
        bandwidth_util_pct=bw_util_pct,
    )


def print_roofline_analysis(results: List[RooflineResult]):
    print("\n" + "=" * 80)
    print("ROOFLINE ANALYSIS — GPT-2 on T4")
    print("=" * 80)

    print(f"""
T4 Hardware Limits:
  Memory bandwidth:   {T4_MEMORY_BANDWIDTH_GBS:.0f} GB/s
  Compute (FP16 TC):  {T4_TFLOPS_FP16_TENSOR:.0f} TFLOPS
  Compute (FP32):     {T4_TFLOPS_FP32:.1f} TFLOPS
  Ridge point (FP16): {RIDGE_POINT_FP16:.0f} FLOPs/byte   ← need this AI to saturate Tensor Cores
  Ridge point (FP32): {RIDGE_POINT_FP32:.1f} FLOPs/byte
""")

    # ── Throughput table ──────────────────────────────────────────────────────
    print("─" * 80)
    print("THROUGHPUT")
    print("─" * 80)
    print(f"Prompt tokens: {results[0].prompt_tokens} | Generated tokens/request: {results[0].max_new_tokens}")
    print(f"{'Batch':>6}  {'tokens/sec':>12}  {'ms/token':>10}  {'total_ms':>10}")
    print(f"{'─'*6}  {'─'*12}  {'─'*10}  {'─'*10}")
    for r in results:
        print(f"{r.batch_size:>6}  {r.tokens_per_sec:>12.1f}  {r.ms_per_token:>10.1f}  {r.total_time_ms:>10.1f}")

    # ── Roofline for linear projections (addmm) ───────────────────────────────
    print(f"\n{'─'*80}")
    print("LINEAR PROJECTIONS (aten::addmm) — where most GPU time goes")
    print("─" * 80)
    print(f"{'Batch':>6}  {'AI (F/B)':>10}  {'Achieved TF':>12}  {'% FP16 Peak':>12}  {'BW util%':>10}  {'Kernel type'}")
    print(f"{'─'*6}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*20}")
    kernel_names = {1: "GEMV (gemvx)", 4: "GEMM (Tensor Core)", 8: "GEMM (Tensor Core)"}
    for r in results:
        compute_pct = r.achieved_tflops_addmm / T4_TFLOPS_FP16_TENSOR * 100
        print(f"{r.batch_size:>6}  {r.arithmetic_intensity_addmm:>10.2f}  "
              f"{r.achieved_tflops_addmm:>12.4f}  {compute_pct:>12.3f}%  "
              f"{r.bandwidth_util_pct:>10.1f}%  {kernel_names.get(r.batch_size, '')}")

    print(f"""
What this means:
  • Arithmetic intensity for linear projections scales linearly with batch size.
    Batch=1 → ~1 F/byte. Batch=8 → ~8 F/byte. Ridge point is {RIDGE_POINT_FP16:.0f} F/byte.
    We're {RIDGE_POINT_FP16/results[0].arithmetic_intensity_addmm:.0f}x below the ridge at batch=1,
    and still {RIDGE_POINT_FP16/results[-1].arithmetic_intensity_addmm:.0f}x below it at batch=8.
  • Compute utilization is <1% even at batch=8. The Tensor Cores are almost idle.
  • The compute throughput grows roughly linearly with batch size (good — batching works),
    but we'd need batch ~{int(RIDGE_POINT_FP16)//2} to actually hit the ridge point.
""")

    # ── KV cache cost ─────────────────────────────────────────────────────────
    print("─" * 80)
    print("KV CACHE (aten::cat) — HuggingFace's naive implementation")
    print("─" * 80)
    print(f"{'Batch':>6}  {'cat ms':>10}  {'% of CUDA time':>16}  {'Why this is bad'}")
    print(f"{'─'*6}  {'─'*10}  {'─'*16}  {'─'*40}")
    for r in results:
        # Total CUDA time ≈ addmm + mm + cat + bmm + other
        cuda_total = r.addmm_cuda_ms + r.mm_cuda_ms + r.cat_cuda_ms + r.bmm_cuda_ms
        cat_pct = r.cat_cuda_ms / cuda_total * 100 if cuda_total > 0 else 0
        print(f"{r.batch_size:>6}  {r.cat_cuda_ms:>10.1f}  {cat_pct:>16.1f}%  "
              f"allocates new tensor every token step")

    print(f"""
HuggingFace's KV cache uses aten::cat to append each new K/V vector to the
accumulated past. This allocates a brand-new tensor on every token step and
copies the entire cache into it. For a sequence of N tokens, that's:
  token 1:  copy 1 row
  token 2:  copy 2 rows
  ...
  token 50: copy 50 rows
  Total:    1+2+...+50 = 1275 copies = O(N²) memory traffic

A proper KV cache (Phase 3) pre-allocates a fixed buffer and writes one row
per token step — O(1) per step, O(N) total. The cat cost goes to ~zero.
""")

    # ── Attention (bmm) ───────────────────────────────────────────────────────
    print("─" * 80)
    print("ATTENTION SCORE COMPUTATION (aten::bmm)")
    print("─" * 80)
    print(f"{'Batch':>6}  {'bmm ms':>10}  {'bmm MFLOPs':>12}  {'AI (F/B)':>10}")
    print(f"{'─'*6}  {'─'*10}  {'─'*12}  {'─'*10}")
    for r in results:
        # For attention BMM: [B*H, S, D] × [B*H, D, S] where H=12, D=64, S=seq_len
        # At decode time S grows (past context). Arithmetic intensity: 2*B*H*S*D / (2*(B*H*S*D + B*H*S*D))
        # ≈ 0.5 — also memory bound, but smaller than linear layers
        ai_bmm = 0.5  # approximate for decode
        print(f"{r.batch_size:>6}  {r.bmm_cuda_ms:>10.2f}  {r.bmm_flops:>12.1f}  {ai_bmm:>10.1f}")

    # ── Text roofline diagram ─────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("ROOFLINE DIAGRAM (log scale, FP16 Tensor Core roof)")
    print("─" * 80)
    _print_ascii_roofline(results)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("─" * 80)
    print("SUMMARY")
    print("─" * 80)
    b1 = results[0]
    b8 = results[-1]
    speedup = b8.tokens_per_sec / b1.tokens_per_sec
    print(f"""
  Batch 1 → Batch {b8.batch_size} throughput gain: {speedup:.1f}x  ({b1.tokens_per_sec:.1f} → {b8.tokens_per_sec:.1f} tokens/sec)

  Why batching helps:
    Arithmetic intensity scales with batch size (same weights, more computation).
    At batch=1: GEMV kernels,  AI ≈ {b1.arithmetic_intensity_addmm:.1f} F/byte → {b1.bandwidth_util_pct:.0f}% bandwidth utilization
    At batch={b8.batch_size}: GEMM kernels, AI ≈ {b8.arithmetic_intensity_addmm:.1f} F/byte → {b8.bandwidth_util_pct:.0f}% bandwidth utilization
    Kernel type changes at batch ≥ 4 (GEMV → Tensor Core GEMM).

  What's still inefficient:
    1. KV cache: aten::cat reallocates on every token step (O(N²) traffic)
    2. Still far below Tensor Core peak — need batch ~{int(RIDGE_POINT_FP16)//2} to hit ridge point
    3. ~{b8.cat_cuda_ms:.0f}ms/{b8.addmm_cuda_ms:.0f}ms = {b8.cat_cuda_ms/b8.addmm_cuda_ms*100:.0f}% of linear-layer GPU time wasted on cache copies

  These are exactly what Phases 3 and 4 address.
""")


def _print_ascii_roofline(results: List[RooflineResult]):
    """
    Rough ASCII roofline plot.
    X axis: arithmetic intensity (log scale, FLOPs/byte)
    Y axis: performance (log scale, TFLOPS)
    """
    width, height = 70, 20
    x_min, x_max = math.log10(0.5), math.log10(500)   # 0.5 to 500 F/B
    y_min, y_max = math.log10(0.001), math.log10(100)  # 0.001 to 100 TFLOPS

    def to_col(ai):
        return int((math.log10(max(ai, 1e-9)) - x_min) / (x_max - x_min) * (width - 1))

    def to_row(tflops):
        return height - 1 - int((math.log10(max(tflops, 1e-12)) - y_min) / (y_max - y_min) * (height - 1))

    grid = [[' '] * width for _ in range(height)]

    # Draw roofline
    bw_tflops_at_ai = lambda ai: min(T4_MEMORY_BANDWIDTH_GBS * ai / 1e3, T4_TFLOPS_FP16_TENSOR)
    for col in range(width):
        ai = 10 ** (x_min + col / (width - 1) * (x_max - x_min))
        tflops = bw_tflops_at_ai(ai)
        row = to_row(tflops)
        if 0 <= row < height:
            grid[row][col] = '─' if ai >= RIDGE_POINT_FP16 else '/'

    # Plot ridge point
    ridge_col = to_col(RIDGE_POINT_FP16)
    if 0 <= ridge_col < width:
        grid[0][ridge_col] = '▼'

    # Plot results
    labels = {1: '①', 4: '④', 8: '⑧'}
    for r in results:
        col = to_col(r.arithmetic_intensity_addmm)
        row = to_row(r.achieved_tflops_addmm)
        if 0 <= col < width and 0 <= row < height:
            grid[row][col] = labels.get(r.batch_size, 'x')

    # Print with Y axis labels
    y_labels = [100, 10, 1, 0.1, 0.01, 0.001]
    label_rows = {to_row(y): f"{y:>6}" for y in y_labels}

    print(f"{'':>7}  TFLOPS (FP16)")
    for row_idx, row in enumerate(grid):
        label = label_rows.get(row_idx, '      ')
        print(f"{label} |{''.join(row)}|")

    print(f"{'':>7}  +{'─'*width}+")
    # X axis labels
    x_ticks = [1, 10, 100]
    x_line = [' '] * width
    for x in x_ticks:
        col = to_col(x)
        if 0 <= col < width:
            lbl = str(x)
            for i, ch in enumerate(lbl):
                if col + i < width:
                    x_line[col + i] = ch
    print(f"{'':>9}{''.join(x_line)}   FLOPs/byte")
    print(f"\n  ① batch=1  ④ batch=4  ⑧ batch=8  ▼ ridge point ({RIDGE_POINT_FP16:.0f} F/byte)")


def print_prompt_sweep_summary(sweep_results: List[RooflineResult], batch_size: int):
    print("\n" + "=" * 80)
    print("PROMPT LENGTH SWEEP")
    print("=" * 80)
    print(f"Batch size: {batch_size} | Generated tokens/request: {sweep_results[0].max_new_tokens}")
    print(f"{'Prompt toks':>12}  {'tokens/sec':>12}  {'ms/token':>10}  {'cat ms':>10}")
    print(f"{'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}")
    for result in sweep_results:
        print(
            f"{result.prompt_tokens:>12}  {result.tokens_per_sec:>12.1f}  "
            f"{result.ms_per_token:>10.1f}  {result.cat_cuda_ms:>10.1f}"
        )

    first = sweep_results[0]
    last = sweep_results[-1]
    print(
        f"\nShort prompt ({first.prompt_tokens} toks) -> long prompt ({last.prompt_tokens} toks): "
        f"{first.tokens_per_sec:.1f} -> {last.tokens_per_sec:.1f} tokens/sec, "
        f"{first.cat_cuda_ms:.1f} -> {last.cat_cuda_ms:.1f} ms of aten::cat."
    )
    print(
        "Use this sweep to separate a clean decode baseline from context-length effects. "
        "The controlled single-prompt benchmark is still useful; this just tells you how "
        "sensitive the results are to longer prefill/context."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Roofline analysis for GPT-2 inference on T4")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Batch sizes for the main roofline analysis",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Number of tokens to generate per request",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional explicit prompt string for the main analysis",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        help="Optional approximate prompt token length for the main analysis",
    )
    parser.add_argument(
        "--prompt-token-sweep",
        type=int,
        nargs="+",
        default=None,
        help="Optional prompt token lengths to sweep, e.g. --prompt-token-sweep 8 64 256",
    )
    parser.add_argument(
        "--sweep-batch-size",
        type=int,
        default=1,
        help="Batch size to use for the prompt-length sweep",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Loading model...")
    model, tokenizer, device = load_model()

    results = []
    for bs in args.batch_sizes:
        print(
            f"\nProfiling batch size {bs} "
            f"(prompt_tokens={'default' if args.prompt_tokens is None else args.prompt_tokens}, "
            f"max_new_tokens={args.max_new_tokens})..."
        )
        r = run_profiled_inference(
            model,
            tokenizer,
            device,
            batch_size=bs,
            max_new_tokens=args.max_new_tokens,
            prompt=args.prompt,
            prompt_tokens=args.prompt_tokens,
        )
        results.append(r)

    print_roofline_analysis(results)

    if args.prompt_token_sweep:
        sweep_results = []
        print("\nRunning prompt-length sweep...")
        for prompt_tokens in args.prompt_token_sweep:
            print(
                f"  Profiling batch size {args.sweep_batch_size} with ~{prompt_tokens} prompt tokens..."
            )
            sweep_results.append(
                run_profiled_inference(
                    model,
                    tokenizer,
                    device,
                    batch_size=args.sweep_batch_size,
                    max_new_tokens=args.max_new_tokens,
                    prompt_tokens=prompt_tokens,
                )
            )
        print_prompt_sweep_summary(sweep_results, batch_size=args.sweep_batch_size)

    # ── Fill in CODEX.md baseline table ──────────────────────────────────────
    if args.batch_sizes == [1, 4, 8] and args.prompt is None and args.prompt_tokens is None and args.max_new_tokens == 50:
        print("\n" + "=" * 80)
        print("BASELINE TABLE (copy into CODEX.md)")
        print("=" * 80)
        print(f"\n| Metric                   | {'Batch 1':>10} | {'Batch 4':>10} | {'Batch 8':>10} |")
        print(f"|{'─'*26}|{'─'*12}|{'─'*12}|{'─'*12}|")
        print(f"| tokens/sec               | {results[0].tokens_per_sec:>10.1f} | {results[1].tokens_per_sec:>10.1f} | {results[2].tokens_per_sec:>10.1f} |")
        print(f"| ms/token                 | {results[0].ms_per_token:>10.1f} | {results[1].ms_per_token:>10.1f} | {results[2].ms_per_token:>10.1f} |")
        print(f"| arith. intensity (F/byte)| {results[0].arithmetic_intensity_addmm:>10.2f} | {results[1].arithmetic_intensity_addmm:>10.2f} | {results[2].arithmetic_intensity_addmm:>10.2f} |")
        print(f"| achieved TFLOPS (addmm)  | {results[0].achieved_tflops_addmm:>10.4f} | {results[1].achieved_tflops_addmm:>10.4f} | {results[2].achieved_tflops_addmm:>10.4f} |")
        print(f"| % FP16 Tensor Core peak  | {results[0].achieved_tflops_addmm/T4_TFLOPS_FP16_TENSOR*100:>9.2f}% | {results[1].achieved_tflops_addmm/T4_TFLOPS_FP16_TENSOR*100:>9.2f}% | {results[2].achieved_tflops_addmm/T4_TFLOPS_FP16_TENSOR*100:>9.2f}% |")
        print(f"| bandwidth util % (addmm) | {results[0].bandwidth_util_pct:>9.1f}% | {results[1].bandwidth_util_pct:>9.1f}% | {results[2].bandwidth_util_pct:>9.1f}% |")
        print(f"| KV cache (cat) ms        | {results[0].cat_cuda_ms:>10.1f} | {results[1].cat_cuda_ms:>10.1f} | {results[2].cat_cuda_ms:>10.1f} |")
