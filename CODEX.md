# LLM Inference Optimization — Project Guide

## What This Project Is

This repository is a GPU inference learning lab.

The goal is not to build a polished GPT-2 app or a production serving stack. The
goal is to understand modern inference optimization well enough to read, reason
about, and eventually contribute to projects like vLLM, FlashAttention, and
CUTLASS.

The output of each segment is:

- a working implementation
- a measurement that proves what changed
- a written explanation of why it changed

If a piece of work does not improve understanding of model execution, GPU
performance, kernel design, or scheduling, it is probably not on the critical
path.

---

## North Star

Build enough intuition and hands-on skill that the core ideas in:

- Hugging Face generation code
- vLLM scheduling and KV-cache management
- FlashAttention-style fused kernels
- CUTLASS tiled GEMM kernels

stop feeling magical.

---

## Hardware

- **GPU:** Tesla T4
- **VRAM:** 15.64 GB
- **Memory bandwidth:** ~320 GB/s
- **Compute:** 65 TFLOPS FP16 Tensor Core, 8.1 TFLOPS FP32
- **Ridge point (FP16):** ~203 FLOPs/byte
- **PyTorch:** 2.2.0+cu121, CUDA 12.1

The T4 is perfect for this project: small enough to force disciplined thinking,
real enough for the bottlenecks to matter.

---

## Guiding Principles

1. **Measure everything.**
   Every segment should end with a before/after number: `tokens/sec`,
   `ms/token`, achieved TFLOPS, bandwidth utilization, or GPU idle time.

2. **Stay close to the computation.**
   Prefer work that exposes how GPT-2 decode actually runs over work that adds
   application scaffolding.

3. **One bottleneck at a time.**
   Pick a bottleneck, explain it, change one thing, and re-measure.

4. **Use GPT-2 as a microscope.**
   The model is small enough to iterate quickly, but still rich enough to teach
   the same ideas that appear in larger inference systems.

5. **Serving is downstream of understanding.**
   The server code can be useful later, but it is not the center of gravity for
   this repository.

---

## Project Shape

There are two possible versions of this project:

- a batching/server project
- an inference optimization learning project

This repository is the second one.

The existing HTTP server and client code are now considered supporting
scaffolding or future side quests, not the main path.

---

## Learning Roadmap

| Segment | What you build | Core question | Primary metric |
|---|---|---|---|
| 2 ✓ | Roofline analysis of Hugging Face GPT-2 decode | Why is decode memory-bound? | Can explain profiler data with real numbers |
| 3 | Manual GPT-2 decode path in PyTorch | What is the model actually computing each token step? | Correct greedy decode + competitive `tokens/sec` |
| 3b | Preallocated KV cache | Why does cache layout matter so much for decode? | `aten::cat` time goes to ~0 |
| 4 | Triton fused attention kernel | Why do fused kernels reduce HBM traffic? | Better bandwidth utilization than unfused baseline |
| 5 | CUDA GEMM: naive then tiled | How does tiling increase arithmetic intensity? | Measured TFLOPS improvement |
| 6 | Continuous batching / token scheduler | How does a serving system keep the GPU busy? | Lower GPU idle time than static batching |

The ordering is deliberate:

- first understand transformer execution
- then understand memory traffic
- then fuse kernels
- then learn lower-level CUDA tiling
- only then return to serving and scheduling

That sequence best supports the long-term goal of contributing to projects like
vLLM and CUTLASS.

---

## Immediate Next Milestone

The next real task is **Segment 3: manual GPT-2 decode**.

Success means:

- no `model.generate()` in the critical path
- a small explicit decode loop you can read line by line
- greedy decoding that matches Hugging Face on the same prompt
- attention and KV-cache behavior that you can explain from first principles

Start by reusing Hugging Face weights while owning the decode logic yourself.
The goal is not to rewrite all of GPT-2 blindly. The goal is to make inference
legible.

---

## Segment 3 Plan

### 3.1 Build a readable decode baseline

Implement a minimal GPT-2 decode path that exposes:

- token embeddings
- positional embeddings
- per-layer attention
- MLP blocks
- residual connections
- final logits and greedy next-token selection

At first, correctness matters more than speed.

### 3.2 Make KV cache explicit

Implement two cache versions:

- **naive cache:** append-style cache to understand the data flow
- **preallocated cache:** fixed buffer indexed by token position

This gives a clean before/after optimization story and prepares you for vLLM
concepts later.

### 3.3 Benchmark it

For each version, measure:

- `tokens/sec`
- `ms/token`
- top GPU kernels
- `aten::cat` cost
- memory allocation behavior

### 3.4 Write down the result

At the end of Segment 3, you should be able to explain:

- why decode is memory-bound at small batch sizes
- why KV-cache append/copy behavior is harmful
- why preallocation helps even before any custom kernel work

---

## Segment 4 Plan

Write a fused attention kernel in Triton after the unfused path is fully
understood.

The point of this segment is not "use Triton because it is cool." The point is
to learn exactly which intermediate tensors disappear from HBM traffic when
attention is fused.

Success means you can compare:

- unfused PyTorch attention
- your Triton fused implementation
- profiler evidence showing where the gain came from

---

## Segment 5 Plan

Write a GEMM kernel in CUDA in two passes:

1. naive global-memory version
2. shared-memory tiled version

This is the bridge to CUTLASS intuition. The main lesson is not syntax. The
main lesson is reuse: load once from HBM, reuse many times from shared memory,
and raise arithmetic intensity.

---

## Segment 6 Plan

Return to serving only after the model and kernel path are clear.

This segment can include:

- continuous batching
- token-level scheduling
- variable-length sequence handling
- understanding the problem PagedAttention is trying to solve

This is where the server code becomes useful again, but now it sits on top of
real understanding instead of replacing it.

---

## What To Learn Alongside The Code

### Transformer internals

Be able to trace one GPT-2 decode step through:

- embedding lookup
- QKV projection
- attention score computation
- masking and softmax
- value mixing
- output projection
- MLP
- logits

### GPU performance model

Use these ideas constantly:

- arithmetic intensity
- roofline model
- bandwidth-bound vs compute-bound
- occupancy
- memory hierarchy: HBM, L2, shared memory, registers

### Kernel programming

Learn in this order:

- read profiler traces
- reason about tensor shapes and memory movement
- Triton for fused kernels
- CUDA for thread/block/shared-memory control

---

## Scope

**In scope:**

- GPT-2 decode and its bottlenecks
- profiling and roofline analysis
- KV-cache design
- fused attention
- CUDA GEMM
- continuous batching as a later systems exercise

**Out of scope for now:**

- production deployment
- web/API polish
- distributed inference
- giant models
- benchmark chasing without understanding

---

## Current Baseline (Segment 2)

The current Hugging Face baseline already shows the central lesson of the
project:

- batch-1 decode is deeply memory-bound
- batching raises arithmetic intensity but does not come close to the FP16 ridge
- `aten::cat` in the KV cache is an obvious target

Use that baseline as the reference point for every future segment.

| Metric                    | Batch 1 | Batch 4 | Batch 8 |
|---------------------------|---------|---------|---------|
| tokens/sec                |    51.1 |   196.9 |   398.3 |
| ms/token                  |    19.6 |    20.3 |    20.1 |
| arith. intensity (F/byte) |    1.00 |    3.97 |    7.89 |
| achieved TFLOPS (addmm)   |  0.1526 |  0.4508 |  0.8779 |
| % FP16 Tensor Core peak   |   0.23% |   0.69% |   1.35% |
| bandwidth util % (addmm)  |   47.8% |   35.5% |   34.8% |
| KV cache (cat) ms         |     7.3 |    12.8 |    17.4 |

---

## Weekly Operating Loop

For any segment, use the same loop:

1. write down the question you are trying to answer
2. measure the current baseline
3. implement the smallest useful change
4. profile again
5. explain the result in plain language

If you cannot explain why a change helped, the segment is not finished.

---

## Success Criteria For The Whole Project

This project is successful if, by the end, you can:

- explain why GPT-2 decode is memory-bound on a T4
- implement and benchmark a readable manual decode path
- explain why KV-cache design affects throughput
- explain why fused attention reduces HBM traffic
- explain how shared-memory tiling improves GEMM
- open source code from vLLM or CUTLASS and recognize the ideas, even if the
  engineering is much more sophisticated

That is the real milestone.
