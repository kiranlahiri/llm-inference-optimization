# LLM Inference Optimization

This repository is a GPU inference learning lab built around GPT-2 on a Tesla
T4.

The goal is not to build a polished app or a production serving system. The
goal is to understand inference deeply enough to read and eventually contribute
to projects like vLLM, FlashAttention, and CUTLASS.

The project guide lives in [CODEX.md](CODEX.md).

## What This Repo Is For

Each segment of the project should produce:

- a working implementation
- a measurement showing what changed
- an explanation of why it changed

The current center of gravity is:

- profiling GPT-2 decode
- understanding why decode is memory-bound
- building a manual GPT-2 decode path
- making KV cache behavior explicit
- later moving into Triton and CUDA kernels

The server/client scripts are still in the repo, but they are secondary. They
are supporting scaffolding for later systems work, not the main learning path.

## Current Roadmap

1. Roofline analysis of Hugging Face GPT-2 decode
2. Manual GPT-2 decode path in PyTorch
3. Preallocated KV cache
4. Triton fused attention kernel
5. CUDA GEMM: naive then tiled
6. Continuous batching and token scheduling

That order is deliberate: understand the computation first, then memory traffic,
then kernels, then serving systems.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Main Workflow

Right now, `roofline.py` is the main benchmark harness.

Use it to answer questions like:

- why is GPT-2 decode memory-bound?
- how does batch size change arithmetic intensity?
- how sensitive are results to prompt length?

### Controlled baseline

Use this as the canonical benchmark for before/after comparisons:

```bash
python3 roofline.py
```

This keeps the short fixed prompt and gives a stable, low-noise decode baseline.

### Prompt-length sweep

Use this to understand how context length changes performance:

```bash
python3 roofline.py --prompt-token-sweep 8 64 256 512 --sweep-batch-size 1
```

### Longer-context batch comparison

Use this when you want the main roofline analysis at a larger prompt length:

```bash
python3 roofline.py --prompt-tokens 256 --batch-sizes 1 4 8
```

## Immediate Next Milestone

The next real task is building a manual GPT-2 decode path without relying on
`model.generate()` in the critical path.

That work should make the following explicit:

- token and positional embeddings
- per-layer attention
- KV cache writes
- greedy next-token selection

The first goal is correctness and legibility. Optimization comes after the
computation is clear.

## Repo Notes

- [CODEX.md](CODEX.md) is the project roadmap and working guide.
- [roofline.py](roofline.py) is the current core benchmark script.
- [server.py](server.py), [client.py](client.py), and [benchmark.py](benchmark.py)
  are still useful later, but they are not the main path right now.

## Guiding Principle

Measure everything.

If a change does not come with a number and an explanation, it is not yet doing
the job this repo exists to do.
