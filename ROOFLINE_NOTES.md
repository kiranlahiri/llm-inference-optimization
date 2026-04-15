# Roofline Script Notes

This note captures two things:

1. what each function in `roofline.py` does
2. how to read the benchmark output in `output.txt`

The goal is to make the roofline script easier to revisit later without needing
the original chat context.

## What Skill This Segment Teaches

This segment is teaching performance reasoning for LLM inference.

PyTorch profiling is part of the workflow, but the deeper skill is learning how
to:

- connect model execution to profiler events
- connect profiler events to GPU hardware limits
- decide whether a workload is memory-bound or compute-bound
- explain why an optimization should help before implementing it

In short: this segment is about learning how to profile, explain, and
eventually optimize transformer decode on GPU.

## What `roofline.py` Is Doing

At a high level, `roofline.py` measures GPT-2 decode on a Tesla T4 and tries to
answer:

- how fast decode runs
- where time goes in the profiler
- whether the dominant work is memory-bound or compute-bound
- how batching changes arithmetic intensity
- how expensive Hugging Face's append-style KV cache is

It does not just benchmark speed. It tries to explain the speed.

## Hardware Limits Used By The Script

The script hard-codes Tesla T4 theoretical peak limits:

- memory bandwidth: `320 GB/s`
- FP16 Tensor Core peak: `65 TFLOPS`
- FP32 peak: `8.1 TFLOPS`

From those, it derives ridge points:

- FP16 ridge point: about `203 FLOPs/byte`
- FP32 ridge point: about `25 FLOPs/byte`

These are not measured live from the machine. They are theoretical reference
ceilings used for comparison in the roofline model.

## Function-By-Function Walkthrough

### `RooflineResult`

`RooflineResult` is a dataclass that stores the result of one benchmark run.

It includes:

- high-level throughput metrics like `tokens_per_sec`, `ms_per_token`, and `total_time_ms`
- profiler-derived values for `aten::addmm`, `aten::mm`, `aten::bmm`, and `aten::cat`
- derived roofline metrics like achieved TFLOPS, arithmetic intensity, and bandwidth utilization

This is the main container passed between the benchmarking and reporting code.

### `load_model()`

`load_model()` prepares the model and tokenizer for benchmarking.

It:

- selects `cuda` if available, otherwise `cpu`
- prints the device and GPU name
- loads the GPT-2 tokenizer and model from Hugging Face
- moves the model to the selected device
- switches the model to eval mode
- converts the model weights to FP16 with `model.half()`
- ensures the tokenizer has a pad token

It returns `(model, tokenizer, device)`.

This function is just setup, but it matters because the whole benchmark is
meant to reflect FP16 inference on a T4.

### `build_prompt(tokenizer, prompt=None, prompt_tokens=None)`

`build_prompt()` determines which prompt the benchmark should use.

It supports three cases:

- if `prompt_tokens` is provided, it creates a synthetic prompt with
  approximately that many tokens by repeating `"The future of AI is "`
- if `prompt` is provided, it uses that exact prompt
- otherwise it falls back to the default short prompt `"The future of AI is"`

This gives the script a stable default baseline while still allowing controlled
experiments with longer prompt lengths.

### `run_profiled_inference(...)`

`run_profiled_inference()` is the core measurement function.

It does the actual benchmark run and computes the roofline metrics.

Step by step, it:

- builds the benchmark prompt
- duplicates it `batch_size` times
- tokenizes the prompt batch and moves tensors to the device
- records the input prompt length
- runs one warmup `model.generate()` call
- runs a second `model.generate()` call under the PyTorch profiler
- measures total wall-clock time
- extracts profiler stats for:
  - `aten::addmm`
  - `aten::mm`
  - `aten::bmm`
  - `aten::cat`
- estimates arithmetic intensity for the dominant linear projections
- computes achieved TFLOPS for `aten::addmm`
- estimates bandwidth utilization from FLOPs and arithmetic intensity
- returns the results as a `RooflineResult`

This is the function where the real "roofline analysis" happens.

#### Why `aten::addmm` gets special focus

The script treats `aten::addmm` as the main representative linear layer because
GPT-2 spends a lot of time in linear projections during decode.

That makes it a useful proxy for answering:

- how much useful compute is happening
- how much data movement is required
- whether the workload is limited by memory traffic or compute throughput

### `print_roofline_analysis(results)`

`print_roofline_analysis()` formats the main benchmark results for humans.

It prints:

- the T4 hardware limits
- a throughput table
- a linear-layer roofline table for `aten::addmm`
- a KV-cache section for `aten::cat`
- an attention section for `aten::bmm`
- an ASCII roofline diagram
- a summary section in plain English

This is the main interpretation layer. It turns the raw measurements into a
systems story.

### `_print_ascii_roofline(results)`

`_print_ascii_roofline()` draws a text-based roofline chart.

It:

- maps arithmetic intensity to the x-axis on a log scale
- maps TFLOPS to the y-axis on a log scale
- draws the memory-bandwidth slope and compute ceiling
- marks the ridge point
- plots the measured batch-size points on the chart

It is only a visualization helper, but it makes the memory-bound vs
compute-bound picture much easier to see.

### `print_prompt_sweep_summary(sweep_results, batch_size)`

`print_prompt_sweep_summary()` prints the results of a prompt-length sweep.

It compares:

- prompt length
- tokens/sec
- ms/token
- `aten::cat` time

This helps separate a clean decode baseline from context-length effects. It is
there to answer questions like: how much slower does decode get as the prompt
gets longer?

### `parse_args()`

`parse_args()` defines the command-line interface.

It supports:

- `--batch-sizes`
- `--max-new-tokens`
- `--prompt`
- `--prompt-tokens`
- `--prompt-token-sweep`
- `--sweep-batch-size`

It returns the parsed arguments object for the main script block.

### `if __name__ == "__main__":`

The main block wires the whole script together.

It:

- parses command-line arguments
- loads the model
- runs `run_profiled_inference()` once for each requested batch size
- prints the main roofline analysis
- optionally runs a prompt-length sweep
- prints a markdown baseline table when the default benchmark settings are used

That final markdown table is meant to be copied into project notes like
`CODEX.md`.

## How To Read `output.txt`

The benchmark output tells one consistent story:

- GPT-2 decode on this T4 is strongly memory-bound
- batching helps throughput a lot
- Hugging Face's `aten::cat` KV-cache path is a meaningful inefficiency

## Section-By-Section Output Annotation

### Setup

`Loading model...`

The script is starting up and loading GPT-2.

`Device: cuda`

PyTorch found a GPU and will use it.

`GPU: Tesla T4`

This confirms that the run matches the T4 assumptions built into the script.

`Profiling batch size 1`, `4`, and `8`

These are the three benchmark runs. Each run uses the same model and prompt
shape, but changes the number of requests processed together.

### Hardware Limits

The hardware section restates the theoretical T4 limits:

- `320 GB/s` memory bandwidth
- `65 TFLOPS` FP16 Tensor Core peak
- ridge point `203 FLOPs/byte`

The key number here is the ridge point. If arithmetic intensity is well below
that value, the workload is memory-bound.

### Throughput Section

The throughput table shows:

- prompt length: `5` tokens
- generation length: `50` new tokens per request
- throughput and latency at batch sizes `1`, `4`, and `8`

The numbers in `output.txt` are:

- batch 1: `52.4 tokens/sec`, `19.1 ms/token`, `954.5 total_ms`
- batch 4: `202.2 tokens/sec`, `19.8 ms/token`, `989.1 total_ms`
- batch 8: `413.9 tokens/sec`, `19.3 ms/token`, `966.4 total_ms`

What to conclude:

- total runtime stays roughly the same across batch sizes
- throughput rises almost linearly with batch size
- per-request decode-step latency stays roughly flat

This means batching is effective. The GPU is doing much more useful work in
parallel without making each request dramatically slower.

Important note: in this script, `ms/token` behaves more like time per decode
step per request, not time per token across the whole batch. That is why
`ms/token` stays around `19 ms` while total throughput increases so much.

### Linear Projections Section

This section focuses on `aten::addmm`, which represents the main linear
projections in GPT-2 decode.

The values are:

- batch 1: `AI 1.00`, `0.1526 TFLOPS`, `0.235% peak`, `47.8% BW util`
- batch 4: `AI 3.97`, `0.4509 TFLOPS`, `0.694% peak`, `35.5% BW util`
- batch 8: `AI 7.89`, `0.8970 TFLOPS`, `1.380% peak`, `35.5% BW util`

What to conclude:

- arithmetic intensity increases with batch size, which is expected
- achieved TFLOPS also increases with batch size
- but even batch 8 is far below the FP16 ridge point of `203 FLOPs/byte`
- compute utilization remains tiny compared to the `65 TFLOPS` peak

This is the core roofline result:

- batch 1 is extremely memory-bound
- batch 8 is still memory-bound
- batching helps, but not enough to make decode compute-bound

The most important comparison is:

- batch 1: about `203x` below the ridge point
- batch 8: still about `26x` below the ridge point

That is why the script says the Tensor Cores are mostly idle.

### KV Cache Section

This section measures `aten::cat`, which is how Hugging Face grows the KV cache
in the profiled decode path.

The values are:

- batch 1: `7.3 ms`, `7.7%`
- batch 4: `12.8 ms`, `8.7%`
- batch 8: `17.1 ms`, `10.9%`

What to conclude:

- even at batch 1, non-trivial time is spent concatenating cache tensors
- that overhead grows with batch size
- by batch 8, more than a tenth of the simplified CUDA-time breakdown is going
  to cache concatenation

This is wasted overhead. It is not useful model math. It comes from repeatedly
allocating a bigger tensor and copying the old cache into it.

That is why the script highlights the `O(N^2)` data movement story:

- token 1 copies 1 row
- token 2 copies 2 rows
- ...
- token 50 copies 50 rows

The point of this section is to justify the next optimization target:
preallocate the KV cache and write into it directly.

### Attention Section

This section reports `aten::bmm`, which corresponds to attention score
computation.

The values are:

- batch 1: `6.53 ms`, `55.1 MFLOPs`, `AI 0.5`
- batch 4: `7.51 ms`, `220.4 MFLOPs`, `AI 0.5`
- batch 8: `9.39 ms`, `440.9 MFLOPs`, `AI 0.5`

What to conclude:

- the attention work scales up with batch size
- but the arithmetic intensity remains low
- attention is also memory-sensitive in this decode regime

So the low-intensity story is not limited to one kernel. Decode in general is
light on compute relative to bytes moved.

### ASCII Roofline Diagram

The ASCII roofline plot turns the tables into a picture.

The symbols mean:

- `①` = batch 1
- `④` = batch 4
- `⑧` = batch 8
- `▼` = ridge point

What to conclude:

- all three measured points are to the left of the ridge point
- left of the ridge means low arithmetic intensity
- therefore all three runs are in the memory-bound region

Batching moves the point up and to the right, which is good, but not far enough
to cross into a compute-bound regime.

### Summary Section

The summary condenses the whole experiment into a few main lessons:

1. Batch 1 to batch 8 gives a `7.9x` throughput gain, which means batching
   works very well for throughput.
2. Even at batch 8, the linear projections are still far below the ridge point,
   so decode remains memory-bound.
3. `aten::cat` KV-cache growth is large enough to be worth fixing next.

The summary line:

`~17ms/82ms = 21% of linear-layer GPU time wasted on cache copies`

is the strongest practical motivation for the next segment of the project.

## Three Main Conclusions To Remember

If everything else fades, keep these three conclusions:

1. Batching improves GPT-2 decode throughput a lot on the T4.
2. Even after batching, decode is still strongly memory-bound.
3. The naive append-style KV cache is a meaningful and fixable source of wasted work.

## Why This Naturally Leads To Segment 3

The roofline script gives a measurement-based reason to move on to manual decode
and explicit KV-cache control.

The next step is not just "write decode from scratch because it sounds cool."
It is:

- make the model execution legible
- own the decode loop directly
- replace append-style cache growth with a preallocated cache
- measure the impact

That is how this segment connects to the next one.
