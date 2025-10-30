# LLM Inference Optimization

Learning project to understand GPU inference optimization through building a batched inference server.

## Phase 1: Basic Server + Batching (Current)

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Start the inference server
python server.py
```

The server will:
- Load GPT-2 model (downloads ~500MB on first run)
- Start on http://localhost:8000
- Automatically batch requests that arrive within 50ms

### Testing Batching

```bash
# Terminal 1: Start server
python server.py

# Terminal 2: Run client tests
python client.py --mode benchmark

# Or test custom scenarios
python client.py --mode concurrent --num-requests 16 --delay-ms 10
```

### Key Concepts (Phase 1)

**Dynamic Batching:**
- Collects requests for max 50ms or until batch size (8) is reached
- Processes multiple requests in single GPU forward pass
- Amortizes kernel launch overhead and memory transfer costs

**Why Batching Helps:**
- GPT-2 is memory-bound for small batches (spending more time moving data than computing)
- Batching increases arithmetic intensity (compute/memory ratio)
- Better GPU utilization through parallelism

**Trade-offs:**
- Latency vs throughput: Individual requests wait for batches to form
- Padding overhead: All sequences must be same length in batch
- Memory: Larger batches need more VRAM

## Next Steps (Phase 2)

- Profile with PyTorch profiler and nsys
- Identify bottlenecks (memory bandwidth, compute, attention)
- Apply optimizations (Flash Attention, quantization, KV cache)
- Measure improvement with hardware counters

## Architecture Notes

From EECS 470 perspective:
- Think of batching like superscalar execution (ILP but for requests)
- GPU memory hierarchy: HBM (DRAM) → L2 → L1/Shared → Registers
- Memory-bound = spending cycles waiting for DRAM, like cache misses
- Goal: Keep compute units fed with data (maximize occupancy)
