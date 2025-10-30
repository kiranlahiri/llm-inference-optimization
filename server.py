"""
LLM Inference Server with Dynamic Batching

Key concepts for optimization:
- GPU inference is memory-bound for small batches (memory bandwidth bottleneck)
- Batching amortizes fixed costs (kernel launch overhead, memory transfers)
- Goal: Maximize GPU utilization by processing multiple requests together
"""

import asyncio
import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Request/Response models
class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 1.0
    request_id: Optional[str] = None

class GenerationResponse(BaseModel):
    generated_text: str
    request_id: Optional[str] = None
    generation_time_ms: float
    batch_size: int

# Batching coordinator
class BatchingQueue:
    """
    Dynamic batching: Accumulate requests for a short time window,
    then process them as a single batch on GPU.

    Why this helps:
    - Single forward pass for N requests vs N forward passes
    - Better GPU utilization (more parallel work)
    - Reduced kernel launch overhead
    """

    def __init__(self, max_batch_size: int = 8, max_wait_ms: float = 50.0):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # Convert to seconds

        self.queue: List[tuple] = []  # (request, future)
        self.lock = asyncio.Lock()
        self.processing = False

    async def add_request(self, request: GenerationRequest) -> GenerationResponse:
        """Add request to queue and wait for result"""
        future = asyncio.Future()

        async with self.lock:
            self.queue.append((request, future))
            queue_size = len(self.queue)

        # Trigger processing if we hit batch size
        if queue_size >= self.max_batch_size:
            asyncio.create_task(self._process_batch())
        elif queue_size == 1:
            # First request in queue, start timer
            asyncio.create_task(self._wait_and_process())

        return await future

    async def _wait_and_process(self):
        """Wait for max_wait_ms, then process whatever we have"""
        await asyncio.sleep(self.max_wait_ms)
        await self._process_batch()

    async def _process_batch(self):
        """Process accumulated requests as a batch"""
        async with self.lock:
            if self.processing or not self.queue:
                return

            self.processing = True
            # Take up to max_batch_size requests
            batch = self.queue[:self.max_batch_size]
            self.queue = self.queue[self.max_batch_size:]

        try:
            # Extract requests and futures
            requests = [item[0] for item in batch]
            futures = [item[1] for item in batch]

            # Run inference (blocking call, runs in thread pool)
            responses = await asyncio.to_thread(
                inference_engine.generate_batch,
                requests
            )

            # Return results
            for future, response in zip(futures, responses):
                future.set_result(response)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)

        finally:
            async with self.lock:
                self.processing = False
                # If more requests accumulated, process them
                if self.queue:
                    asyncio.create_task(self._process_batch())


class InferenceEngine:
    """Handles model loading and batch inference"""

    def __init__(self, model_name: str = "gpt2"):
        logger.info(f"Loading model: {model_name}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Inference mode

        # Set pad token (GPT-2 doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model loaded successfully")

    def generate_batch(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        """
        Process multiple requests in a single batch.

        Performance considerations:
        - Padding: All sequences in batch must be same length (GPU parallelism requirement)
        - Attention mask: Tells model which tokens are padding vs real
        - Memory: Batch size limited by GPU VRAM
        """
        batch_size = len(requests)
        logger.info(f"Processing batch of {batch_size} requests")

        start_time = time.perf_counter()

        # Tokenize all prompts
        prompts = [req.prompt for req in requests]

        # Batch tokenization with padding
        inputs = self.tokenizer(
            prompts,
            padding=True,  # Pad to longest sequence in batch
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Get generation parameters (using first request's params for simplicity)
        max_new_tokens = requests[0].max_new_tokens
        temperature = requests[0].temperature

        # Generate (this is where the GPU does work)
        with torch.no_grad():  # Disable gradient computation (inference only)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True
        )

        end_time = time.perf_counter()
        generation_time_ms = (end_time - start_time) * 1000

        # Create responses
        responses = []
        for i, req in enumerate(requests):
            responses.append(GenerationResponse(
                generated_text=generated_texts[i],
                request_id=req.request_id,
                generation_time_ms=generation_time_ms / batch_size,  # Amortized time
                batch_size=batch_size
            ))

        logger.info(f"Batch processed in {generation_time_ms:.2f}ms "
                   f"({generation_time_ms/batch_size:.2f}ms per request)")

        return responses


# Initialize FastAPI app and components
app = FastAPI(title="LLM Inference Server")
inference_engine: Optional[InferenceEngine] = None
batching_queue: Optional[BatchingQueue] = None


@app.on_event("startup")
async def startup_event():
    """Initialize model and batching queue on startup"""
    global inference_engine, batching_queue

    inference_engine = InferenceEngine(model_name="gpt2")
    batching_queue = BatchingQueue(max_batch_size=8, max_wait_ms=50.0)

    logger.info("Server startup complete")


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    Generate text from prompt.

    This endpoint is async and non-blocking:
    - Request is added to batching queue
    - FastAPI can handle other requests while waiting
    - Multiple requests get batched together automatically
    """
    if batching_queue is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    try:
        response = await batching_queue.add_request(request)
        return response
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": inference_engine.device if inference_engine else "not initialized"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
