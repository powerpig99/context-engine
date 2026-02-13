"""
RLM baseline test — verify we can run Recursive Language Models
with our local vllm-mlx server (Qwen3-8B-4bit).

Requires vllm-mlx running:
    vllm-mlx serve mlx-community/Qwen3-8B-4bit --port 8000 --continuous-batching
"""

from rlm import RLM
from rlm.logger import RLMLogger

logger = RLMLogger(log_dir="./logs")

rlm = RLM(
    backend="vllm",
    backend_kwargs={
        "model_name": "mlx-community/Qwen3-8B-4bit",
        "base_url": "http://localhost:8000/v1",
        "api_key": "not-needed",
    },
    environment="local",
    max_depth=1,
    max_iterations=10,
    logger=logger,
    verbose=True,
)

# Simple test first
print("=" * 60)
print("TEST 1: Simple computation")
print("=" * 60)
result = rlm.completion("What is 17 * 23? Use Python to compute it.")
print(f"\nResult: {result}")

print("\n" + "=" * 60)
print("TEST 2: Context-based question")
print("=" * 60)
result = rlm.completion(
    "How many words in the following text contain the letter 'e'? "
    "Text: 'The quick brown fox jumps over the lazy dog near the river edge'. "
    "Write Python code to count them."
)
print(f"\nResult: {result}")
