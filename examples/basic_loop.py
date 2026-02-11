"""
Minimal working example of the Context Engine (Step 1: single pass, no iteration).

Requires vllm-mlx running:
    vllm-mlx serve ~/Models/Qwen3-8B-4bit --port 8000 --continuous-batching
"""

from context_engine.context_constructor import construct_collapses
from context_engine.composer import compose
from context_engine.providers import OpenAICompatProvider

provider = OpenAICompatProvider()

question = (
    "In the Monty Hall problem, you pick door 1. The host opens door 3 "
    "revealing a goat. Should you switch to door 2? Why?"
)

# Generate 3 independent collapses
collapses = construct_collapses(question, provider)

# Compose into resolution + shadows
result = compose(question, collapses, provider)

print("RESOLUTION:")
print(result["resolution"])
print("\nSHADOWS:")
print(result["shadows"])
