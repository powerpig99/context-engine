"""Context Constructor: generate N independent collapses with different framings."""

from concurrent.futures import ThreadPoolExecutor

from context_engine.config import DEFAULT_NUM_COLLAPSES, DEFAULT_TEMPERATURE
from context_engine.providers.base import Provider

# Three default framings — different attention orientations, not roles.
# Each directs the model to construct different context from the same question.
FRAMINGS = [
    # Structural: what are the entities, relationships, and constraints?
    (
        "You are solving a problem. Focus on the formal structure: "
        "what are the entities, relationships, and constraints? "
        "Identify what logically follows from what. "
        "Be precise about what is given versus what is inferred."
    ),
    # Skeptical: where does intuition mislead?
    (
        "You are solving a problem. Focus on where intuition misleads: "
        "what assumptions feel obvious but might be wrong? "
        "Consider the problem from the perspective of common mistakes. "
        "Explicitly check each reasoning step for hidden assumptions."
    ),
    # Concrete: trace through specific cases
    (
        "You are solving a problem. Focus on making it concrete: "
        "enumerate specific cases, trace through examples, "
        "simulate the scenario step by step. "
        "Ground abstract claims in specific instances."
    ),
]


def construct_collapses(
    question: str,
    provider: Provider,
    num_collapses: int = DEFAULT_NUM_COLLAPSES,
    framings: list[str] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[dict]:
    """
    Generate N independent collapses for the given question.

    Returns list of dicts: [{"framing": str, "response": str}, ...]

    Collapses are generated concurrently to leverage vllm-mlx batching.
    Independence is structural: each is a separate API call with no shared context.
    """
    active_framings = framings or FRAMINGS[:num_collapses]

    def _generate_one(framing: str) -> dict:
        response = provider.generate(
            system_prompt=framing,
            user_prompt=question,
            temperature=temperature,
        )
        return {"framing": framing, "response": response}

    with ThreadPoolExecutor(max_workers=num_collapses) as pool:
        results = list(pool.map(_generate_one, active_framings))

    return results
