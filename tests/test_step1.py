"""
Step 1 validation tests.

Run with: pytest tests/test_step1.py -v -s

Requires vllm-mlx running locally:
    vllm-mlx serve mlx-community/Qwen3-8B-4bit --port 8000 --continuous-batching

Tests are a mix of:
- Structural assertions (does the pipeline produce the right shape of output?)
- Human-evaluable outputs (printed for manual inspection with -s flag)
"""

import pytest

from context_engine.composer import compose
from context_engine.context_constructor import FRAMINGS, construct_collapses
from context_engine.providers.openai_compat import OpenAICompatProvider


@pytest.fixture(scope="module")
def provider():
    return OpenAICompatProvider()


# --- Structural tests ---


class TestPipelineStructure:
    def test_collapses_count_and_diversity(self, provider):
        """Each collapse should be different (not identical strings)."""
        question = "What is 17 * 23?"
        collapses = construct_collapses(question, provider, num_collapses=3)

        assert len(collapses) == 3
        responses = [c["response"] for c in collapses]
        unique = set(responses)
        assert len(unique) >= 2, "Collapses produced identical outputs"

    def test_composition_has_resolution_and_shadows(self, provider):
        """Composer output must contain both sections."""
        question = "Should you switch doors in the Monty Hall problem?"
        collapses = construct_collapses(question, provider)
        result = compose(question, collapses, provider)

        assert "resolution" in result
        assert "shadows" in result
        assert len(result["resolution"]) > 50, "Resolution suspiciously short"
        assert len(result["shadows"]) > 20, "Shadows suspiciously short"

    def test_custom_framings(self, provider):
        """User-provided framings should work."""
        custom = [
            "Analyze this mathematically.",
            "Analyze this intuitively.",
        ]
        collapses = construct_collapses(
            "What is the probability of drawing two aces from a deck?",
            provider,
            num_collapses=2,
            framings=custom,
        )
        assert len(collapses) == 2
        assert collapses[0]["framing"] == custom[0]


# --- Logic puzzle evaluation (human-evaluable) ---

MONTY_HALL = (
    "In the Monty Hall problem, you pick door 1. "
    "The host, who knows what's behind all doors, opens door 3 revealing a goat. "
    "Should you switch to door 2? Explain your reasoning."
)

KNIGHTS_AND_KNAVES = (
    "On an island, knights always tell the truth and knaves always lie. "
    "You meet two people, A and B. "
    "A says: 'At least one of us is a knave.' "
    "What are A and B?"
)

OPEN_ENDED = (
    "What are the strongest arguments for and against the claim that "
    "consciousness is substrate-independent?"
)


class TestLogicPuzzles:
    """Print outputs for human evaluation. Run with -s flag."""

    @pytest.mark.parametrize(
        "name,question,expected_keyword",
        [
            ("monty_hall", MONTY_HALL, "switch"),
            ("knights_knaves", KNIGHTS_AND_KNAVES, "knight"),
        ],
    )
    def test_logic_puzzle(self, provider, name, question, expected_keyword):
        """Logic puzzles: composition should contain the correct answer keyword."""
        collapses = construct_collapses(question, provider)
        result = compose(question, collapses, provider)

        print(f"\n{'=' * 60}")
        print(f"PUZZLE: {name}")
        print(f"{'=' * 60}")
        for i, c in enumerate(collapses, 1):
            print(f"\n--- Collapse {i} ---")
            print(c["response"][:500])
        print(f"\n--- Resolution ---")
        print(result["resolution"])
        print(f"\n--- Shadows ---")
        print(result["shadows"])

        assert expected_keyword in result["resolution"].lower(), (
            f"Expected '{expected_keyword}' in resolution for {name}"
        )

    def test_open_ended(self, provider):
        """Open-ended question: shadows should identify genuine limitations."""
        collapses = construct_collapses(OPEN_ENDED, provider)
        result = compose(OPEN_ENDED, collapses, provider)

        print(f"\n{'=' * 60}")
        print("OPEN-ENDED: Consciousness substrate independence")
        print(f"{'=' * 60}")
        for i, c in enumerate(collapses, 1):
            print(f"\n--- Collapse {i} ---")
            print(c["response"][:500])
        print(f"\n--- Resolution ---")
        print(result["resolution"])
        print(f"\n--- Shadows ---")
        print(result["shadows"])

        assert len(result["shadows"]) > 100, (
            "Shadow report too short for open-ended question"
        )
