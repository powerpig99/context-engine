"""Composer: trace what necessarily follows from multiple collapses. Produce resolution + shadows."""

from context_engine.config import COMPOSITION_TEMPERATURE, DEFAULT_MAX_TOKENS
from context_engine.providers.base import Provider

COMPOSITION_PROMPT = """\
You have been given {n} independent analyses of the same question. \
Each was produced independently with a different analytical framing. \
None could see the others during generation.

Your task is NOT to vote, average, or select the best one. \
Your task is to trace what NECESSARILY follows from all of them taken together.

Specifically:
1. What do all analyses establish that must be true regardless of framing?
2. Where do they disagree, and what does the disagreement reveal about the problem?
3. What does each analysis see that the others miss?

Produce two sections:

## Resolution
The most complete and accurate answer achievable by combining what all analyses establish. \
State your reasoning. Where analyses conflict, explain what the conflict reveals rather than \
picking a side.

## Shadows
What this resolution cannot see. What questions remain open. What assumptions had to be made. \
What would a further analysis need to examine? These are not errors—they are the boundaries \
of what these analyses could reach. Be specific and actionable.\
"""


def compose(
    question: str,
    collapses: list[dict],
    provider: Provider,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """
    Compose N collapses into a resolution + shadow report.

    Returns: {"resolution": str, "shadows": str, "raw_composition": str}
    """
    parts = [f"Original question: {question}\n"]
    for i, collapse in enumerate(collapses, 1):
        parts.append(f"--- Analysis {i} ---")
        parts.append(collapse["response"])
        parts.append("")

    raw = provider.generate(
        system_prompt=COMPOSITION_PROMPT.format(n=len(collapses)),
        user_prompt="\n".join(parts),
        temperature=COMPOSITION_TEMPERATURE,
        max_tokens=max_tokens,
    )

    resolution, shadows = _parse_sections(raw)

    return {
        "resolution": resolution,
        "shadows": shadows,
        "raw_composition": raw,
    }


def _parse_sections(text: str) -> tuple[str, str]:
    """Extract Resolution and Shadows sections from composition output."""
    lower = text.lower()

    res_markers = ["## resolution", "# resolution", "**resolution**", "resolution:"]
    shad_markers = ["## shadows", "# shadows", "**shadows**", "shadows:"]

    res_start = -1
    shad_start = -1

    for marker in res_markers:
        idx = lower.find(marker)
        if idx != -1:
            res_start = idx + len(marker)
            break

    for marker in shad_markers:
        idx = lower.find(marker)
        if idx != -1:
            shad_start = idx + len(marker)
            break

    if res_start != -1 and shad_start != -1:
        if res_start < shad_start:
            # Resolution comes first — find where it ends (at shadows header)
            for marker in shad_markers:
                idx = lower.find(marker)
                if idx != -1:
                    resolution = text[res_start:idx].strip()
                    break
            shadows = text[shad_start:].strip()
        else:
            # Shadows comes first (unusual but handle it)
            for marker in res_markers:
                idx = lower.find(marker)
                if idx != -1:
                    shadows = text[shad_start:idx].strip()
                    break
            resolution = text[res_start:].strip()
    elif res_start != -1:
        resolution = text[res_start:].strip()
        shadows = "(No shadows section produced)"
    elif shad_start != -1:
        resolution = text[:shad_start].strip()
        shadows = text[shad_start:].strip()
    else:
        # Model didn't follow format — entire output is the resolution
        resolution = text.strip()
        shadows = "(Model did not produce a separate shadows section)"

    return resolution, shadows
