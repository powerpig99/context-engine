# Context Engine

**Token prediction is the mechanism. Context is everything else.**

A reasoning system derived from first principles. Not an improvement on prior architectures—a rebuild from the generative ground.

## The Principle

From the [Not a ToE](https://github.com/powerpig99/ontological-clarity):

> *Everything is layered projections of the infinite-dimensional orthogonal binary hyperspace from Nothing—the infinitely self-referencing Contradiction.*

Applied to language models, this reduces to four statements:

1. **One mechanism**: Next-token prediction. The collapse. Fixed and universal.
2. **One variable**: Context. What the collapse operates on. Everything else is context engineering.
3. **One principle**: Better context → more clarifying collapse. Always relative, always provisional.
4. **One limit**: The system is finite. The human provides what the system cannot generate.

Everything in this repo is derived from these four statements.

## What It Does

The Context Engine constructs contexts that a single forward pass cannot construct for itself, then composes the results into the most clarifying answer achievable—and reports what it couldn't see.

Three components:

- **Context Constructor**: Generates multiple independent collapses—different models, different framings, different context strategies—each producing a different projection of the problem.
- **Composer**: Traces what necessarily follows from all collapses combined. Produces a resolution and a shadow report (what the resolution can't see).
- **Precondition Tracer**: Finds the generative ground of any context so that more context always adds clarity, never noise.

These operate in a loop: construct → compose → check shadows → reframe if possible → stop when the system can't generate new framings → present to human.

## The Key Insight

More context is never detrimental in principle. What we call "noise" is signal orthogonal to the current question—a limitation of the system's interpretive capacity, not a property of the data. With the right generative ground, every additional piece of context is clarifying signal.

The models already have the patterns. All of training is already in the weights. We don't need to compress, chunk, or discard context. We need the right principle to trace the patterns that are already there.

## Installation

```bash
git clone https://github.com/powerpig99/context-engine.git
cd context-engine
pip install -e .
```

### Local Model Setup (Step 1)

Step 1 uses [vllm-mlx](https://github.com/waybarrios/vllm-mlx) to serve a local model via an OpenAI-compatible API. Install vllm-mlx and download a model:

```bash
pip install vllm-mlx
mlx_lm.convert --hf-path Qwen/Qwen3-8B -q --mlx-path ~/Models/Qwen3-8B-4bit
```

Start the inference server:

```bash
vllm-mlx serve ~/Models/Qwen3-8B-4bit --port 8000 --continuous-batching
```

### API Setup (Step 2+)

For cross-model collapses, configure API keys:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
```

## Usage

```bash
# Basic: run the loop on a question (requires vllm-mlx running on port 8000)
python -m context_engine "Your question here"

# Point to a different OpenAI-compatible server
python -m context_engine --base-url http://localhost:8000/v1 "Your question here"

# Control number of collapses and temperature
python -m context_engine --num-collapses 3 --temperature 0.7 "Your question here"

# Save full trace (all collapses + composition)
python -m context_engine --save output.json "Your question here"

# Verbose: print individual collapses before composition
python -m context_engine --verbose "Your question here"
```

## Development Steps

Each step is a working system. Each step's results inform the next.

| Step | What | Tests | Key Question |
|------|------|-------|-------------|
| 1 | Single model, 3 collapses, compose, shadows | Logic puzzles, open-ended | Does the loop structure itself produce value? |
| 2 | Cross-model collapses | Embedding distance, mutual information | Does model diversity produce genuinely different collapses? |
| 3 | Full iteration loop | Diminishing returns, stopping reliability | Does shadow-driven reframing improve results? |
| 4 | Precondition tracing | Document QA vs. chunking baselines | Does grounding context produce more clarity than treating it as data? |
| 5 | Human-in-the-loop interface | Human evaluation of shadow reports | Is the shadow report actionable for human intervention? |

No plan beyond Step 5. The architecture is itself a projection held provisionally.

## Lineage

This project derives from insights discovered through two prior implementations, both now archived:

- [Dialectical-TTS](https://github.com/powerpig99/Dialectical-TTS) — First exploration. Local reasoning engine on Apple Silicon using fixed Believer/Logician/Contrarian traces with recursive self-refinement. Proved the core claim: multiple independent collapses composed through necessity outperform single-pass inference. Limitation discovered: the specific trace roles were scaffolding, not the principle.

- [Recursive Dialectical Engine (RDE)](https://github.com/powerpig99/recursive-dialectical-engine) — Second iteration. Full multi-model, multi-provider implementation with orchestrator, arbiter, REPL context environment, benchmarks (OOLONG, S-NIAH), 10 ablation studies, training data pipeline, 291 tests. Validated that cross-model traces produce genuinely different collapses. Limitation discovered: the architecture accumulated improvements on the Dialectical-TTS frame rather than deriving from first principles. The engineering was sound; the foundation was inherited rather than generated.

Both projects were stopped not because they failed, but because the insights they produced pointed to a deeper starting point. The same move the Not a ToE makes: trace accumulated projections back to their generative ground, discard the projections, rebuild from the principle. The benchmarks, ablation harness, and test infrastructure from the RDE remain useful and are referenced in the Context Engine's evaluation steps.

Additionally informed by:

- [Recursive Language Models (Zhang et al.)](https://arxiv.org/abs/2512.24601) — recursive context externalization as an alternative to cramming everything into one attention pass
- [Not a ToE](https://github.com/powerpig99/ontological-clarity) — the generative ground from which this architecture derives. One line: everything else is derived.

## License
MIT
