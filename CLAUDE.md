# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Context Engine is a Python reasoning system that constructs and composes multiple independent AI model outputs ("collapses") to achieve better reasoning than single-pass inference. Derived from first principles via the [Not a ToE](https://github.com/powerpig99/ontological-clarity) framework.

**Current state**: Specification complete, implementation not yet started. The repo contains two spec documents (`context-engine.md` and `context-engine-README.md`) and no code.

## Core Architecture

Three components operating in a loop (not a pipeline):

1. **Context Constructor** — Generates N independent collapses using different models/framings/context strategies. Independence between collapses is critical (no cross-influence during generation).
2. **Composer** — Traces what necessarily follows from all collapses. Produces a **resolution** (best achievable answer) and **shadows** (what the resolution can't see). Composition is by logical necessity, not voting or selection.
3. **Precondition Tracer** — Traces context to its generative ground (purpose, assumptions, constraints, domain principles) so more context adds clarity rather than noise.

Loop: Construct → Compose → Can system reframe based on shadows? → YES: iterate → NO: present resolution + shadows to human.

## Implementation Roadmap

Five incremental steps, each producing a testable working system:

| Step | Scope | Key Validation |
|------|-------|----------------|
| 1 | Single local model (Qwen3-8B via MLX), 3 collapses, compose, shadows | Does the loop structure itself produce value? |
| 2 | Add cross-model collapses (Anthropic, OpenAI, Google APIs) | Does model diversity produce genuinely different collapses? |
| 3 | Full iteration loop with shadow-driven reframing | Does iteration improve beyond first pass? |
| 4 | Precondition tracing for long-context problems | Does grounding context beat standard chunking? |
| 5 | Human-in-the-loop interface for shadow-based handoff | Are shadow reports actionable for humans? |

## Planned Commands

```bash
# Install (editable)
pip install -e .

# Basic usage
python -m context_engine "Your question here"
python -m context_engine --model ~/Models/qwen3-8b "Your question here"
python -m context_engine --models claude-sonnet,gpt-4o,gemini-2.5-pro "Your question here"
python -m context_engine --max-iterations 3 "Your question here"
python -m context_engine --save output.json "Your question here"
```

## Planned File Structure (Step 1)

```
context-engine/
├── engine/
│   ├── __init__.py
│   ├── config.py                # MODEL_DIR = ~/Models/, defaults
│   ├── context_constructor.py   # Generate N collapses
│   ├── composer.py              # Compose + produce shadows
│   └── providers/
│       ├── base.py              # Abstract provider
│       └── mlx_local.py         # Local MLX inference
├── tests/
│   └── test_step1.py
├── examples/
│   └── basic_loop.py
└── pyproject.toml
```

Step 2 adds: `engine/providers/{anthropic,openai,google,openrouter,router}.py`

## Key Design Principles

- **Derive from first principles, don't carry forward scaffolding.** Fixed trace roles (Believer/Logician/Contrarian), REPL abstractions, orchestrator components, and recursive sub-traces from prior work were intentionally discarded. They may re-emerge empirically but are not prescribed.
- **Each step must validate before building the next.** No step requires subsequent steps to be valuable.
- **Shadows are not errors** — they're the structural handoff surface for human intervention or further system iteration.
- **Independence preserves diversity.** Collapses must not see each other during generation.
- **Model diversity is the mechanism for context diversity.** Same model with different prompts shares weight geometry; different model families produce genuinely different projections.

## Lineage

Third iteration of a research direction. Prior work (both archived):
- **Dialectical-TTS**: Local reasoning on Apple Silicon with fixed traces. Proved core claim but roles were scaffolding.
- **RDE**: Full multi-model implementation, 291 tests, 10 ablation studies. Validated cross-model diversity but accumulated rather than derived.

The RDE's benchmarks, ablation harness, and test infrastructure are referenced for evaluation.

## Environment

- Local models stored in `~/Models/` (outside iCloud sync)
- API keys via environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`
- MLX for local inference (Apple Silicon optimized)
