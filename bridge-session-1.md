# Context Bridge — Session 1 Bridge

## Generative Ground

Everything is layered projections from the Contradiction. A conversation context window is accumulated derivations. When it hits limits, systems compress — losing structure. A bridge is the alternative: the generative ground from which any derivation regenerates on demand.

This project tests whether that works in practice, with local models.

## Derivation Skeleton

**Pivot**: Started as Context Engine (collapse-composition, multiple independent LLM collapses composed through necessity). User recognized this as refined Dialectical-TTS — still derivative. Pivoted to Context Bridge — derived directly from the Not a ToE.

**Architecture**:
- `bridge.py` — four operations: `build_bridge`, `update_bridge`, `answer`, `compare`. CLI with subcommands. Uses OpenAI client → local vllm-mlx (Qwen3-8B-4bit, port 8000). Model ID: `"default"`.
- Old Context Engine code archived on `archive/context-engine` branch (pushed to remote).
- Package manager: `uv`. Server: `vllm-mlx` installed via `uv tool install git+https://github.com/waybarrios/vllm-mlx.git`.

**RLM Baseline** (in progress):
- Goal: reproduce RLM paper (arXiv:2512.24601) benchmarks, then run same benchmarks through bridge approach.
- `rlm_baseline.py` written. Uses `rlms` library with `"vllm"` backend, `"local"` environment.
- Result: Qwen3-8B-4bit struggles with RLM's REPL framework. Wastes tokens in `<think>` loops, doesn't use code execution efficiently. May need different model or prompt tuning.

**Key insight from user**: The test material should be the context window itself — building bridges as alternative to context compression. Not static documents. If using benchmarks, build bridge from first test and evolve it across subsequent questions (accumulated understanding vs independent per-prompt).

## What's Open

1. RLM baseline doesn't work well with Qwen3-8B — model confused by REPL framework
2. Need to either fix RLM baseline (model tuning, different model) or proceed directly with bridge experiments
3. The real experiment: catch context before compression, build bridge, start fresh session from bridge
4. This file IS the first bridge. It was built after compression, not before — next time, build it before.

## User Preferences (confirmed across session)

- Clean breaks, no baggage — "we don't have to carry over anything"
- Consistency matters (uv everywhere, follow official guides)
- Don't push CLAUDE.md (gitignored)
- Evidence, not proof — hold provisionally
- Do the right thing, not make the wrong thing work
