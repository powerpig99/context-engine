# Context Bridge vs RLM: OOLONG Benchmark Findings

## Setup

- **Model**: Qwen3-8B-4bit (quantized, local via vllm-mlx on Apple Silicon)
- **Benchmark**: OOLONG-synth trec_coarse — semantic aggregation over general-knowledge questions
- **Tasks**: 30 balanced tasks (10 each at 1024, 2048, 4096 token contexts)
- **OOLONG task types**: MOST_FREQ, LEAST_FREQ, RELATIVE_FREQ (counting/comparing label distributions)
- **Scoring**: OOLONG-style — extract answer after last colon, case-insensitive exact match for labels

## Results

### Four-Way Comparison (30 tasks)

| Approach | Exact Match | Avg Score | Time/Task | Context Used |
|----------|-------------|-----------|-----------|--------------|
| Base (direct prompt) | 0/30 = 0.0% | 0.000 | 12s | 100% |
| RLM (with REPL) | 4/30 = 13.3% | 0.133 | 114s | 100% + code exec |
| Full context (/no_think) | 9/30 = 30.0% | 0.300 | 6s | 100% |
| **Bridge (6% of ctx)** | **11/30 = 36.7%** | **0.367** | **3s** | **6%** |

### By Context Length

| Context | Bridge | Full ctx | RLM | Base |
|---------|--------|----------|-----|------|
| 1024 | 6/10 = 60% | 4/10 = 40% | 1/10 = 10% | 0/10 = 0% |
| 2048 | 0/10 = 0% | 0/10 = 0% | 2/10 = 20% | 0/10 = 0% |
| 4096 | 5/10 = 50% | 5/10 = 50% | 1/10 = 10% | 0/10 = 0% |

### Comparison to RLM Paper (arXiv:2512.24601)

The paper reports on **full Qwen3-8B** (not quantized) at **132K tokens**:

| Approach | Paper (Qwen3-8B, 132K) | Ours (Qwen3-8B-4bit, 1K-4K) |
|----------|------------------------|-------------------------------|
| Base | 0% | 0% |
| RLM | 24% | 13.3% |
| Bridge | — | **36.7%** |

Our RLM score (13.3%) is lower than the paper's 24%, likely due to:
1. 4-bit quantization reducing model capability
2. Shorter context lengths (1K-4K vs 132K) — fewer questions to aggregate, less signal
3. Qwen3-8B struggles to use `llm_query()` for semantic classification via REPL

## Key Finding: Why Bridge Wins

The bridge approach outperforms all others by a different mechanism than RLM:

**RLM** gives the model tools (REPL, code execution, `llm_query()`) to process long context iteratively. But Qwen3-8B-4bit can't effectively use these tools — it tries text parsing instead of semantic classification, gets empty results, and loops.

**Bridge** condenses context to ~6% by extracting the task structure (generative ground):
- "14 questions, 6 categories, count frequencies"
- Removes the actual question text (noise for aggregation)
- The model then answers from parametric knowledge about what categories questions fall into

This works because OOLONG's task is fundamentally about **semantic classification** — something the model already knows from training. The full context is actually a distraction: the model tries to reason about each question individually instead of giving a direct aggregate answer.

### The /no_think Factor

The single biggest improvement came from disabling Qwen3's `<think>` mode:
- With `<think>`: 0% (model generates long reasoning chains, never gives clean answers)
- With `/no_think`: 30% (same model, same context, direct answers)

This is NOT a bridge-specific finding — it applies to all Qwen3 usage on structured tasks.

## Technical Notes

- **Scoring bug**: Original scoring iterated over characters of `"['abbreviation']"` (string repr) instead of list items — every answer scored as correct. Fixed with `ast.literal_eval()`.
- **`/no_think` directive**: Prepend `/no_think\n` to user messages for Qwen3 models. The model outputs empty `<think></think>` and responds directly.
- **Bridge size**: 139-375 chars (3-9% of context). Captures task schema, not data.
- **Timing**: Bridge builds in 1-12s (scales with context length). Total benchmark time: 77s vs 3409s for RLM (44x faster).

## Limitations

1. **Small scale**: 30 tasks, max 4096 tokens. Paper uses 132K tokens where RLM's advantage should be larger.
2. **Bridge content**: Current bridge captures task structure but not label distributions. A statistics-aware bridge prompt might score higher.
3. **Quantization**: 4-bit model limits all approaches. Results may differ with full-precision Qwen3-8B.
4. **Task specificity**: OOLONG's semantic aggregation particularly suits bridge (schema > data). Other benchmarks may not.

## Result Files

- `results/rlm_oolong_20260212_090513.json` — RLM 30-task run
- `results/rlm_oolong_20260212_080817.json` — Base model 30-task run
- `results/bridge_oolong_20260212_092156.json` — Bridge 30-task run (includes full-context control)
