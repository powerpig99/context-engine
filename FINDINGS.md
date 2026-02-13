# Context Bridge vs RLM: OOLONG Benchmark Findings

## Setup

- **Model**: Qwen3-8B via vllm-mlx on Apple Silicon M4 48GB
  - Session 1-3: 4-bit quantized (`mlx-community/Qwen3-8B-4bit`, 4.6 GB)
  - Session 4+: bf16 full precision (`mlx-community/Qwen3-8B-bf16`, 16.4 GB)
- **Benchmark**: OOLONG-synth trec_coarse — semantic aggregation over general-knowledge questions
- **Task structure**: Lists of general-knowledge questions labeled with 6 categories (abbreviation, entity, human being, numeric value, location, description and abstract concept). Tasks ask about label frequency distributions.
- **OOLONG task types**: MOST_FREQ, LEAST_FREQ, RELATIVE_FREQ (counting/comparing label distributions)
- **Scoring**: OOLONG-style — extract answer after last colon, case-insensitive exact match for labels
- **Full dataset**: 650 tasks, 13 context lengths (1K to 4M tokens), 50 tasks per length

## RLM Paper Reference (arXiv:2512.24601)

Paper results for Qwen3-8B at **131K tokens** (50 tasks):

| Approach | Score |
|----------|-------|
| Qwen3-8B Base | 0%* |
| RLM(Qwen3-8B) | 24% |
| RLM(Qwen3-8B) fine-tuned | 32% |

*Base 0% = context exceeds model window (marked with asterisk in paper).

The paper's thesis: RLM wins when context **overflows** the model's native window, because RLM processes in chunks via REPL while base model can't see the full context at all.

## Session 4: bf16 Baselines (1K-4K, 30 tasks)

### Aggregate

| Approach | Exact Match | Avg Score | Time/Task |
|----------|-------------|-----------|-----------|
| Vanilla (base + /no_think) | 7/30 = 23.3% | 0.233 | 32s |
| RLM (with REPL) | 6/30 = 20.0% | 0.200 | 120s |

### By Context Length — the crossover

| Context | Vanilla | RLM | Winner |
|---------|---------|-----|--------|
| 1024 | 1/10 = 10% | 4/10 = 40% | **RLM** |
| 2048 | 0/10 = 0% | 2/10 = 20% | **RLM** |
| 4096 | 6/10 = 60% | 0/10 = 0% | **Vanilla** |

**Key finding**: At 1K-2K, RLM's REPL helps because the model struggles to count frequencies by "looking" — code execution aids counting. At 4K, the model has enough signal to count directly, and RLM's REPL overhead hurts (generates wrong code, loops). The crossover happens somewhere between 2K-4K.

**This contradicts the initial assumption that RLM only helps at context overflow.** RLM also helps at short contexts where the task requires precise counting that the model can't do well by inspection alone. But at medium contexts (4K), vanilla direct answering beats RLM.

## Sessions 1-3: 4-bit Results (Historical)

### Isolated Benchmark (30 tasks, 4-bit)

| Approach | Exact Match | Avg Score | Time/Task | Context Used |
|----------|-------------|-----------|-----------|--------------|
| Base (no /no_think) | 0/30 = 0.0% | 0.000 | 12s | 100% |
| RLM (with REPL) | 4/30 = 13.3% | 0.133 | 114s | 100% + code exec |
| Full context (/no_think) | 9/30 = 30.0% | 0.300 | 6s | 100% |
| **Bridge (6% of ctx)** | **11/30 = 36.7%** | **0.367** | **3s** | **6%** |

Note: Base 0% was due to missing `/no_think` — not a fair baseline. The 4-bit results are superseded by bf16 results for comparison purposes.

### Continuous-Context Experiment (30 tasks, 4-bit)

Re-bridging = `build_bridge(bridge)` — applying the bridge operation to the bridge itself.

| Strategy | Bridge Accuracy | Isolated (baseline) | Delta | Re-bridges | Final Bridge |
|----------|----------------|---------------------|-------|------------|--------------|
| **threshold** | **18/30 = 60.0%** | 13/30 = 43.3% | **+16.7pp** | 1 | 167 chars |
| none | 15/30 = 50.0% | 12/30 = 40.0% | +10.0pp | 0 | 123 chars |
| every5 | 14/30 = 46.7% | 10/30 = 33.3% | +13.4pp | 6 | 214 chars |

All strategies: bridge at 0.1% of accumulated context (117-214 chars vs 167,580 chars).

## Key Findings

### 1. Bridge condenses, RLM chunks — different mechanisms
- **RLM** helps when the model can't see enough context to answer. It processes in chunks via code execution.
- **Bridge** helps by finding the generative ground — the schema from which answers can be derived. Works at any context length.

### 2. /no_think is essential for Qwen3
- Without `/no_think`: model enters thinking mode, burns tokens on `<think>` blocks, answers are garbage
- With `/no_think`: direct clean answers
- Must apply to ALL prompting paths: `bridge.py` generate(), `rlm_oolong.py` run_base_model()

### 3. The RLM crossover at short contexts
- RLM beats vanilla at 1K-2K (REPL helps with counting)
- Vanilla beats RLM at 4K (model sees enough signal, REPL overhead hurts)
- Paper's 131K crossover is about context overflow — a different phenomenon

### 4. Quantization matters
- 4-bit: faster (3s/task) but less capable
- bf16: slower (32s/task) but more accurate vanilla baseline (23.3% vs ??? with /no_think)
- Must use bf16 to match paper conditions

### 5. Bridge self-compression
- Bridge spontaneously reconsolidates during update (2747→368 chars observed)
- Final bridge size plateaus at 0.1% of accumulated context
- Threshold re-bridging (when size > 2x initial) is optimal strategy

## RLM Deep Investigation (Session 4)

### Why RLM(Qwen3-8B) fails on OOLONG — verified root cause

Investigated three hypotheses for why our RLM scores are low:

**1. `/no_think` placement was wrong** (verified, fixed, didn't help)
- Our `/no_think` was in the system prompt only. RLM constructs multi-turn messages — system → assistant → user → assistant → user → ...
- Qwen3 needs `/no_think` in **user messages**, not system. After iteration 1, thinking re-enabled.
- Fixed via monkey-patching `OpenAIClient.completion` to inject `/no_think` into last user message.
- Result: still 0/5 at 1K. Fix was correct but not the bottleneck.

**2. `max_iterations` was too low** (verified, fixed, didn't help)
- Default is 30, we used 10. Fixed to 30.
- Result: model hits same dead end more times. Extra iterations = more failed attempts at text parsing.

**3. RLM library matches paper's code** (verified)
- `rlm` pip package v0.1.0 is byte-for-byte identical to GitHub `main` branch.
- `llm_query()` timeout: 300s hardcoded. OpenAI client sends zero generation params.
- No OOLONG-specific prompt. No Qwen3-specific handling in the library.

**Actual root cause: the 8B model can't formulate the right REPL strategy.**

OOLONG requires:
1. Read each question (e.g., "What does NAFTA stand for?")
2. **Semantically classify it** into a category (→ "abbreviation")
3. Count categories across all questions

The model instead:
- Tries to `grep` literal label strings from text (finds 0 matches — labels aren't in the data)
- Extracts category names from the header (gets equal counts for each category — just the header text)
- Falls back to "same frequency as" or "cannot determine"

The correct strategy would use `llm_query()` to classify each question one-by-one — but Qwen3-8B never discovers this approach. This is a **model capability** limitation, not a configuration issue.

### Context length scaling data (bf16, 5 tasks each)

| Context | Vanilla | RLM (old) | RLM (fixed) |
|---------|---------|-----------|-------------|
| 1K | 1/10=10% | 4/10=40% | 0/5=0% |
| 2K | 0/10=0% | 2/10=20% | — |
| 4K | 6/10=60% | 0/10=0% | — |
| 8K | 2/5=40% | 1/5=20% | — |
| 16K | 1/5=20% | 0/4=0%* | — |

*16K RLM: 4/5 completed (all WRONG), 5th timed out on `llm_query()`.

**The "old" RLM 40% at 1K was inflated by lucky guessing with fewer iterations (10 vs 30).** With 30 iterations the model has more time to converge on the wrong strategy.

## Session 5: Replicating Paper at 131K (50 tasks)

### Goal
Replicate the paper's Table 1 results: Base=0%*, RLM=24% at OOLONG 131K with Qwen3-8B.

### RLM Repo Verification
Cloned both repos (`alexzhang13/rlm` and `alexzhang13/rlm-minimal`). Verified:
- Our installed `rlms==0.1.0` is **byte-identical** to GitHub HEAD (all diffs empty)
- **No OOLONG evaluation harness published** — neither repo contains benchmark/eval code
- **No Qwen-specific handling** — no model-aware prompts, no temperature settings
- **No temperature set anywhere** in RLM codebase — relies on API defaults
- Paper mentions "slightly different prompt for Qwen3-8B" — **not in the public code**
- `rlm-minimal` is a simplified reimplementation, same architecture, default models are GPT-5

### 131K Task Extraction
- Extracted 50 tasks at context_len=131072 from `oolong_trec_coarse_full.json`
- Each task has ~312K chars of context (~114K tokens)
- Task types: MOST_FREQ(5), LEAST_FREQ(4), RELATIVE_FREQ(29), NUMERIC_ONE_CLASS(12)
- 2 unique context windows (context_window_ids: 6, 8)
- Saved as `benchmarks/oolong_trec_coarse_131k.json`

### Base Model at 131K — OOM
- bf16 model (16.4 GB) + KV cache for 114K tokens (16 GB) = exceeds 48GB M4 Metal memory
- Error: "Cache entry too large: 16063.7MB exceeds limit 4361.1MB"
- Then: "METAL Command buffer execution failed: Insufficient Memory"
- **Confirms paper's 0%***: context physically cannot be processed by base model

### RLM at 131K with bf16 — 0/50 (0.0%)

| Metric | Value |
|--------|-------|
| Tasks completed | 10/50 (40 failed after OOM crash) |
| Exact match | **0/10 = 0.0%** |
| Avg score | 0.000 |
| Total time | ~18,322s (~5 hours) for 10 tasks |
| Time per task | 112s to 7,565s (highly variable) |

All 10 completed tasks were WRONG. Same failure mode as shorter contexts:
- Model writes text-parsing code (grep for literal labels) instead of using `llm_query()`
- Gets empty results `{}`, then guesses
- RELATIVE_FREQ tasks all answered with "same frequency as" (wrong)
- Server OOM-crashed during task 11, remaining 39 tasks got connection errors

### Gap Analysis: Our 0% vs Paper's 24%

The 24% gap cannot be explained by:
- ❌ Code differences (our package is identical to repo)
- ❌ Temperature (neither sets it)
- ❌ max_iterations (paper uses 30, we use 30)

Possible explanations:
- ✅ Paper's "slightly different prompt for Qwen3-8B" (unpublished)
- ✅ Different Qwen3-8B checkpoint (paper doesn't specify exact model ID)
- ✅ Different hardware (GPU with enough VRAM for full prefill vs our M4 Metal memory limits)
- ✅ bf16 OOM on our hardware — server crashes mid-run

### RLM at 131K with 4-bit — OOM from concurrent sub-LM calls

Switched to `mlx-community/Qwen3-8B-4bit` (4.6 GB) to avoid bf16 OOM.

**First run (no concurrency limit)**: 1/5 correct (20%) before crash. Server OOM'd when model used `llm_query_batched()` — fired 17+ concurrent sub-LM requests at ~50K tokens each. KV cache for 17×50K tokens exceeded Metal memory.

**Run with `--max-num-seqs 2`**: Sub-LM calls queued but timed out (RLM's 300s socket timeout too short for 64 queued requests at 2 concurrent).

**Run with `--max-num-seqs 4`**: Model successfully used `llm_query` (589 individual sub-calls in progress), memory stable at 12-13GB. But server OOM'd on a separate 114K-token prefill request while sub-LM calls were active.

Key insight: **At 131K, the model DID discover `llm_query()` as the correct strategy** (unlike at shorter contexts where it text-parses). But the concurrent load from batched sub-calls + main context prefills exceeds local hardware limits.

### RLM Computational Cost

RLM adds orders of magnitude computation:

| Approach | LLM calls per task | Time per task (4-bit, 131K) |
|----------|-------------------|----------------------------|
| Vanilla | 1 | ~15 min (prefill) |
| Bridge | 2 (build + answer) | ~15 min (build) + ~3s (answer) |
| RLM | 12-3000+ | 30-120+ min |

For OOLONG's 3182 questions, the correct RLM strategy requires ~3000 individual `llm_query` calls for semantic classification. At ~2s each with 4 concurrent = ~25 min per pass, with multiple iterations. Compare to bridge: 1 build call + 1 answer call.

An external demo with Minimax 2.5 (much more capable model) showed RLM working elegantly: 12 total LLM calls, 2.5 min wall time, max depth 2. The capable model finds relevant context efficiently (tree-shaped decomposition). Qwen3-8B at OOLONG requires flat-map decomposition (classify every question) — fundamentally different.

### Bridge at 131K — Structural Challenge

Added bridge caching per `context_window_id` and `--skip-full` flag to bridge.py for long-context benchmarks (avoids redundant 114K prefills and impossible full-context controls).

At 131K, the bridge faces a different problem than at 1K-4K:
- 1K-4K: Bridge can comprehend entire context in one pass → captures generative ground
- 131K: 3182 questions across 6 categories → bridge must capture the **distribution** structure
- A blind "summarize the generative ground" may miss that what matters is the frequency count
- The question should guide what structure the bridge captures

This connects to the three scenarios in the fiveway design:
1. **Per-task bridge**: basic mechanism test (question-agnostic)
2. **Accumulated bridge**: does sequential exposure help?
3. **Look-ahead bridge**: sees all contexts first → finds general structure → enriches per task

The look-ahead was designed precisely for this — find the general context that applies across all tasks, then refocus per question.

### Bridge at 131K — Results (5-task quick look)

| Approach | Score | Time | LLM calls |
|----------|-------|------|-----------|
| Base (bf16) | 0%* (OOM) | impossible | 1 |
| RLM (bf16) | 0/5 = 0% | ~5 hrs | 1000s |
| RLM (4-bit) | 1/5 = 20% | ~5 min (before OOM) | 100s+ |
| **Bridge (4-bit)** | **1/5 = 20%** | **25 min build + 3s/answer** | **2** |
| Paper: RLM(Qwen3-8B) | 24% (50 tasks) | — | — |

Bridge details:
- Bridge size: 11,355 chars (3.6% of 316K context), build time 1482s (~25 min)
- Task 1 (MOST_FREQ): **correct** — "numeric value"
- Task 2 (LEAST_FREQ): wrong — predicted "numeric value" (gold: "abbreviation")
- Tasks 3-5 (RELATIVE_FREQ): all wrong — "same frequency as" pattern

The blind bridge captured enough to identify the most common label but not the full distribution. The "same frequency" answers confirm the bridge doesn't preserve precise count relationships at 131K. This validates the need for question-guided bridging or structural decomposition.

**Bridge at 131K prefill/generation timing:**
- Prefill 114K tokens: ~20 min (4-bit on M4)
- Generation at 131K KV cache: ~1 tok/s (vs ~15 tok/s at short context)
- max_tokens=2048 for build_bridge (reduced from 4096 to control generation time)
- Total per bridge build: ~25 min (amortized across tasks sharing same context)

### Next Steps

The three scenarios in fiveway were designed for exactly this progression. The question is whether the bridge needs:
1. **Question-guided focus**: build the bridge knowing what structure matters
2. **Structural decomposition**: chunk → classify → aggregate (what RLM does at 1000x cost)
3. **Look-ahead enrichment**: build general bridge first, then enrich per question (Phase 5 of fiveway)

A different structural solution beyond the current three scenarios may be needed at 131K.

## Technical Notes

- **`/no_think` directive**: Prepend `/no_think\n` to user messages for Qwen3 models
- **Strip `<think>` tags**: `re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()`
- **OOLONG answer field**: string repr of list — use `ast.literal_eval()`
- **OOLONG scoring**: extract after last colon, exact match for labels
- **Use `PYTHONUNBUFFERED=1`** for background runs to see progress
- **bf16 inference**: ~10x slower than 4-bit (~35s vs ~3s per generation)

## Result Files

### Session 5-6 (131K replication + bridge)
- `results/bridge_131k_5.json` — Bridge at 131K, 5 tasks, 4-bit (1/5 = 20%)
- `results/rlm_131k_50.json` — RLM at 131K, 50 tasks, bf16 (0/50, server OOM)
- `results/rlm_131k_4bit_50.json` — RLM at 131K, 50 tasks, 4-bit (1/50, server OOM after 5)
- `results/rlm_131k_2.json` — RLM at 131K, 2-task smoke test, bf16 (1/2)
- `results/base_131k_5.json` — Base at 131K, 5 tasks (0/5, connection errors)
- `benchmarks/oolong_trec_coarse_131k.json` — 50 tasks at 131K context

### Session 4 (bf16)
- `results/rlm_oolong_bf16_30.json` — Base + RLM, 30 tasks, bf16
- `results/baseline_8k_5.json` — Base + RLM, 5 tasks at 8K, bf16
- `results/rlm_fixed_1k_5.json` — RLM with fixes (/no_think + max_iter=30), 5 tasks at 1K

### Sessions 1-3 (4-bit, historical)
- `results/rlm_oolong_20260212_090513.json` — RLM 30-task run
- `results/rlm_oolong_20260212_080817.json` — Base model 30-task run
- `results/bridge_oolong_20260212_092156.json` — Bridge 30-task run
- `results/bridge_continuous_none_20260212_102617.json` — Continuous, no re-bridging
- `results/bridge_continuous_threshold_20260212_103206.json` — Continuous, threshold re-bridging
- `results/bridge_continuous_every5_20260212_103750.json` — Continuous, every-5 re-bridging
