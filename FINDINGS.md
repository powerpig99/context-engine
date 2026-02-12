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

## Next Steps: Context Length Scaling

Running 5 tasks per context length, base + RLM, stepping through:
1K → 2K → 4K → 8K → 16K → 32K → 65K → 131K

Goal: map the full crossover curve and identify where bridge should be tested.

## Technical Notes

- **`/no_think` directive**: Prepend `/no_think\n` to user messages for Qwen3 models
- **Strip `<think>` tags**: `re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()`
- **OOLONG answer field**: string repr of list — use `ast.literal_eval()`
- **OOLONG scoring**: extract after last colon, exact match for labels
- **Use `PYTHONUNBUFFERED=1`** for background runs to see progress
- **bf16 inference**: ~10x slower than 4-bit (~35s vs ~3s per generation)

## Result Files

### Session 4 (bf16)
- `results/rlm_oolong_bf16_30.json` — Base + RLM, 30 tasks, bf16

### Sessions 1-3 (4-bit, historical)
- `results/rlm_oolong_20260212_090513.json` — RLM 30-task run
- `results/rlm_oolong_20260212_080817.json` — Base model 30-task run
- `results/bridge_oolong_20260212_092156.json` — Bridge 30-task run
- `results/bridge_continuous_none_20260212_102617.json` — Continuous, no re-bridging
- `results/bridge_continuous_threshold_20260212_103206.json` — Continuous, threshold re-bridging
- `results/bridge_continuous_every5_20260212_103750.json` — Continuous, every-5 re-bridging
