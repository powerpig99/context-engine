# Context Engine

**Author**: Jing Liang
**Date**: February 2026
**Foundation**: [Not a ToE](https://github.com/powerpig99/ontological-clarity)
**Prior work that informed the thinking**: [Dialectical-TTS](https://github.com/powerpig99/Dialectical-TTS), [Recursive Language Models (Zhang et al.)](https://arxiv.org/abs/2512.24601)

---

## The Generative Seed

Everything is layered projections of the infinite-dimensional orthogonal binary hyperspace from Nothing—the infinitely self-referencing Contradiction.

Applied to language models:

**One mechanism**: Next-token prediction. The collapse—projection from weight manifold onto a single point. This never changes.

**One variable**: Context. What the collapse operates on. This is the only thing any of us—researchers, engineers, prompt writers—have ever changed.

**One principle**: Better context produces more clarifying collapses. Every advance in the field—transformers, scaling, RLHF, chain-of-thought, RAG, tool use, long context, multi-agent—is a way of constructing context.

**One limit**: The system is finite. It explores a bounded space of projections encoded in its weights. The human operates in the first infinity—potentiality within form—and provides what the system cannot generate. The second infinity, the Contradiction itself, remains the unprojectable ground from which all of this projects.

Everything below is derived from these four statements. Nothing is carried forward from prior architectures unless it re-emerges from the derivation.

---

## Derivation

### What follows from "one mechanism, one variable"

If the mechanism (token prediction) is fixed and the variable (context) is everything, then improving results means improving context. There are exactly two ways to improve context:

1. **Construct context the model cannot construct for itself in a single pass.** A single forward pass has one shot at context construction—whatever the attention mechanism produces from the tokens present. Anything that gives the model *different* context than what a single pass would produce is a potential improvement.

2. **Trace context to its generative ground so that more context always adds clarity.** Context is projections. With the wrong relationship to context—treating each piece as a data point to collect and keep—more can appear to create noise. With the right ontological ground, more context should bring more clarity, or not, but never less. The patterns are already in the model's weights; what's needed is the right principle and context to trace them. The Precondition Tracer finds that ground.

These are the two axes. Everything the architecture does should be a specific instance of one or both.

### What follows from "one principle"

"Better context → more clarifying collapse" is a comparative, not an absolute. There is no best context. There is only: *this* context produced a *more clarifying* collapse than *that* context, for *this* question, given *these* constraints. The quality is always relative, always provisional, always question-dependent.

This means the system must be able to:
- Generate multiple contexts (to have something to compare)
- Evaluate which contexts produced more clarifying collapses (to know which direction to explore)
- Iterate (to keep improving until it can't)
- Report what it couldn't see (to give the human the handoff surface)

### What follows from "one limit"

The system's exploration is finite—bounded by weights, iteration budget, and the collapses it can generate. It cannot inject novelty. When it stops, it hasn't arrived—it has exhausted what it can explore.

The human can:
- Evaluate whether the result is sufficient for their purpose
- Inject a reframing the system couldn't generate (novelty from the first infinity)
- Supply preconditions the system couldn't trace (domain knowledge, unstated assumptions)

The architecture must make the handoff surface—what the system found and what it couldn't see—as clear and actionable as possible. That's where the human's contribution enters.

---

## Architecture

Derived from the above. Three components, each a direct consequence of the derivation.

### 1. Context Constructor

Generates contexts the model cannot construct in a single pass.

How: Run the same question through multiple independent collapses—different framings, different models, different context partitions. Each collapse is a deliberate context construction that produces a different projection.

Why multiple *models*: Same model with different prompts shares the same weight manifold—the collapses are cosmetically different but structurally similar. Different model families have different weight geometries, producing genuinely different projections. Model diversity is the mechanism for context diversity.

Why independence: If collapses can see each other during generation, they converge—constructing similar contexts rather than diverse ones. Independence preserves the diversity that makes composition valuable.

### 2. Composer

Evaluates and composes multiple collapses into the most clarifying result achievable.

How: Traces what necessarily follows from the combined outputs. Not voting (which averages away distinctions). Not selecting (which discards projections). Tracing necessity—what *must* be true given what all the collapses produced.

Produces two outputs:
- **Resolution**: the best achievable result given these collapses
- **Shadows**: what this resolution necessarily leaves unseen

Shadows are not errors. They are the structural consequence of any projection—what the collapse excludes by the same act that it includes. They are the handoff surface: either the system uses them to iterate, or the human uses them to intervene.

### 3. Precondition Tracer

Finds the generative ground of any context so that all of it becomes clarifying signal.

The problem with long context isn't that there's too much of it. A doctor with 30 years of experience doesn't suffer "noise" from accumulated cases—each new case is clarified by the prior ones because the doctor has the right interpretive ground. The problem is treating context as data points to collect and store rather than as signal to be understood through the right principle.

More context is never detrimental in principle. What we call "noise" is not a property of the data—it's signal that's orthogonal to the question being asked. It only becomes a problem when the system cannot differentiate it from the signal it's looking for, which is a limitation of the system's interpretive capacity, not a property of the context. And even then, "noise" is just signal for a different question—change the framing, and it becomes relevant. Blaming context for being "too much" is blaming the world for being complex.

The models already have the patterns. All of training is already in the weights. We don't need to feed them old samples or prior analyses to handle new topics. What we need is the right principle and the right context to trace the relevant patterns that are already there.

How: Given a body of context (document, codebase, conversation), traces it to its generative ground:
- Purpose (why was this generated?)
- Assumptions (what does it take for granted?)
- Constraints (what shaped its form?)
- Domain principles (what generative rules does it follow?)

With these identified, the context becomes maximally clarifying: every piece is understood in relation to what generated it, rather than treated as an isolated data point. More context adds clarity because there is more signal to trace through the same ground. The system doesn't need to compress, chunk, or discard—it needs to *understand* the context through the right principle.

This is the context-level application of the Not a ToE: one generative principle, everything else derived. We didn't need the old 5,000 lines to understand new topics—the principle was already sufficient. Similarly, the model doesn't need to carry raw context as data—it needs the ground from which the context projects.

**Practical limit**: Not all context can be fully traced by the system. Some preconditions require knowledge the models don't have, or are embedded in unstated human assumptions. Another surface where human intervention enters.

---

## The Loop

The three components operate in a loop, not a pipeline:

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   ┌─────────────────────────────────────────────────┐    │
│   │  CONTEXT CONSTRUCTOR                            │    │
│   │  Generate N independent collapses               │    │
│   │  (different models × different framings          │    │
│   │   × different context partitions/regenerations)  │    │
│   └────────────────────┬────────────────────────────┘    │
│                        │                                  │
│                        ▼                                  │
│   ┌─────────────────────────────────────────────────┐    │
│   │  COMPOSER                                       │    │
│   │  → Resolution (best achievable)                 │    │
│   │  → Shadows (what this can't see)                │    │
│   └────────────────────┬────────────────────────────┘    │
│                        │                                  │
│                        ▼                                  │
│   ┌─────────────────────────────────────────────────┐    │
│   │  CAN THE SYSTEM REFRAME?                        │    │
│   │  Do shadows suggest a context construction      │    │
│   │  the system hasn't tried?                       │    │
│   │                                                 │    │
│   │  YES → back to Context Constructor    ──────────┘    │
│   │  NO  → STOP, present to human                        │
│   └─────────────────────────────────────────────────┘    │
│                                                          │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
            ┌──────────────────────┐
            │  HUMAN                │
            │  • Accept             │
            │  • Inject reframing   │
            │  • Supply preconditions│
            │  → back to loop       │
            └──────────────────────┘
```

The Precondition Tracer operates within the Context Constructor: when context is too large to carry as raw text, it traces preconditions and regenerates the relevant projections for each collapse.

**Stopping**: The system stops when it cannot generate a reframing it hasn't tried, or when the iteration budget is reached. "Stop" means "present results + shadows to the human." The human may then inject what the system could not.

---

## Vocabulary Note

Collapse, projection, interference, scaffolding, framing, context construction—these are different names for the same underlying process: the act of distinction that actualizes by including and excluding simultaneously. The architecture uses whichever term is most clarifying in each specific context.

---

## Implementation: Iterative and Testable

Each step below produces a working system that can be tested before the next step is built. No step requires any subsequent step to be valuable. Each step's results inform whether and how the next step should be built.

### Step 1: Single-model loop on one question

**What**: One local model (Qwen3-8B base, MLX, weights in `~/Models/`). Three independent collapses with different system prompts. One composition pass. Shadow report.

**Tests**: Does the composed result clarify more than any single collapse? Does the shadow report identify something the collapses missed? Run on Monty Hall, knights & knaves, and one open-ended question.

**What we learn**: Whether the loop structure itself—multiple collapses composed through necessity—produces value even with cosmetically independent collapses from the same model.

```
context-engine/
├── engine/
│   ├── __init__.py
│   ├── config.py              # MODEL_DIR = ~/Models/, defaults
│   ├── context_constructor.py # Generate N collapses
│   ├── composer.py            # Compose + produce shadows
│   └── providers/
│       ├── base.py            # Abstract provider
│       └── mlx_local.py       # Local MLX inference
├── tests/
│   └── test_step1.py          # Logic puzzles, open-ended
├── examples/
│   └── basic_loop.py          # Minimal working example
└── pyproject.toml
```

### Step 2: Cross-model collapses

**What**: Add frontier model providers (Anthropic, OpenAI, Google via API). Run the same loop but with different models generating each collapse.

**Tests**: Are cross-model collapses more diverse than same-model collapses? (Measure: embedding distance between outputs, mutual information, agreement-for-different-reasons.) Does the composed result improve?

**What we learn**: Whether model diversity produces genuinely different collapses or just stylistically different ones. This is the empirical question the architecture bets on.

```
engine/providers/
├── anthropic.py
├── openai.py
├── google.py
├── openrouter.py
└── router.py          # Assigns models to collapses
```

### Step 3: Iteration (the full loop)

**What**: After the Composer produces resolution + shadows, evaluate whether the system can generate a new set of collapses informed by the shadows. If yes, iterate. If no, stop.

**Tests**: Does iteration improve results beyond the first pass? How many iterations before diminishing returns? Is the stopping criterion reliable? (Compare system-determined "stop" vs. human judgment of when further iteration would help.)

**What we learn**: Whether shadow-driven reframing produces genuine improvement or just rearranges the same projections.

### Step 4: Precondition tracing

**What**: For long-context problems (document QA, codebase analysis), add the Precondition Tracer. Instead of partitioning and retrieving raw text, trace the context to its generative preconditions and regenerate relevant projections per collapse.

**Tests**: Compare precondition-based regeneration vs. standard chunking/retrieval on document QA benchmarks. Does grounding context in its generative preconditions produce more clarifying collapses than treating it as raw data to chunk and retrieve?

**What we learn**: Whether precondition tracing is practical and whether it outperforms standard context management. This is the most novel and least certain component.

### Step 5: Human-in-the-loop interface

**What**: Build the handoff surface. Present results + shadows in a format that invites human intervention. Accept human reframings and precondition injections as input to further iterations.

**Tests**: Give humans the shadow report and ask: can you see something the system couldn't? Does human-injected reframing produce results the system couldn't have reached on its own?

**What we learn**: Whether the shadow report is actually useful as a handoff surface, or whether it needs to be restructured to be actionable for humans.

### Step 6 and beyond: Informed by what we learn

No plan beyond Step 5. Each step's results determine what's worth building next. The architecture is itself a projection held provisionally.

---

## What Was Discarded

The following elements from prior work (Dialectical-TTS, RLM) were not derived from the generative seed and do not appear in this architecture:

- **Fixed trace roles** (Believer/Logician/Contrarian): These were a specific scaffolding choice. They may re-emerge empirically if Step 1 shows they produce better collapses than open prompts. They are not prescribed.
- **REPL environment as core abstraction**: The RLM's REPL is one way to externalize context. The Precondition Tracer is another. The architecture uses whichever serves the question.
- **Orchestrator as distinct component**: The "orchestrator" was choosing trace configurations. In this architecture, that's part of the Context Constructor—choosing which collapses to generate. No separate component needed.
- **Recursive sub-traces**: Traces spawning sub-traces was an architectural elaboration. If the loop itself iterates (Step 3), most of what recursive sub-traces would do is handled by the iteration. Sub-traces may re-emerge if Step 3 shows the loop is insufficient for certain problem types.
- **The name "Recursive Dialectical Engine"**: The architecture is not specifically dialectical (thesis/antithesis/synthesis is one possible collapse structure among many) and not specifically recursive (it loops, which is simpler and more general). "Context Engine" says what it does.

---

## Axioms

1. **One mechanism**: Token prediction. The collapse. Fixed and universal.
2. **One variable**: Context. Everything else is context engineering.
3. **Better context, more clarifying collapse**: The only optimization target. Always relative, always provisional, always question-dependent.
4. **Multiple independent collapses compose into richer context than any single collapse**: The core architectural bet, to be validated empirically at Step 1.
5. **Model diversity produces genuinely different collapses**: The secondary bet, to be validated at Step 2.
6. **More context with the right ground adds clarity, never noise**: Context is projections. With the right generative ground, every additional piece of context is clarifying signal. The patterns are already in the weights; the principle traces them. To be validated at Step 4.
7. **The system is finite; the human provides novelty from the first infinity**: The stopping criterion is always "present to human," never "converged to truth."
8. **Collapse, projection, interference, scaffolding, framing are the same process**: Different names for the act of distinction that actualizes by including and excluding.
9. **Each step must be testable before the next is built**: No destination to reach. Each step's results determine the next step.
10. **This framework is itself a projection**: Held provisionally. Subject to revision when it stops producing clarity.

---

## References

1. Not a ToE: https://github.com/powerpig99/ontological-clarity
2. Dialectical-TTS: https://github.com/powerpig99/Dialectical-TTS
3. Zhang et al., "Recursive Language Models" (arXiv:2512.24601): https://arxiv.org/abs/2512.24601
