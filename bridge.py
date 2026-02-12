"""
Context Bridge — Step 1 experiment.

Build a bridge from growing context. Test if answering from the bridge
produces clarity comparable to answering from the full text.

Requires vllm-mlx running:
    vllm-mlx serve mlx-community/Qwen3-8B-4bit --port 8000 --continuous-batching

Usage:
    python bridge.py build documents/text.md
    python bridge.py update bridge.md new_content.md
    python bridge.py compare bridge.md documents/full.md "question"
    python bridge.py run documents/text.md "question1" "question2" ...
    python bridge.py benchmark benchmarks/oolong_trec_coarse.json --limit 10
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# --- Provider (minimal, no abstraction needed for one experiment) ---

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
MODEL = "default"


def generate(system_prompt: str, user_prompt: str, temperature: float = 0.5, max_tokens: int = 4096) -> str:
    # Prepend /no_think to disable Qwen3's <think> mode for faster, direct responses
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "/no_think\n" + user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content
    # Strip any residual <think>...</think> blocks from Qwen3
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text


# --- Core operations ---

def build_bridge(text: str, temperature: float = 0.5) -> str:
    """Build a bridge from text. Not a summary — the generative ground."""
    return generate(
        system_prompt=(
            "You are building a bridge — a condensed generative ground from which "
            "the key content of this text can be regenerated on demand.\n\n"
            "This is NOT a summary. A summary compresses by discarding. "
            "A bridge condenses by finding the generative seed — the core principles, "
            "purpose, and structure from which the text's content derives.\n\n"
            "The bridge should be:\n"
            "- The generative ground (what generated this text?)\n"
            "- The minimal derivation skeleton (how do the parts connect?)\n"
            "- Short enough to hold in working memory\n"
            "- Rich enough to derive answers to questions about the text\n\n"
            "Produce only the bridge. No commentary."
        ),
        user_prompt=text,
        temperature=temperature,
    )


def update_bridge(bridge: str, new_text: str, temperature: float = 0.5) -> str:
    """Update an existing bridge with new content. Grow only where genuinely new."""
    return generate(
        system_prompt=(
            "You have an existing bridge — a condensed generative ground for a body of context. "
            "New content has arrived.\n\n"
            "Update the bridge. Rules:\n"
            "- Grow only where the new content adds genuinely new understanding\n"
            "- Don't append — reconsolidate. The bridge should remain a unified ground.\n"
            "- If the new content is just more derivations from the same ground, "
            "the bridge may not need to grow at all\n"
            "- If it reveals new ground, integrate it\n\n"
            "Produce only the updated bridge. No commentary."
        ),
        user_prompt=f"CURRENT BRIDGE:\n{bridge}\n\nNEW CONTENT:\n{new_text}",
        temperature=temperature,
    )


def answer(question: str, context: str, temperature: float = 0.5) -> str:
    """Answer a question using the given context."""
    return generate(
        system_prompt=(
            "Answer the question using the provided context. "
            "Derive your answer from the context. Be direct and clear."
        ),
        user_prompt=f"CONTEXT:\n{context}\n\nQUESTION:\n{question}",
        temperature=temperature,
    )


# --- Commands ---

def cmd_build(args):
    """Build a bridge from a document."""
    text = Path(args.document).read_text()
    bridge = build_bridge(text)

    print(f"Source: {len(text)} chars")
    print(f"Bridge: {len(bridge)} chars ({len(bridge)/len(text)*100:.0f}% of source)")
    print(f"\n{'=' * 60}")
    print("BRIDGE")
    print(f"{'=' * 60}")
    print(bridge)

    if args.save:
        Path(args.save).write_text(bridge)
        print(f"\nBridge saved to {args.save}")


def cmd_update(args):
    """Update a bridge with new content."""
    bridge = Path(args.bridge).read_text()
    new_text = Path(args.new_content).read_text()

    old_len = len(bridge)
    updated = update_bridge(bridge, new_text)

    print(f"Old bridge: {old_len} chars")
    print(f"New content: {len(new_text)} chars")
    print(f"Updated bridge: {len(updated)} chars (growth: {len(updated) - old_len:+d})")
    print(f"\n{'=' * 60}")
    print("UPDATED BRIDGE")
    print(f"{'=' * 60}")
    print(updated)

    if args.save:
        Path(args.save).write_text(updated)
        print(f"\nUpdated bridge saved to {args.save}")


def cmd_compare(args):
    """Compare answers from bridge vs. full document."""
    bridge = Path(args.bridge).read_text()
    full_text = Path(args.document).read_text()
    question = args.question

    print(f"Bridge: {len(bridge)} chars | Full text: {len(full_text)} chars")
    print(f"Ratio: {len(bridge)/len(full_text)*100:.0f}%")
    print(f"\nQuestion: {question}")

    answer_bridge = answer(question, bridge)
    answer_full = answer(question, full_text)

    print(f"\n{'=' * 60}")
    print("FROM BRIDGE")
    print(f"{'=' * 60}")
    print(answer_bridge)
    print(f"\n{'=' * 60}")
    print("FROM FULL TEXT")
    print(f"{'=' * 60}")
    print(answer_full)

    if args.save:
        result = {
            "question": question,
            "bridge_size": len(bridge),
            "full_text_size": len(full_text),
            "ratio": len(bridge) / len(full_text),
            "answer_from_bridge": answer_bridge,
            "answer_from_full_text": answer_full,
            "timestamp": datetime.now().isoformat(),
        }
        Path(args.save).write_text(json.dumps(result, indent=2))
        print(f"\nComparison saved to {args.save}")


def cmd_run(args):
    """Full experiment: build bridge from growing chunks, compare at each stage."""
    text = Path(args.document).read_text()
    questions = args.questions

    # Split into roughly equal chunks
    n_chunks = args.chunks
    chunk_size = len(text) // n_chunks
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = len(text) if i == n_chunks - 1 else (i + 1) * chunk_size
        chunks.append(text[start:end])

    print(f"Document: {len(text)} chars, split into {n_chunks} chunks")
    print(f"Questions: {questions}")

    bridge = None
    accumulated = ""
    results = []

    for i, chunk in enumerate(chunks):
        accumulated += chunk
        print(f"\n{'#' * 60}")
        print(f"STAGE {i + 1}: +{len(chunk)} chars (total: {len(accumulated)} chars)")
        print(f"{'#' * 60}")

        if bridge is None:
            bridge = build_bridge(accumulated)
            print(f"Built initial bridge: {len(bridge)} chars")
        else:
            bridge = update_bridge(bridge, chunk)
            print(f"Updated bridge: {len(bridge)} chars")

        print(f"Ratio: {len(bridge)/len(accumulated)*100:.0f}%")

        for q in questions:
            print(f"\n  Q: {q}")
            ab = answer(q, bridge)
            af = answer(q, accumulated)
            print(f"  FROM BRIDGE: {ab[:200]}...")
            print(f"  FROM FULL:   {af[:200]}...")
            results.append({
                "stage": i + 1,
                "accumulated_size": len(accumulated),
                "bridge_size": len(bridge),
                "question": q,
                "answer_bridge": ab,
                "answer_full": af,
            })

    if args.save:
        Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.save}")


def cmd_benchmark(args):
    """Run bridge approach on OOLONG benchmark, compare to full-context baseline."""
    from rlm_oolong import parse_answer_field, extract_answer, score_answer

    tasks = json.loads(Path(args.data).read_text())
    for task in tasks:
        task["answer"] = parse_answer_field(task["answer"])
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"Loaded {len(tasks)} tasks from {args.data}")

    results = []
    for i, task in enumerate(tasks):
        ctx = task["context_window_text"]
        question = task["question"]
        print(f"\n  [{i+1}/{len(tasks)}] ctx={task['context_len']} task={task['task']}")

        # 1. Build bridge from context
        t0 = time.time()
        bridge = build_bridge(ctx)
        bridge_time = time.time() - t0
        ratio = len(bridge) / len(ctx)
        print(f"    Bridge: {len(bridge)} chars ({ratio*100:.0f}% of {len(ctx)}) in {bridge_time:.1f}s")

        # 2. Answer from bridge
        t0 = time.time()
        answer_from_bridge = answer(question, bridge)
        bridge_answer_time = time.time() - t0
        bridge_extracted = extract_answer(answer_from_bridge)
        bridge_score = score_answer(answer_from_bridge, task["answer"], task["answer_type"])

        # 3. Answer from full context (control)
        t0 = time.time()
        answer_from_full = answer(question, ctx)
        full_answer_time = time.time() - t0
        full_extracted = extract_answer(answer_from_full)
        full_score = score_answer(answer_from_full, task["answer"], task["answer_type"])

        b_label = f"OK ({bridge_score:.2f})" if bridge_score > 0 else "WRONG"
        f_label = f"OK ({full_score:.2f})" if full_score > 0 else "WRONG"
        print(f"    Bridge answer: {b_label} [{bridge_extracted[:60]}]")
        print(f"    Full answer:   {f_label} [{full_extracted[:60]}]")

        results.append({
            "id": task["id"],
            "context_len": task["context_len"],
            "task": task["task"],
            "task_group": task["task_group"],
            "question": question,
            "gold": task["answer"],
            "bridge_size": len(bridge),
            "bridge_ratio": ratio,
            "bridge_answer": answer_from_bridge,
            "bridge_extracted": bridge_extracted,
            "bridge_score": bridge_score,
            "bridge_time_s": bridge_time + bridge_answer_time,
            "full_answer": answer_from_full,
            "full_extracted": full_extracted,
            "full_score": full_score,
            "full_time_s": full_answer_time,
        })

    # Summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    bridge_exact = sum(1 for r in results if r["bridge_score"] == 1.0)
    full_exact = sum(1 for r in results if r["full_score"] == 1.0)
    bridge_avg = sum(r["bridge_score"] for r in results) / len(results)
    full_avg = sum(r["full_score"] for r in results) / len(results)
    avg_ratio = sum(r["bridge_ratio"] for r in results) / len(results)
    print(f"  Bridge: {bridge_exact}/{len(results)} exact = {bridge_exact/len(results)*100:.1f}%, avg_score={bridge_avg:.3f}")
    print(f"  Full:   {full_exact}/{len(results)} exact = {full_exact/len(results)*100:.1f}%, avg_score={full_avg:.3f}")
    print(f"  Avg bridge ratio: {avg_ratio*100:.0f}%")

    # Save
    save_path = args.save or f"results/bridge_oolong_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "data_file": args.data,
        "num_tasks": len(tasks),
        "results": results,
    }
    Path(save_path).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {save_path}")


def cmd_continuous(args):
    """Continuous-context variant: accumulate context, build evolving bridge."""
    from rlm_oolong import parse_answer_field, extract_answer, score_answer

    tasks = json.loads(Path(args.data).read_text())
    for task in tasks:
        task["answer"] = parse_answer_field(task["answer"])
    # Sort by context_len ascending to simulate growing context
    tasks.sort(key=lambda t: (t["context_len"], t["id"]))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"Loaded {len(tasks)} tasks from {args.data} (sorted by context_len)")
    print(f"Re-bridge strategy: {args.rebridge}")

    accumulated = ""
    bridge = None
    initial_bridge_size = None
    rebridge_count = 0
    results = []

    for i, task in enumerate(tasks):
        ctx = task["context_window_text"]
        question = task["question"]
        accumulated += ctx + "\n\n"

        print(f"\n  [{i+1}/{len(tasks)}] ctx={task['context_len']} task={task['task']}")
        print(f"    Accumulated: {len(accumulated)} chars")

        # Build or update bridge
        t0 = time.time()
        if bridge is None:
            bridge = build_bridge(accumulated)
            initial_bridge_size = len(bridge)
        else:
            bridge = update_bridge(bridge, ctx)

        # Re-bridge if strategy triggers it
        did_rebridge = False
        if args.rebridge == "threshold" and initial_bridge_size and len(bridge) > initial_bridge_size * 2:
            bridge = build_bridge(bridge)
            did_rebridge = True
            rebridge_count += 1
        elif args.rebridge == "every5" and (i + 1) % 5 == 0 and i > 0:
            bridge = build_bridge(bridge)
            did_rebridge = True
            rebridge_count += 1

        bridge_build_time = time.time() - t0
        ratio = len(bridge) / len(accumulated) if accumulated else 0
        rb_tag = " [RE-BRIDGED]" if did_rebridge else ""
        print(f"    Bridge: {len(bridge)} chars ({ratio*100:.1f}% of {len(accumulated)}) in {bridge_build_time:.1f}s{rb_tag}")

        # Answer from two sources: bridge vs isolated context
        # 1. Bridge (the thing we're testing — can it replace growing context?)
        t0 = time.time()
        answer_bridge = answer(question, bridge)
        bridge_time = time.time() - t0
        bridge_extracted = extract_answer(answer_bridge)
        bridge_score = score_answer(answer_bridge, task["answer"], task["answer_type"])

        # 2. Isolated (just this task's context — the gold standard)
        t0 = time.time()
        answer_iso = answer(question, ctx)
        iso_time = time.time() - t0
        iso_extracted = extract_answer(answer_iso)
        iso_score = score_answer(answer_iso, task["answer"], task["answer_type"])

        b_label = f"OK ({bridge_score:.2f})" if bridge_score > 0 else "WRONG"
        i_label = f"OK ({iso_score:.2f})" if iso_score > 0 else "WRONG"
        print(f"    Bridge:   {b_label} [{bridge_extracted[:50]}]")
        print(f"    Isolated: {i_label} [{iso_extracted[:50]}]")

        results.append({
            "step": i + 1,
            "id": task["id"],
            "context_len": task["context_len"],
            "task": task["task"],
            "task_group": task["task_group"],
            "question": question,
            "gold": task["answer"],
            "accumulated_size": len(accumulated),
            "bridge_size": len(bridge),
            "bridge_ratio": ratio,
            "did_rebridge": did_rebridge,
            "bridge_extracted": bridge_extracted,
            "bridge_score": bridge_score,
            "iso_extracted": iso_extracted,
            "iso_score": iso_score,
        })

    # Summary
    print(f"\n{'=' * 60}")
    print("CONTINUOUS-CONTEXT RESULTS")
    print(f"{'=' * 60}")
    n = len(results)
    for label, key in [("Bridge", "bridge_score"), ("Isolated", "iso_score")]:
        exact = sum(1 for r in results if r[key] == 1.0)
        avg = sum(r[key] for r in results) / n
        print(f"  {label:12s}: {exact}/{n} exact = {exact/n*100:.1f}%, avg_score={avg:.3f}")
    final_ratio = results[-1]["bridge_ratio"] if results else 0
    print(f"  Final bridge ratio: {final_ratio*100:.1f}%")
    print(f"  Re-bridge operations: {rebridge_count}")
    print(f"  Bridge size: {results[0]['bridge_size']} → {results[-1]['bridge_size']} chars")
    print(f"  Accumulated size: {results[0]['accumulated_size']} → {results[-1]['accumulated_size']} chars")

    # Save
    save_path = args.save or f"results/bridge_continuous_{args.rebridge}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "data_file": args.data,
        "num_tasks": n,
        "rebridge_strategy": args.rebridge,
        "rebridge_count": rebridge_count,
        "results": results,
    }
    Path(save_path).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {save_path}")


def cmd_fiveway(args):
    """5-way comparison experiment: vanilla, RLM, per-task bridge, accumulated bridge, look-ahead bridge."""
    from rlm_oolong import parse_answer_field, extract_answer, score_answer

    TEMP = 0.0  # Deterministic for reproducibility

    tasks = json.loads(Path(args.data).read_text())
    for task in tasks:
        task["answer"] = parse_answer_field(task["answer"])
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"Loaded {len(tasks)} tasks from {args.data}")

    n = len(tasks)
    # Pre-index unique context windows
    ctx_map = {}  # context_window_id -> context text
    for task in tasks:
        cid = task["context_window_id"]
        if cid not in ctx_map:
            ctx_map[cid] = task["context_window_text"]
    print(f"Unique context windows: {len(ctx_map)} (ids: {sorted(ctx_map.keys())})")

    # Initialize per-task result dicts
    results = []
    for task in tasks:
        results.append({
            "id": task["id"],
            "context_len": task["context_len"],
            "context_window_id": task["context_window_id"],
            "task": task["task"],
            "task_group": task["task_group"],
            "question": task["question"],
            "gold": task["answer"],
        })

    # ── Phase 1: Vanilla baseline ──
    print(f"\n{'=' * 60}")
    print("PHASE 1: Vanilla (isolated context, direct answer)")
    print(f"{'=' * 60}")
    t_start = time.time()
    for i, task in enumerate(tasks):
        ctx = task["context_window_text"]
        raw = answer(task["question"], ctx, temperature=TEMP)
        extracted = extract_answer(raw)
        sc = score_answer(raw, task["answer"], task["answer_type"])
        results[i]["vanilla_raw"] = raw
        results[i]["vanilla_extracted"] = extracted
        results[i]["vanilla_score"] = sc
        label = f"OK ({sc:.2f})" if sc > 0 else "WRONG"
        print(f"  [{i+1}/{n}] {label} [{extracted[:60]}]")
    print(f"  Phase 1 done in {time.time() - t_start:.1f}s")

    # ── Phase 2: RLM from cache ──
    print(f"\n{'=' * 60}")
    print("PHASE 2: RLM + REPL (from cache)")
    print(f"{'=' * 60}")
    if args.rlm_cache:
        rlm_data = json.loads(Path(args.rlm_cache).read_text())
        rlm_results_raw = rlm_data["results"]
        # Index by task id
        rlm_by_id = {r["id"]: r for r in rlm_results_raw}
        for i, task in enumerate(tasks):
            cached = rlm_by_id.get(task["id"])
            if cached and cached.get("predicted"):
                # Rescore with current score_answer for consistency
                sc = score_answer(cached["predicted"], task["answer"], task["answer_type"])
                extracted = extract_answer(cached["predicted"])
                results[i]["rlm_raw"] = cached["predicted"]
                results[i]["rlm_extracted"] = extracted
                results[i]["rlm_score"] = sc
            else:
                results[i]["rlm_raw"] = None
                results[i]["rlm_extracted"] = ""
                results[i]["rlm_score"] = 0.0
            label = f"OK ({results[i]['rlm_score']:.2f})" if results[i]["rlm_score"] > 0 else "WRONG"
            print(f"  [{i+1}/{n}] {label} [{results[i]['rlm_extracted'][:60]}]")
        print(f"  Loaded {sum(1 for r in results if r.get('rlm_raw'))} cached RLM results")
    else:
        print("  No --rlm-cache provided, skipping RLM. Scores set to 0.")
        for i in range(n):
            results[i]["rlm_raw"] = None
            results[i]["rlm_extracted"] = ""
            results[i]["rlm_score"] = 0.0

    # ── Phase 3: Per-task bridge ──
    print(f"\n{'=' * 60}")
    print("PHASE 3: Per-task bridge (bridge per unique context window)")
    print(f"{'=' * 60}")
    t_start = time.time()
    bridge_cache = {}  # context_window_id -> bridge text
    for cid, ctx_text in ctx_map.items():
        print(f"  Building bridge for context_window_id={cid} ({len(ctx_text)} chars)...")
        bridge_cache[cid] = build_bridge(ctx_text, temperature=TEMP)
        print(f"    Bridge: {len(bridge_cache[cid])} chars ({len(bridge_cache[cid])/len(ctx_text)*100:.0f}%)")

    for i, task in enumerate(tasks):
        bridge = bridge_cache[task["context_window_id"]]
        raw = answer(task["question"], bridge, temperature=TEMP)
        extracted = extract_answer(raw)
        sc = score_answer(raw, task["answer"], task["answer_type"])
        results[i]["pertask_raw"] = raw
        results[i]["pertask_extracted"] = extracted
        results[i]["pertask_score"] = sc
        label = f"OK ({sc:.2f})" if sc > 0 else "WRONG"
        print(f"  [{i+1}/{n}] {label} [{extracted[:60]}]")
    print(f"  Phase 3 done in {time.time() - t_start:.1f}s")

    # ── Phase 4: Accumulated bridge ──
    print(f"\n{'=' * 60}")
    print("PHASE 4: Accumulated bridge (sequential, threshold re-bridge)")
    print(f"{'=' * 60}")
    t_start = time.time()
    # Sort tasks by context_len for accumulation
    sorted_indices = sorted(range(n), key=lambda j: (tasks[j]["context_len"], tasks[j]["id"]))
    accum_bridge = None
    initial_bridge_size = None
    rebridge_count = 0

    for step, idx in enumerate(sorted_indices):
        task = tasks[idx]
        ctx = task["context_window_text"]

        if accum_bridge is None:
            accum_bridge = build_bridge(ctx, temperature=TEMP)
            initial_bridge_size = len(accum_bridge)
        else:
            accum_bridge = update_bridge(accum_bridge, ctx, temperature=TEMP)

        # Threshold re-bridge: if bridge > 2x initial size
        did_rebridge = False
        if initial_bridge_size and len(accum_bridge) > initial_bridge_size * 2:
            accum_bridge = build_bridge(accum_bridge, temperature=TEMP)
            did_rebridge = True
            rebridge_count += 1

        raw = answer(task["question"], accum_bridge, temperature=TEMP)
        extracted = extract_answer(raw)
        sc = score_answer(raw, task["answer"], task["answer_type"])
        results[idx]["accum_raw"] = raw
        results[idx]["accum_extracted"] = extracted
        results[idx]["accum_score"] = sc
        results[idx]["accum_bridge_size"] = len(accum_bridge)
        results[idx]["accum_did_rebridge"] = did_rebridge

        rb_tag = " [RE-BRIDGED]" if did_rebridge else ""
        label = f"OK ({sc:.2f})" if sc > 0 else "WRONG"
        print(f"  [{step+1}/{n}] ctx={task['context_len']} bridge={len(accum_bridge)} {label}{rb_tag} [{extracted[:50]}]")

    print(f"  Phase 4 done in {time.time() - t_start:.1f}s (re-bridges: {rebridge_count})")

    # ── Phase 5: Look-ahead bridge ──
    print(f"\n{'=' * 60}")
    print("PHASE 5: Look-ahead bridge (shared bridge from all contexts)")
    print(f"{'=' * 60}")
    t_start = time.time()
    # Concatenate all unique contexts
    all_contexts = "\n\n".join(ctx_map[cid] for cid in sorted(ctx_map.keys()))
    print(f"  Building shared bridge from {len(all_contexts)} chars ({len(ctx_map)} contexts)...")
    shared_bridge = build_bridge(all_contexts, temperature=TEMP)
    print(f"  Shared bridge: {len(shared_bridge)} chars ({len(shared_bridge)/len(all_contexts)*100:.1f}%)")

    for i, task in enumerate(tasks):
        ctx = task["context_window_text"]
        # Enrich shared bridge with task-specific context
        enriched = update_bridge(shared_bridge, ctx, temperature=TEMP)
        raw = answer(task["question"], enriched, temperature=TEMP)
        extracted = extract_answer(raw)
        sc = score_answer(raw, task["answer"], task["answer_type"])
        results[i]["lookahead_raw"] = raw
        results[i]["lookahead_extracted"] = extracted
        results[i]["lookahead_score"] = sc
        results[i]["lookahead_bridge_size"] = len(enriched)
        label = f"OK ({sc:.2f})" if sc > 0 else "WRONG"
        print(f"  [{i+1}/{n}] enriched={len(enriched)} {label} [{extracted[:60]}]")
    print(f"  Phase 5 done in {time.time() - t_start:.1f}s")

    # ── Phase 6: Summary table ──
    print(f"\n{'=' * 60}")
    print("5-WAY COMPARISON RESULTS")
    print(f"{'=' * 60}")

    approaches = [
        ("Vanilla (baseline)", "vanilla_score"),
        ("RLM + REPL", "rlm_score"),
        ("Per-task bridge", "pertask_score"),
        ("Accumulated bridge", "accum_score"),
        ("Look-ahead bridge", "lookahead_score"),
    ]

    print(f"\n{'Approach':<26s} {'Exact':>8s} {'Pct':>8s} {'Avg Score':>10s}")
    print("-" * 54)
    summary = {}
    for label, key in approaches:
        exact = sum(1 for r in results if r.get(key, 0) == 1.0)
        avg = sum(r.get(key, 0) for r in results) / n
        pct = exact / n * 100
        print(f"{label:<26s} {exact:>4d}/{n:<3d} {pct:>6.1f}% {avg:>10.3f}")
        summary[key.replace("_score", "")] = {"exact": exact, "total": n, "pct": round(pct, 1), "avg_score": round(avg, 3)}

    # Save
    save_path = args.save or f"results/fiveway_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"temperature": TEMP, "data_file": args.data, "num_tasks": n, "rlm_cache": args.rlm_cache},
        "summary": summary,
        "results": results,
    }
    Path(save_path).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {save_path}")


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(prog="bridge", description="Context Bridge experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build a bridge from a document")
    p_build.add_argument("document", help="Path to source document")
    p_build.add_argument("--save", help="Save bridge to file")
    p_build.set_defaults(func=cmd_build)

    p_update = sub.add_parser("update", help="Update a bridge with new content")
    p_update.add_argument("bridge", help="Path to existing bridge file")
    p_update.add_argument("new_content", help="Path to new content")
    p_update.add_argument("--save", help="Save updated bridge to file")
    p_update.set_defaults(func=cmd_update)

    p_compare = sub.add_parser("compare", help="Compare answers from bridge vs full text")
    p_compare.add_argument("bridge", help="Path to bridge file")
    p_compare.add_argument("document", help="Path to full document")
    p_compare.add_argument("question", help="Question to answer")
    p_compare.add_argument("--save", help="Save comparison to JSON file")
    p_compare.set_defaults(func=cmd_compare)

    p_run = sub.add_parser("run", help="Full experiment: build, grow, compare")
    p_run.add_argument("document", help="Path to document (will be split into chunks)")
    p_run.add_argument("questions", nargs="+", help="Questions to test at each stage")
    p_run.add_argument("--chunks", type=int, default=3, help="Number of chunks (default: 3)")
    p_run.add_argument("--save", help="Save results to JSON file")
    p_run.set_defaults(func=cmd_run)

    p_bench = sub.add_parser("benchmark", help="Run bridge on OOLONG benchmark")
    p_bench.add_argument("data", help="Path to OOLONG benchmark JSON")
    p_bench.add_argument("--limit", type=int, help="Limit number of tasks")
    p_bench.add_argument("--save", help="Save results to JSON")
    p_bench.set_defaults(func=cmd_benchmark)

    p_cont = sub.add_parser("continuous", help="Continuous-context: accumulate and bridge over time")
    p_cont.add_argument("data", help="Path to OOLONG benchmark JSON")
    p_cont.add_argument("--limit", type=int, help="Limit number of tasks")
    p_cont.add_argument("--rebridge", choices=["none", "threshold", "every5"], default="none",
                         help="Re-bridging strategy (default: none)")
    p_cont.add_argument("--save", help="Save results to JSON")
    p_cont.set_defaults(func=cmd_continuous)

    p_five = sub.add_parser("fiveway", help="5-way comparison experiment")
    p_five.add_argument("data", nargs="?", default="benchmarks/oolong_trec_coarse_30.json")
    p_five.add_argument("--limit", type=int, help="Limit number of tasks")
    p_five.add_argument("--rlm-cache", help="Cached RLM results JSON")
    p_five.add_argument("--save", help="Save results JSON")
    p_five.set_defaults(func=cmd_fiveway)

    args = parser.parse_args()

    try:
        args.func(args)
    except Exception as e:
        if "Connection" in type(e).__name__ or "connection" in str(e).lower():
            print(
                "Could not connect to vllm-mlx.\n"
                "Start it with: vllm-mlx serve mlx-community/Qwen3-8B-4bit --port 8000 --continuous-batching",
                file=sys.stderr,
            )
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
