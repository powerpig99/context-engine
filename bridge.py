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

def build_bridge(text: str) -> str:
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
    )


def update_bridge(bridge: str, new_text: str) -> str:
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
    )


def answer(question: str, context: str) -> str:
    """Answer a question using the given context."""
    return generate(
        system_prompt=(
            "Answer the question using the provided context. "
            "Derive your answer from the context. Be direct and clear."
        ),
        user_prompt=f"CONTEXT:\n{context}\n\nQUESTION:\n{question}",
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
