"""
RLM baseline on OOLONG-synth trec_coarse benchmark.

Runs RLM (with REPL) on OOLONG semantic aggregation tasks.
Compares to base model (no REPL, direct prompting).

Requires vllm-mlx running:
    vllm-mlx serve mlx-community/Qwen3-8B-4bit --port 8000 --continuous-batching

Usage:
    python rlm_oolong.py                         # Run small subset (context_len <= 4096)
    python rlm_oolong.py --limit 5               # Run first 5 tasks only
    python rlm_oolong.py --mode base              # Base model only (no REPL)
    python rlm_oolong.py --mode rlm               # RLM only (with REPL)
    python rlm_oolong.py --mode both              # Both (default)
    python rlm_oolong.py --data benchmarks/oolong_trec_coarse_full.json  # Full dataset
"""

import argparse
import ast
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.prompts import RLM_SYSTEM_PROMPT


# --- Data loading ---

def parse_answer_field(answer_raw) -> list[str]:
    """Parse answer field which may be a list or a string repr of a list."""
    if isinstance(answer_raw, list):
        return answer_raw
    if isinstance(answer_raw, str):
        try:
            parsed = ast.literal_eval(answer_raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
        return [answer_raw]
    return [str(answer_raw)]


# --- Scoring (matches OOLONG eval: synth_attempt_answer_parse + comparison) ---

def extract_answer(text: str) -> str:
    """Extract answer from model output, matching OOLONG's synth_attempt_answer_parse.

    Strategy: find last colon, take everything after it. Clean up formatting.
    Fallback: if short (<20 chars) return whole thing, else return last word.
    """
    text = text.strip()
    # Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Remove code blocks (RLM artifacts)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL).strip()
    # Take last line that looks like an answer (skip empty lines)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return ""
    # Look for answer-bearing lines from the end
    for line in reversed(lines):
        for prefix in ["label:", "answer:", "final answer:"]:
            if prefix in line.lower():
                text = line
                break
        else:
            continue
        break
    else:
        text = lines[-1]

    # OOLONG's primary strategy: extract after last colon
    if ':' in text:
        text = text.rsplit(':', 1)[1]

    # Clean formatting markers
    text = text.replace('*', '').replace('[', '').replace(']', '')
    text = text.strip().strip('.').strip()
    return text


def score_answer(predicted: str, gold: list[str], answer_type: str) -> float:
    """Score predicted answer against gold. Returns 0.0-1.0.

    Matches OOLONG evaluation:
    - LABEL/COMPARISON: case-insensitive exact match
    - NUMERIC: partial credit via 0.75^(abs(gold - pred))
    """
    if not predicted:
        return 0.0
    pred = extract_answer(predicted).lower()

    # For comparison answers, normalize variants
    if answer_type == "ANSWER_TYPE.COMPARISON":
        for variant, canonical in [
            ("more common", "more common than"),
            ("less common", "less common than"),
            ("same frequency", "same frequency as"),
        ]:
            if variant in pred:
                pred = canonical
                break

    # Exact match for labels/comparisons
    for g in gold:
        g_norm = g.strip().lower()
        if pred == g_norm:
            return 1.0

    # For numeric answers, partial credit
    if answer_type == "ANSWER_TYPE.NUMERIC":
        pred_nums = re.findall(r'\d+', pred)
        if pred_nums:
            try:
                pred_val = int(pred_nums[-1])
                for g in gold:
                    gold_nums = re.findall(r'\d+', g)
                    if gold_nums:
                        gold_val = int(gold_nums[0])
                        return 0.75 ** abs(gold_val - pred_val)
            except ValueError:
                pass

    # Fallback: substring containment (lenient)
    for g in gold:
        g_norm = g.strip().lower()
        if g_norm in pred:
            return 1.0

    return 0.0


# --- Base model (direct prompting, no REPL) ---

def run_base_model(tasks: list[dict], client: OpenAI, model: str) -> list[dict]:
    """Run tasks through base model with direct prompting."""
    results = []
    for i, task in enumerate(tasks):
        prompt = f"{task['context_window_text']}\n\n{task['question']}"
        print(f"  Base [{i+1}/{len(tasks)}] ctx={task['context_len']} "
              f"task={task['task']} ...", end="", flush=True)

        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "/no_think\n" + prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            answer = response.choices[0].message.content
            # Strip any residual <think>...</think> blocks from Qwen3
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
            elapsed = time.time() - t0
            score = score_answer(answer, task["answer"], task["answer_type"])
            extracted = extract_answer(answer)
            label = f"OK ({score:.2f})" if score > 0 else "WRONG"
            print(f" {label} ({elapsed:.1f}s) [{extracted[:60]}]")
            results.append({
                "id": task["id"],
                "mode": "base",
                "context_len": task["context_len"],
                "task": task["task"],
                "task_group": task["task_group"],
                "question": task["question"],
                "gold": task["answer"],
                "predicted": answer,
                "extracted": extracted,
                "score": score,
                "time_s": elapsed,
            })
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                "id": task["id"],
                "mode": "base",
                "context_len": task["context_len"],
                "task": task["task"],
                "task_group": task["task_group"],
                "question": task["question"],
                "gold": task["answer"],
                "predicted": None,
                "extracted": "",
                "score": 0.0,
                "time_s": 0,
                "error": str(e),
            })
    return results


# --- RLM (with REPL) ---

def run_rlm(tasks: list[dict], rlm_instance: RLM) -> list[dict]:
    """Run tasks through RLM with REPL."""
    results = []
    for i, task in enumerate(tasks):
        print(f"  RLM  [{i+1}/{len(tasks)}] ctx={task['context_len']} "
              f"task={task['task']} ...", end="", flush=True)

        t0 = time.time()
        try:
            # Separate context (→ REPL `context` var) from question (→ iteration prompts)
            result = rlm_instance.completion(
                task["context_window_text"], root_prompt=task["question"]
            )
            # RLM returns RLMChatCompletion object, extract the response string
            answer = result.response if hasattr(result, "response") else str(result)
            elapsed = time.time() - t0
            score = score_answer(answer, task["answer"], task["answer_type"])
            extracted = extract_answer(answer)
            label = f"OK ({score:.2f})" if score > 0 else "WRONG"
            print(f" {label} ({elapsed:.1f}s) [{extracted[:60]}]")
            results.append({
                "id": task["id"],
                "mode": "rlm",
                "context_len": task["context_len"],
                "task": task["task"],
                "task_group": task["task_group"],
                "question": task["question"],
                "gold": task["answer"],
                "predicted": answer,
                "extracted": extracted,
                "score": score,
                "time_s": elapsed,
            })
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                "id": task["id"],
                "mode": "rlm",
                "context_len": task["context_len"],
                "task": task["task"],
                "task_group": task["task_group"],
                "question": task["question"],
                "gold": task["answer"],
                "predicted": None,
                "extracted": "",
                "score": 0.0,
                "time_s": 0,
                "error": str(e),
            })
    return results


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="RLM baseline on OOLONG-synth")
    parser.add_argument("--data", default="benchmarks/oolong_trec_coarse.json",
                        help="Path to benchmark JSON")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of tasks")
    parser.add_argument("--mode", choices=["base", "rlm", "both"], default="both",
                        help="Which mode to run")
    parser.add_argument("--save", default=None,
                        help="Save results to JSON (default: results/rlm_oolong_TIMESTAMP.json)")
    args = parser.parse_args()

    # Load tasks
    tasks = json.loads(Path(args.data).read_text())
    # Parse answer field (may be string repr of list from HuggingFace export)
    for task in tasks:
        task["answer"] = parse_answer_field(task["answer"])
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"Loaded {len(tasks)} tasks from {args.data}")

    all_results = []

    # Base model
    if args.mode in ("base", "both"):
        print(f"\n{'=' * 60}")
        print("BASE MODEL (direct prompting, no REPL)")
        print(f"{'=' * 60}")
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed", timeout=1800)
        base_results = run_base_model(tasks, client, "default")
        all_results.extend(base_results)

        avg_score = sum(r["score"] for r in base_results) / len(base_results)
        exact = sum(1 for r in base_results if r["score"] == 1.0)
        print(f"\nBase: {exact}/{len(base_results)} exact match = {exact/len(base_results)*100:.1f}%, avg_score={avg_score:.3f}")

    # RLM
    if args.mode in ("rlm", "both"):
        print(f"\n{'=' * 60}")
        print("RLM (with REPL)")
        print(f"{'=' * 60}")
        logger = RLMLogger(log_dir="./logs")
        custom_prompt = (
            RLM_SYSTEM_PROMPT
            + "\n\nCRITICAL: Never use <think> tags. Write ```repl``` code blocks immediately. "
            "Always end with FINAL(your_answer) when you have the answer."
        )
        # Monkey-patch OpenAIClient to inject /no_think into user messages for Qwen3
        from rlm.clients.openai import OpenAIClient
        if not hasattr(OpenAIClient, '_original_completion'):
            OpenAIClient._original_completion = OpenAIClient.completion
            def _patched_completion(self, prompt, model=None):
                if isinstance(prompt, list):
                    for msg in reversed(prompt):
                        if msg.get("role") == "user":
                            msg["content"] = "/no_think\n" + msg["content"]
                            break
                return OpenAIClient._original_completion(self, prompt, model=model)
            OpenAIClient.completion = _patched_completion

        rlm_instance = RLM(
            backend="openai",
            backend_kwargs={
                "model_name": "default",
                "base_url": "http://localhost:8000/v1",
                "api_key": "not-needed",
            },
            environment="local",
            max_depth=1,
            max_iterations=30,
            custom_system_prompt=custom_prompt,
            logger=logger,
            verbose=True,
        )
        rlm_results = run_rlm(tasks, rlm_instance)
        all_results.extend(rlm_results)

        avg_score = sum(r["score"] for r in rlm_results) / len(rlm_results)
        exact = sum(1 for r in rlm_results if r["score"] == 1.0)
        print(f"\nRLM: {exact}/{len(rlm_results)} exact match = {exact/len(rlm_results)*100:.1f}%, avg_score={avg_score:.3f}")

    # Save results
    save_path = args.save or f"results/rlm_oolong_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "data_file": args.data,
        "num_tasks": len(tasks),
        "mode": args.mode,
        "results": all_results,
    }
    Path(save_path).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {save_path}")

    # Summary
    if args.mode == "both":
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        for mode in ("base", "rlm"):
            mode_results = [r for r in all_results if r["mode"] == mode]
            exact = sum(1 for r in mode_results if r["score"] == 1.0)
            avg_score = sum(r["score"] for r in mode_results) / len(mode_results)
            total_time = sum(r["time_s"] for r in mode_results)
            print(f"{mode.upper():>5}: {exact}/{len(mode_results)} exact = "
                  f"{exact/len(mode_results)*100:.1f}%, "
                  f"avg={avg_score:.3f} "
                  f"({total_time:.0f}s total)")


if __name__ == "__main__":
    main()
