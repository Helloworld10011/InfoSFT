"""
Evaluation script for the Science Q&A (Chemistry L-3) dataset.

Usage:
    python eval/Science.py --model_path /path/to/model --exp_name my_experiment \
        --data_path /path/to/science_data/eval_data
"""

import argparse
import json
import os
import re
from typing import Optional

import torch
from datasets import load_from_disk
from vllm import LLM, SamplingParams

VALID_LETTERS = {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
def extract_answer(text: str) -> Optional[str]:
    """Extract A/B/C/D from a model response with multiple fallbacks."""
    if not text:
        return None

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 1. Strict: last <answer>...</answer> block
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        cand = _normalize_letter(matches[-1], case_sensitive=False)
        if cand:
            return cand

    # 2. Unclosed <answer> tag (truncated generation)
    m = re.search(r"<answer>\s*(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        cand = _normalize_letter(m.group(1), case_sensitive=False)
        if cand:
            return cand

    # 3. Explicit cue words
    cue_patterns = [
        r"(?:final\s+answer|the\s+answer\s+is|answer\s*[:\-])\s*\(?\*?\*?([ABCDabcd])\b",
        r"\boption\s*\(?\*?\*?([ABCDabcd])\b",
        r"\b([ABCDabcd])\s*(?:is\s+(?:the\s+)?correct|is\s+right)",
        r"\\boxed\{\s*\(?([ABCDabcd])\)?\s*\}",
    ]
    for pat in cue_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 4. Leading "A." / "(A)" on a line — uppercase only
    m = re.search(r"^\s*\(?\*?\*?([ABCD])\)?\s*[\.\):]", text, flags=re.MULTILINE)
    if m:
        return m.group(1)

    # 5. Last-resort bare letter — uppercase only
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip().strip("*`()[].:\"' ")
        if stripped in VALID_LETTERS:
            return stripped

    return None


def _normalize_letter(snippet: str, case_sensitive: bool = False) -> Optional[str]:
    if not snippet:
        return None
    cleaned = re.sub(r"[\*\(\)\[\]\.\:`\"']", " ", snippet).strip()

    if case_sensitive:
        if cleaned in VALID_LETTERS:
            return cleaned
        m = re.match(r"\s*([ABCD])\b", cleaned)
        return m.group(1) if m else None
    else:
        if cleaned.upper() in VALID_LETTERS:
            return cleaned.upper()
        m = re.match(r"\s*([ABCD])\b", cleaned, flags=re.IGNORECASE)
        return m.group(1).upper() if m else None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_eval(model_path: str, eval_ds, max_new_tokens: int = 2048):
    num_gpus = torch.cuda.device_count()
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        trust_remote_code=True,
    )

    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    prompts = [
        tokenizer.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        for ex in eval_ds
    ]

    outputs = llm.generate(prompts, sampling)

    n_total = len(eval_ds)
    n_correct = 0
    n_parse_fail = 0
    per_letter_correct = {l: 0 for l in VALID_LETTERS}
    per_letter_total = {l: 0 for l in VALID_LETTERS}

    for ex, out in zip(eval_ds, outputs):
        gen = out.outputs[0].text
        pred = extract_answer(gen)
        gold = ex["answer"].strip().upper()

        per_letter_total[gold] = per_letter_total.get(gold, 0) + 1
        if pred is None:
            n_parse_fail += 1
        elif pred == gold:
            n_correct += 1
            per_letter_correct[gold] = per_letter_correct.get(gold, 0) + 1

    return {
        "accuracy": n_correct / n_total,
        "n_correct": n_correct,
        "n_total": n_total,
        "n_parse_fail": n_parse_fail,
        "parse_fail_rate": n_parse_fail / n_total,
        "per_letter_accuracy": {
            l: (per_letter_correct[l] / per_letter_total[l] if per_letter_total[l] else None)
            for l in VALID_LETTERS
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on Science Q&A.")
    parser.add_argument("--model_path",     type=str, required=True)
    parser.add_argument("--exp_name",       type=str, required=True)
    parser.add_argument("--data_path",      type=str, default="data/science_data/eval_datal",
                        help="Path to preprocessed eval dataset (load_from_disk)")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--results_file",   type=str, default="eval/results/science/eval_results.jsonl")
    args = parser.parse_args()

    print(f"[{args.exp_name}] Loading dataset from {args.data_path}")
    eval_ds = load_from_disk(args.data_path)
    print(f"[{args.exp_name}] Dataset size: {len(eval_ds)}")

    print(f"[{args.exp_name}] Loading model from {args.model_path}")
    stats = run_eval(args.model_path, eval_ds, max_new_tokens=args.max_new_tokens)

    record = {
        "exp_name": args.exp_name,
        "model_path": args.model_path,
        **stats,
    }

    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    with open(args.results_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    print("\n=== Results ===")
    print(f"Experiment:        {args.exp_name}")
    print(f"Accuracy:          {stats['accuracy']:.4f}  ({stats['n_correct']}/{stats['n_total']})")
    print(f"Parse failures:    {stats['n_parse_fail']}/{stats['n_total']} "
          f"({stats['parse_fail_rate']:.2%})")
    print(f"Per-letter acc.:   {stats['per_letter_accuracy']}")
    print(f"Appended to:       {args.results_file}")


if __name__ == "__main__":
    main()