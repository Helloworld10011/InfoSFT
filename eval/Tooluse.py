"""
Evaluation script for the ToolAlpaca (Tool Use) dataset.
Uses repo-style scoring: multi-set Action match + merged Action_Input dict equality.

Usage:
    python eval/Tooluse.py --model_path /path/to/model --exp_name my_experiment \
        --data_path /path/to/tooluse_data/eval_data
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Optional

import torch
from datasets import load_from_disk
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Action / JSON parsing  (balanced-brace extractor)
# ---------------------------------------------------------------------------
ACTION_RE = re.compile(
    r"^\s*Action\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
ACTION_INPUT_RE = re.compile(r"Action\s*Input\s*:", flags=re.IGNORECASE)


def find_balanced_json(text: str, start_idx: int) -> Optional[str]:
    obj_start, opening, closing = None, None, None
    for i in range(start_idx, len(text)):
        if text[i] == "{":
            obj_start, opening, closing = i, "{", "}"
            break
        if text[i] == "[":
            obj_start, opening, closing = i, "[", "]"
            break
    if obj_start is None:
        return None

    depth, in_str, escape = 0, False, False
    for i in range(obj_start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[obj_start : i + 1]
    return None


def parse_json_strict(json_text: Optional[str]) -> Optional[Any]:
    if json_text is None:
        return None
    s = json_text.strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def extract_tool_calls(response: str):
    response = re.sub(r"<think>.*?</think>", "", response,
                      flags=re.DOTALL | re.IGNORECASE)

    calls = []
    action_matches = list(ACTION_RE.finditer(response))

    for idx, am in enumerate(action_matches):
        action = am.group(1).strip()
        chunk_start = am.end()
        chunk_end = (action_matches[idx + 1].start()
                     if idx + 1 < len(action_matches) else len(response))
        chunk = response[chunk_start:chunk_end]

        im = ACTION_INPUT_RE.search(chunk)
        if not im:
            calls.append({"Action": action, "Action_Input": None})
            continue

        json_start = chunk_start + im.end()
        json_text = find_balanced_json(response, json_start)
        calls.append({
            "Action": action,
            "Action_Input": parse_json_strict(json_text),
        })
    return calls


def parse_golden_answer(golden_answer):
    out = []
    for item in golden_answer:
        action = item["Action"].strip()
        ai = item["Action_Input"]
        parsed = json.loads(ai) if isinstance(ai, str) else ai
        out.append({"Action": action, "Action_Input": parsed})
    return out


# ---------------------------------------------------------------------------
# Repo-style scoring
# ---------------------------------------------------------------------------
def merged_inputs(calls):
    merged = {}
    for call in calls:
        ai = call["Action_Input"]
        if isinstance(ai, dict):
            merged.update(ai)
        elif ai is None:
            pass
        else:
            merged["__non_dict_action_input__"] = ai
    return merged


def score_response(response: str, golden_answer) -> dict:
    pred_calls = extract_tool_calls(response)
    gold_calls = parse_golden_answer(golden_answer)

    pred_actions = [c["Action"] for c in pred_calls]
    gold_actions = [c["Action"] for c in gold_calls]
    action_match = Counter(pred_actions) == Counter(gold_actions)

    input_match = merged_inputs(pred_calls) == merged_inputs(gold_calls)

    return {
        "correct":       int(action_match and input_match),
        "action_match":  int(action_match),
        "input_match":   int(input_match),
        "n_pred_calls":  len(pred_calls),
        "n_gold_calls":  len(gold_calls),
    }


# ---------------------------------------------------------------------------
# Eval loop
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

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop=["\nObservation:", "Observation:"],
    )

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in eval_ds
    ]
    outputs = llm.generate(prompts, sampling)

    n_total        = len(eval_ds)
    n_correct      = 0
    n_action_match = 0
    n_input_match  = 0
    n_parse_fail   = 0

    for ex, out in zip(eval_ds, outputs):
        gen = out.outputs[0].text
        s = score_response(gen, ex["golden_answer"])
        n_correct      += s["correct"]
        n_action_match += s["action_match"]
        n_input_match  += s["input_match"]
        if s["n_pred_calls"] == 0:
            n_parse_fail += 1

    return {
        "accuracy":        n_correct / n_total,
        "action_accuracy": n_action_match / n_total,
        "input_accuracy":  n_input_match / n_total,
        "n_correct":       n_correct,
        "n_action_match":  n_action_match,
        "n_input_match":   n_input_match,
        "n_total":         n_total,
        "n_parse_fail":    n_parse_fail,
        "parse_fail_rate": n_parse_fail / n_total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on ToolAlpaca.")
    parser.add_argument("--model_path",     type=str, required=True)
    parser.add_argument("--exp_name",       type=str, required=True)
    parser.add_argument("--data_path",      type=str, default="data/tooluse_data/eval_data",
                        help="Path to preprocessed eval dataset (load_from_disk)")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--results_file",   type=str, default="eval/results/tool/eval_results.jsonl")
    args = parser.parse_args()

    print(f"[{args.exp_name}] Loading dataset from {args.data_path}")
    eval_ds = load_from_disk(args.data_path)
    print(f"[{args.exp_name}] Dataset size: {len(eval_ds)}")

    print(f"[{args.exp_name}] Loading model from {args.model_path}")
    stats = run_eval(args.model_path, eval_ds, max_new_tokens=args.max_new_tokens)

    record = {
        "exp_name":   args.exp_name,
        "model_path": args.model_path,
        **stats,
    }
    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    with open(args.results_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    print("\n=== Results ===")
    print(f"Experiment:       {args.exp_name}")
    print(f"Full Accuracy:    {stats['accuracy']:.4f}  ({stats['n_correct']}/{stats['n_total']})")
    print(f"Action Accuracy:  {stats['action_accuracy']:.4f}  ({stats['n_action_match']}/{stats['n_total']})")
    print(f"Input  Accuracy:  {stats['input_accuracy']:.4f}  ({stats['n_input_match']}/{stats['n_total']})")
    print(f"Parse failures:   {stats['n_parse_fail']}/{stats['n_total']} ({stats['parse_fail_rate']:.2%})")
    print(f"Appended to:      {args.results_file}")


if __name__ == "__main__":
    main()