import argparse
import json
import os
from math import comb

import numpy as np
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from math_grader import boxed_reward_fn, extract_answer

# ── Config ──
MAX_K = 8
KS = [1, 2, 4, 8]
PASS_TEMPERATURE = 0.7
ACC_TEMPERATURE = 0.0

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  type=str, required=True)
    p.add_argument("--exp_name",    type=str, required=True)
    p.add_argument("--dataset",     type=str, required=True,
                   choices=["aime", "amc", "math500", "hmmt", "minerva"])
    p.add_argument("--base_folder", type=str, default="eval/results")
    p.add_argument("--max_tokens",  type=int, default=2048,
                   help="Use 16384 for long-CoT models (e.g. OpenR1)")
    args = p.parse_args()

    ks = [k for k in KS if k <= MAX_K]

    # ── Load dataset ──
    if args.dataset == "aime":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        base_dir = f"{args.base_folder}/aime"
    elif args.dataset == "amc":
        dataset = load_dataset("AI-MO/aimo-validation-amc", split="train")
        base_dir = f"{args.base_folder}/amc"
    elif args.dataset == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        base_dir = f"{args.base_folder}/math500"
    elif args.dataset == "hmmt":
        dataset = load_dataset("MathArena/hmmt_feb_2024", split="train")
        base_dir = f"{args.base_folder}/hmmt"
    elif args.dataset == "minerva":
        dataset = load_dataset("math-ai/minervamath", split="test")
        dataset = dataset.rename_column("question", "problem")
        base_dir = f"{args.base_folder}/minerva"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # ── Load model ──
    num_gpus = torch.cuda.device_count()
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=num_gpus,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    # ── Build prompts ──
    prompts = []
    for item in dataset:
        if "llama" in args.model_path.lower():
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Problem:\n{item['problem']}\n\n"
                f"Solution:\n"
            )
            prompts.append(prompt)
        else:
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user",   "content": item["problem"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

    pass_sampling = SamplingParams(
        temperature=PASS_TEMPERATURE, max_tokens=args.max_tokens, n=MAX_K,
    )
    acc_sampling = SamplingParams(
        temperature=ACC_TEMPERATURE, max_tokens=args.max_tokens, n=1,
    )

    # ── Generate ──
    pass_outputs = llm.generate(prompts, pass_sampling)
    acc_outputs  = llm.generate(prompts, acc_sampling)

    # ── Score & Save ──
    data_path = f"{base_dir}/{args.exp_name}.jsonl"
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if os.path.exists(data_path):
        os.remove(data_path)

    all_n_correct = []
    acc_corrects  = []

    for pass_output, acc_output, item in zip(pass_outputs, acc_outputs, dataset):
        item = dict(item)
        gold = item["answer"]

        # pass@k samples
        pass_solutions = [comp.text for comp in pass_output.outputs]
        pass_answers   = []
        pass_corrects  = []
        for sol in pass_solutions:
            _, reward = boxed_reward_fn(sol, gold)
            pass_answers.append(extract_answer(sol))
            pass_corrects.append(bool(reward > 0))

        # acc@1 deterministic sample
        acc_solution = acc_output.outputs[0].text
        acc_answer   = extract_answer(acc_solution)
        _, acc_reward = boxed_reward_fn(acc_solution, gold)
        acc_correct   = bool(acc_reward > 0)

        item["pass_predicted_answers"] = pass_answers
        item["pass_correct"]           = pass_corrects
        item["pass_solutions"]         = pass_solutions
        item["acc_predicted_answer"]   = acc_answer
        item["acc_correct"]            = acc_correct
        item["acc_solution"]           = acc_solution

        with open(data_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        all_n_correct.append(sum(pass_corrects))
        acc_corrects.append(acc_correct)

    n_problems = len(dataset)

    # ── Summary ──
    summary = {
        "model": args.exp_name,
        "pass_temperature": PASS_TEMPERATURE,
        "acc_temperature": ACC_TEMPERATURE,
        "n": MAX_K,
    }
    for k in ks:
        vals = [pass_at_k(MAX_K, c, k) for c in all_n_correct]
        summary[f"pass@{k}"] = float(np.mean(vals))

    summary["acc@1_temp0"]           = float(np.mean(acc_corrects))
    summary["n_correct_acc@1_temp0"] = int(sum(acc_corrects))
    summary["n_total"]               = n_problems
    summary["avg_sample_acc"]        = float(np.mean(all_n_correct) / MAX_K)

    print(f"\n{'=' * 50}")
    print(f"  {args.exp_name}")
    print(f"  pass@k: n={MAX_K}, T={PASS_TEMPERATURE}")
    print(f"  acc@1:  n=1, T={ACC_TEMPERATURE}")
    print(f"{'=' * 50}")
    for k in ks:
        print(f"pass@{k}: {summary[f'pass@{k}']:.4f}")
    print(f"acc@1_temp0: {summary['acc@1_temp0']:.4f} "
          f"({summary['n_correct_acc@1_temp0']}/{n_problems})")
    print(f"avg_sample_acc: {summary['avg_sample_acc']:.4f}")

    summary_path = f"{base_dir}/summary.jsonl"
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()