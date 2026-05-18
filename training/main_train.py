"""
Unified SFT training script for InfoSFT experiments. These experiments conclude the results of Table 1 in the infoSFT paper: 
https://arxiv.org/abs/2605.14967

Effective batch size = per_device_batch_size × grad_accum × num_gpus
  → grad_accum is computed automatically from --batch_size and --num_gpus.

Examples:
  accelerate launch --multi_gpu --num_processes ... training/train.py \
    --task math --model_id Qwen/Qwen2.5-Math-7B --model_family qwen \
    --exp_name qwen7b_math_infosft --loss_type infosft --P 0.93 \
    --lr 2e-5 --batch_size 256 ...
"""

import argparse
import os
import random
from functools import partial

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from infosft_loss import dft_loss, infosft_normalized_loss


# ------------------------------------------------------------------ #
#  Data                                                               #
# ------------------------------------------------------------------ #

MATH_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

CODE_HINTS = (
    "code", "function", "program", "script", "python", "java",
    "javascript", "c++", "sql", "html", "algorithm", "```",
)


def build_math_dataset(num_samples: int, seed: int, model_id: str):
    """NuminaMath-CoT, prompt format depends on model family."""
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))

    def general_format(ex):
        ex["messages"] = [
            {"role": "system",    "content": MATH_INSTRUCTION},
            {"role": "user",      "content": ex["problem"]},
            {"role": "assistant", "content": ex["solution"]},
        ]
        return ex

    def format_llama(ex):
        ex["messages"] = [
            {"role": "user",      "content": f"{MATH_INSTRUCTION}\n\n{ex['problem']}"},
            {"role": "assistant", "content": ex["solution"]},
        ]
        return ex

    fmt = format_llama if "llama" in model_id else general_format
    return ds.map(fmt, num_proc=8)


def build_code_dataset(num_samples: int, seed: int):
    """UltraFeedback filtered to code-related prompts, best completion."""
    raw = load_dataset("openbmb/UltraFeedback")["train"]
    raw = raw.filter(lambda x: len(x["completions"]) > 0)

    def _score(c):
        a = c.get("annotations", {})
        try:
            return (float(a["instruction_following"]["Rating"])
                    + float(a["helpfulness"]["Rating"]))
        except (KeyError, TypeError, ValueError):
            return -1

    def _build(ex):
        if not any(h in ex["instruction"].lower() for h in CODE_HINTS):
            return {"keep": False, "prompt": None, "completion": None}
        best = max(ex["completions"], key=_score)
        if _score(best) < 0:
            return {"keep": False, "prompt": None, "completion": None}
        return {
            "keep": True,
            "prompt": [
                {"role": "system", "content": ""},
                {"role": "user",   "content": ex["instruction"]},
            ],
            "completion": [
                {"role": "assistant", "content": best["response"]},
            ],
        }

    ds = raw.map(_build, remove_columns=raw.column_names, num_proc=8)
    ds = ds.filter(lambda x: x["keep"]).remove_columns("keep")
    return ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))


# ------------------------------------------------------------------ #
#  Loss                                                               #
# ------------------------------------------------------------------ #

def get_loss_fn(loss_type: str, P: float):
    if loss_type == "nll":
        return None
    if loss_type == "infosft":
        return partial(infosft_normalized_loss, P=P)
    if loss_type == "dft":
        return dft_loss
    raise ValueError(f"Unknown loss type: {loss_type}")


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser()

    # experiment
    p.add_argument("--exp_name",    type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--seed",        type=int, default=42)

    # task & model
    p.add_argument("--task",         type=str, required=True, choices=["math", "code"])
    p.add_argument("--model_id",     type=str, required=True)

    # paths
    p.add_argument("--chat_template_path", type=str, default=None)
    p.add_argument("--deepspeed_config",   type=str, default=None)

    # loss
    p.add_argument("--loss_type", type=str, default="nll", choices=["nll", "infosft", "dft"])
    p.add_argument("--P",        type=float, default=0.93)

    # optimization
    p.add_argument("--lr",           type=float, required=True)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length",   type=int,   default=4096)

    # batch size  (effective = per_device_bs × grad_accum × num_gpus)
    p.add_argument("--batch_size",            type=int, required=True,
                   help="Total effective batch size across all GPUs")
    p.add_argument("--per_device_batch_size", type=int, default=8)

    # data
    p.add_argument("--num_samples", type=int, default=100000)

    # LoRA (off by default → full fine-tuning)
    p.add_argument("--use_lora",     action="store_true")
    p.add_argument("--lora_r",       type=int,   default=32)
    p.add_argument("--lora_alpha",   type=int,   default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    return p.parse_args()


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    num_gpus = torch.cuda.device_count()
    args = parse_args()

    # reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # naming
    exp_name   = args.exp_name + (f"_P{args.P}" if args.loss_type == "infosft" else "")
    output_dir = os.path.join(args.output_root, exp_name)

    # batch size → gradient accumulation
    assert args.batch_size % (args.per_device_batch_size * num_gpus) == 0, (
        f"batch_size ({args.batch_size}) must be divisible by "
        f"per_device_batch_size × num_gpus "
        f"({args.per_device_batch_size} × {num_gpus})"
    )
    grad_accum = args.batch_size // (args.per_device_batch_size * num_gpus)

    # dataset
    if args.task == "math":
        train_ds = build_math_dataset(args.num_samples, args.seed, args.model_id)
    else:
        train_ds = build_code_dataset(args.num_samples, args.seed)

    # LoRA (optional)
    peft_cfg = None
    if args.use_lora:
        peft_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "up_proj", "gate_proj", "down_proj"],
        )

    # SFT config (fixed: cosine schedule, 1 epoch, bf16, grad ckpt)
    sft_args = SFTConfig(
        seed=args.seed,
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        bf16=True,
        loss_type="nll",
        assistant_only_loss=True,
        logging_strategy="steps", logging_steps=10, logging_first_step=True,
        save_strategy="epoch", save_only_model=True,
        report_to=["wandb"], run_name=exp_name,
        deepspeed=args.deepspeed_config,
        chat_template_path=args.chat_template_path,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "use_cache": False,
            "attn_implementation": "flash_attention_2",
        },
    )

    trainer = SFTTrainer(
        model=args.model_id,
        args=sft_args,
        train_dataset=train_ds,
        compute_loss_func=get_loss_fn(args.loss_type, args.P),
        peft_config=peft_cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()