"""
OpenR1-Math-220k experiment (further studies).
Qwen2.5-7B-Instruct · full fine-tuning · 2 epochs · max_length 8192

Usage:
  bash training/scripts/run_openR1.sh Qwen/Qwen2.5-7B-Instruct nll 5e-6
"""

import argparse
import os
import random
from functools import partial

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from infosft_loss import dft_loss, infosft_normalized_loss

MATH_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def build_openr1_dataset(model_id, max_tokens=8192):
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    def filter_length(ex):
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        return len(tokenizer(text)["input_ids"]) < max_tokens

    ds = ds.filter(filter_length, num_proc=64)

    def prepend_system(ex):
        ex["messages"] = [
            {"role": "system", "content": MATH_INSTRUCTION}
        ] + ex["messages"]
        return ex

    return ds.map(prepend_system, num_proc=8)


def get_loss_fn(loss_type, P):
    if loss_type == "nll":
        return None
    if loss_type == "infosft":
        return partial(infosft_normalized_loss, P=P)
    if loss_type == "dft":
        return dft_loss
    raise ValueError(f"Unknown loss type: {loss_type}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name",    type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--model_id",    type=str, required=True)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--loss_type",   type=str, default="nll",
                   choices=["nll", "infosft", "dft"])
    p.add_argument("--P",           type=float, default=0.93)
    p.add_argument("--lr",          type=float, default=5e-6)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--deepspeed_config",   type=str, default=None)
    p.add_argument("--chat_template_path", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    exp_name   = args.exp_name + (f"_P{args.P}" if args.loss_type == "infosft" else "")
    output_dir = os.path.join(args.output_root, exp_name)

    train_ds = build_openr1_dataset(args.model_id)

    # auto-detect world size from accelerate launcher
    world_size    = int(os.environ.get("WORLD_SIZE", 1))
    per_device_bs = 4                       # 8192 seqlen → small batch
    grad_accum    = args.batch_size // (per_device_bs * world_size)
    assert grad_accum >= 1

    sft_args = SFTConfig(
        seed=args.seed,
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=8192,
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
    )
    trainer.train()


if __name__ == "__main__":
    main()