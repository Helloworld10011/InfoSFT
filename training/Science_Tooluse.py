"""
Hyperparameter sweep experiments: science & tool-use (further studies).
Qwen2.5-7B-Instruct · full fine-tuning · sweep over {loss} × {epochs} × {lr}

Usage:
  bash training/scripts/run_science_sweep.sh Qwen/Qwen2.5-7B-Instruct /path/to/data
  bash training/scripts/run_tool_sweep.sh    Qwen/Qwen2.5-7B-Instruct /path/to/data
"""

import argparse
import os
import random
from functools import partial

import torch
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer

from infosft_loss import dft_loss, infosft_normalized_loss

# per_device_bs is fixed per task (memory-bound); grad_accum adjusts to GPU count
TASK_CONFIGS = {
    "science": {"per_device_bs": 4, "max_length": 2048},
    "tool":    {"per_device_bs": 8, "max_length": 4096},
}


# ------------------------------------------------------------------ #
#  Data                                                               #
# ------------------------------------------------------------------ #

def prepare_science(data_path):
    ds = load_from_disk(data_path)

    def convert(ex):
        msgs = list(ex["messages"])
        msgs.append({"role": "assistant", "content": ex["output_text"]})
        return {"messages": msgs}

    return ds.map(convert)


def prepare_tool(data_path):
    ds = load_from_disk(data_path)

    def convert(ex):
        return {
            "messages": [
                {"role": "user",      "content": ex["prompt"].strip()},
                {"role": "assistant", "content": "\n".join(ex["golden_response"]).strip()},
            ]
        }

    return ds.map(convert, remove_columns=ds.column_names)


# ------------------------------------------------------------------ #
#  Loss                                                               #
# ------------------------------------------------------------------ #

def get_loss_fn(loss_type, P):
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
    p.add_argument("--task",        type=str, required=True, choices=["science", "tool"])
    p.add_argument("--data_path",   type=str, required=True,
                   help="Path to preprocessed dataset (load_from_disk)")
    p.add_argument("--model_id",    type=str, required=True)
    p.add_argument("--exp_name",    type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--epochs",      type=int,   default=2)
    p.add_argument("--lr",          type=float, required=True)
    p.add_argument("--batch_size",  type=int,   required=True,
                   help="Total effective batch size (science: 16, tool: 32)")
    p.add_argument("--loss_type",   type=str,   default="nll",
                   choices=["nll", "infosft", "dft"])
    p.add_argument("--P",           type=float, default=0.93)
    p.add_argument("--deepspeed_config",   type=str, default=None)
    p.add_argument("--chat_template_path", type=str, default=None)
    return p.parse_args()


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    exp_name   = args.exp_name + (f"_P{args.P}" if args.loss_type == "infosft" else "")
    output_dir = os.path.join(args.output_root, exp_name)
    cfg        = TASK_CONFIGS[args.task]

    # dataset
    if args.task == "science":
        train_ds = prepare_science(args.data_path)
    else:
        train_ds = prepare_tool(args.data_path)

    # batch size: keep effective BS constant regardless of GPU count
    world_size    = int(os.environ.get("WORLD_SIZE", 1))
    per_device_bs = cfg["per_device_bs"]
    grad_accum    = args.batch_size // (per_device_bs * world_size)
    assert grad_accum >= 1, (
        f"batch_size ({args.batch_size}) too small for "
        f"{world_size} GPUs × per_device_bs {per_device_bs}"
    )

    sft_args = SFTConfig(
        seed=args.seed,
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=cfg["max_length"],
        bf16=True,
        loss_type="nll",
        assistant_only_loss=True,
        logging_strategy="steps", logging_steps=10, logging_first_step=True,
        save_strategy="epoch", save_only_model=True, save_total_limit=1,
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