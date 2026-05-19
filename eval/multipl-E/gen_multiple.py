import argparse, json, os
from datasets import load_dataset
from vllm import LLM, SamplingParams

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang",  required=True)
    ap.add_argument("--out_dir", required=True, help="dir to save per-problem JSONs")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset("nuprl/MultiPL-E", f"humaneval-{args.lang}", split="test")
    print(f"Loaded {len(ds)} problems for {args.lang}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Stop tokens are uniform within a language → batch all prompts in one call
    stop_tokens = ds[0]["stop_tokens"]
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=stop_tokens,
        n=1,
    )
    outputs = llm.generate([ex["prompt"] for ex in ds], sp)

    # Write one file per problem
    for ex, out in zip(ds, outputs):
        rec = {
            "name":        ex["name"],
            "language":    ex["language"],
            "prompt":      ex["prompt"],
            "tests":       ex["tests"],
            "stop_tokens": ex["stop_tokens"],
            "completions": [out.outputs[0].text],
        }
        with open(os.path.join(args.out_dir, f"{ex['name']}.json"), "w") as f:
            json.dump(rec, f)

    print(f"Saved {len(ds)} files to {args.out_dir}")

if __name__ == "__main__":
    main()
