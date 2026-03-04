"""
Download and sample writing prompts from the WritingPrompts dataset (HuggingFace).

Run once:
    python data/prepare_prompts.py

Writes data/prompts.jsonl with {prompt_id, prompt} records.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def clean_prompt(text: str) -> str:
    """Strip common Reddit WP markers."""
    for prefix in ("[WP]", "[SP]", "[EU]", "[CW]", "[TT]"):
        text = text.replace(prefix, "").strip()
    return text.strip()


def main():
    from datasets import load_dataset

    print("Loading WritingPrompts from HuggingFace...")
    ds = load_dataset("euclaise/writingprompts", split="train")

    prompts = []
    seen = set()
    for i, row in enumerate(ds):
        text = clean_prompt(row["prompt"])
        if len(text) < 20 or text in seen:
            continue
        seen.add(text)
        prompts.append({"prompt_id": len(prompts), "prompt": text})
        if len(prompts) >= config.N_PROMPTS * 2:   # sample 2× to allow filtering
            break

    # Take first N_PROMPTS
    prompts = prompts[: config.N_PROMPTS]
    print(f"Saving {len(prompts)} prompts → {config.DATA_PATH}")

    os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)
    with open(config.DATA_PATH, "w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
