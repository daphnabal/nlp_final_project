"""
Standalone script: score already-generated stories with OPT coherence metric.
Saves results to stories_opt_scored.jsonl alongside each stories.jsonl.

Usage:
    python run_opt_coherence.py
    python run_opt_coherence.py --schedules fixed_0.5 fixed_1.0
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.generation import load_stories
from src.evaluation.metrics import compute_coherence_opt


def main(schedules):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {config.OPT_COHERENCE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.OPT_COHERENCE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.OPT_COHERENCE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    print("Model loaded.")

    for sched_name in schedules:
        in_path = os.path.join(config.results_dir(config.MODEL_NAME, sched_name), "stories.jsonl")
        out_path = os.path.join(config.results_dir(config.MODEL_NAME, sched_name), "stories_opt_scored.jsonl")

        if not os.path.exists(in_path):
            print(f"WARNING: {in_path} not found, skipping.")
            continue

        stories = load_stories(in_path)
        print(f"{sched_name}: scoring {len(stories)} stories -> {out_path}")

        with open(out_path, "w") as f:
            for s in tqdm(stories, desc=sched_name):
                result = compute_coherence_opt(model, tokenizer, s["prompt"], s["story"], device)
                s.update(result)
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                f.flush()

        print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedules", nargs="+",
        default=config.SWEEP_SCHEDULES,
        help="Schedule names to score (default: all SWEEP_SCHEDULES from config).",
    )
    args = parser.parse_args()
    main(args.schedules)
