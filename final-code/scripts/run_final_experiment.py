"""
run_final_experiment.py — generation entrypoint for the final experiment.

Experiment design
-----------------
- 30 curated WritingPrompts prompts  (data/chosen_prompts.jsonl)
- 30 independent shadow stories per prompt  (N_SHADOWS)
- 7 chunks per story, 70 tokens per chunk  → ~490 tokens per story
- 11 temperature schedules:
    Fixed-temperature baselines (7):
        fixed_temperature_{0.01, 0.334, 0.667, 1.0, 1.334, 1.667, 2.0}
    Named dynamic schedules (4):
        increasing, decreasing, valley, peak
- Total: 30 prompts × 30 shadows × 11 schedules = 9,900 stories

Output location
---------------
    results_final/<safe_model>_<schedule_name>/stories.jsonl

Each JSONL line is one story record:
    {prompt_id, shadow_id, model, schedule, temperatures,
     prompt, tokens_per_chunk, chunk_texts, story,
     chunk_entropies, chunk_nucleus_sizes}

Usage
-----
    # Run all 11 schedules (default)
    python scripts/run_final_experiment.py

    # Run a subset of schedules
    python scripts/run_final_experiment.py --schedules increasing decreasing

    # Quick sanity-check without loading the model
    python scripts/run_final_experiment.py --dry_run

    # Override number of shadows (e.g. 3 for debugging)
    python scripts/run_final_experiment.py --n_shadows 3 --schedules fixed_temperature_1.0
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Experiment constants  (edit here to change the run configuration)
# ---------------------------------------------------------------------------
MODEL_NAME        = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_FORMAT     = "instruct"       # uses the model's chat template
N_CHUNKS          = 7
TOKENS_PER_CHUNK  = 70               # 7 × 70 = 490 tokens per story
N_SHADOWS         = 30               # independent stories per prompt
REPETITION_PENALTY = 1.3
DATA_PATH         = os.path.join(ROOT, "data", "chosen_prompts.jsonl")
RESULTS_DIR       = os.path.join(ROOT, "results_final")


def results_dir_for(model_name: str, schedule_name: str) -> str:
    """Return (and create) the output directory for a (model, schedule) pair."""
    safe = model_name.replace("/", "_")
    path = os.path.join(RESULTS_DIR, f"{safe}_{schedule_name}")
    os.makedirs(path, exist_ok=True)
    return path


def main(args):
    from src.schedules import get_all_final_schedules, get_final_schedule
    from src.generation import run_generation_final

    # ---- Prompts ----
    with open(DATA_PATH) as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    print(f"[experiment] Loaded {len(prompts)} prompts from {DATA_PATH}")

    # ---- Schedules ----
    all_schedules = get_all_final_schedules(N_CHUNKS)
    sched_names = args.schedules if args.schedules else list(all_schedules.keys())
    n_shadows = args.n_shadows if args.n_shadows is not None else N_SHADOWS

    print(f"[experiment] Schedules ({len(sched_names)}): {sched_names}")
    print(f"[experiment] n_shadows={n_shadows} | n_prompts={len(prompts)} "
          f"| n_chunks={N_CHUNKS} | tokens_per_chunk={TOKENS_PER_CHUNK}")
    print(f"[experiment] Total stories to generate: "
          f"{len(sched_names) * len(prompts) * n_shadows:,}")

    # ---- Dry run: just print schedule definitions and exit ----
    if args.dry_run:
        print("\n[dry_run] Schedule definitions:")
        for name in sched_names:
            sched = get_final_schedule(name, N_CHUNKS)
            print(f"  {name:35s}: {sched}")
        print("[dry_run] Exiting without loading model.")
        return

    # ---- Model ----
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = args.model or MODEL_NAME
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[experiment] Device: {device} | Model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    # ---- Generation loop ----
    for sched_name in sched_names:
        schedule = get_final_schedule(sched_name, N_CHUNKS)
        out_dir = results_dir_for(model_name, sched_name)
        print(f"\n=== {sched_name} | temps: {schedule} ===")
        print(f"    output → {out_dir}/stories.jsonl")
        run_generation_final(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            schedule_name=sched_name,
            schedule=schedule,
            tokens_per_chunk=TOKENS_PER_CHUNK,
            model_name=model_name,
            output_dir=out_dir,
            prompt_format=PROMPT_FORMAT,
            repetition_penalty=REPETITION_PENALTY,
            n_shadows=n_shadows,
        )

    print(f"\n[experiment] Done. All results saved under: {RESULTS_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final experiment: dynamic temperature story generation"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"HuggingFace model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--schedules", nargs="+", default=None,
        help="Subset of schedules to run (default: all 11). "
             "E.g. --schedules increasing decreasing fixed_temperature_1.0",
    )
    parser.add_argument(
        "--n_shadows", type=int, default=None,
        help=f"Independent stories per prompt (default: {N_SHADOWS})",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print schedule definitions and exit without loading the model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
