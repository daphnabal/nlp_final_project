"""
run_pipeline.py — CLI entry point for the full experiment pipeline.

Usage examples:
    python scripts/run_pipeline.py generate
    python scripts/run_pipeline.py evaluate
    python scripts/run_pipeline.py analyze
    python scripts/run_pipeline.py all

    # Override config via flags:
    python scripts/run_pipeline.py generate --n_prompts 5 --schedules fixed increasing
    python scripts/run_pipeline.py evaluate --eval_mode gemini
    python scripts/run_pipeline.py all --n_prompts 10 --model Qwen/Qwen2.5-0.5B
"""

import argparse
import json
import os
import sys

# Ensure project root is on the path regardless of where the script is called from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config


# ---------------------------------------------------------------------------
# Stage: generate
# ---------------------------------------------------------------------------

def run_generate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.schedules import get_schedule
    from src.generation import run_generation

    model_name        = args.model or config.MODEL_NAME
    n_prompts         = args.n_prompts or config.N_PROMPTS
    schedules         = args.schedules or config.SCHEDULES
    n_chunks          = args.n_chunks or config.N_CHUNKS
    tpc               = args.tokens_per_chunk or config.TOKENS_PER_CHUNK
    prompt_format     = args.prompt_format or config.PROMPT_FORMAT
    repetition_penalty = args.repetition_penalty or config.REPETITION_PENALTY

    # Load prompts
    with open(config.DATA_PATH) as f:
        all_prompts = [json.loads(line) for line in f if line.strip()]
    prompts = all_prompts[:n_prompts]
    print(f"[generate] Using {len(prompts)} prompts from {config.DATA_PATH}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[generate] Device: {device} | Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    for sched_name in schedules:
        schedule = get_schedule(sched_name, n_chunks)
        out_dir = config.results_dir(model_name, sched_name)
        print(f"\n=== Schedule: {sched_name} | temps: {schedule} | format: {prompt_format} ===")
        run_generation(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            schedule_name=sched_name,
            schedule=schedule,
            tokens_per_chunk=tpc,
            model_name=model_name,
            output_dir=out_dir,
            prompt_format=prompt_format,
            repetition_penalty=repetition_penalty,
        )
    print("\n[generate] Done.")


# ---------------------------------------------------------------------------
# Stage: evaluate
# ---------------------------------------------------------------------------

def run_evaluate(args):
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.generation import load_stories
    from src.evaluation.metrics import score_coherence_batch, score_diversity

    model_name = args.model or config.MODEL_NAME
    schedules  = args.schedules or config.SCHEDULES
    eval_mode  = args.eval_mode or config.EVAL_MODE

    # Load generated stories
    stories_by_schedule = {}
    for sched_name in schedules:
        path = os.path.join(config.results_dir(model_name, sched_name), "stories.jsonl")
        if os.path.exists(path):
            stories_by_schedule[sched_name] = load_stories(path)
            print(f"[evaluate] {sched_name}: {len(stories_by_schedule[sched_name])} stories")
        else:
            print(f"[evaluate] WARNING: {path} not found — run generate first")

    if not stories_by_schedule:
        print("[evaluate] No stories found. Aborting.")
        return

    if eval_mode in ("metrics", "all"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model = model.to(device)
        model.eval()

        for sched_name, stories in stories_by_schedule.items():
            print(f"[evaluate] Scoring coherence: {sched_name}")
            score_coherence_batch(model, tokenizer, stories)

        diversity_scores = score_diversity(stories_by_schedule)
        print("\n[evaluate] Diversity (self-BLEU, lower = more diverse):")
        for sched, score in diversity_scores.items():
            print(f"  {sched}: {score:.4f}")

        # Free generation model from GPU before loading judge
        if eval_mode == "all":
            import gc
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if eval_mode in ("local_llm", "all"):
        from src.evaluation.local_judge import load_judge_model, score_stories_batch as local_score_batch
        judge_model, judge_tokenizer = load_judge_model()
        for sched_name, stories in stories_by_schedule.items():
            print(f"[evaluate] Local LLM judging: {sched_name}")
            local_score_batch(judge_model, judge_tokenizer, stories)

    elif eval_mode == "gemini":
        from src.evaluation.llm_judge import score_stories_batch
        for sched_name, stories in stories_by_schedule.items():
            print(f"[evaluate] Gemini judging: {sched_name}")
            score_stories_batch(stories)

    # Save scores
    for sched_name, stories in stories_by_schedule.items():
        out_dir = config.results_dir(model_name, sched_name)
        scores_path = os.path.join(out_dir, "scores.jsonl")
        with open(scores_path, "w") as f:
            for s in stories:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[evaluate] Saved → {scores_path}")

    if eval_mode in ("metrics", "all") and "diversity_scores" in dir():
        div_path = os.path.join(config.RESULTS_DIR, "diversity.json")
        with open(div_path, "w") as f:
            json.dump(diversity_scores, f, indent=2)
        print(f"[evaluate] Saved diversity → {div_path}")

    print("\n[evaluate] Done.")


# ---------------------------------------------------------------------------
# Stage: analyze
# ---------------------------------------------------------------------------

def run_analyze(args):
    import json
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (safe for scripts)
    import matplotlib.pyplot as plt

    from src.generation import load_stories

    model_name = args.model or config.MODEL_NAME
    schedules  = args.schedules or config.SCHEDULES

    rows = []
    for sched_name in schedules:
        scores_path = os.path.join(config.results_dir(model_name, sched_name), "scores.jsonl")
        if not os.path.exists(scores_path):
            print(f"[analyze] WARNING: {scores_path} not found")
            continue
        for s in load_stories(scores_path):
            rows.append(s)

    if not rows:
        print("[analyze] No scored stories found. Aborting.")
        return

    df = pd.DataFrame(rows)
    print(f"[analyze] Loaded {len(df)} rows | columns: {df.columns.tolist()}")

    # Summary table
    score_cols = [c for c in ["coherence", "coherence_llm", "creativity"] if c in df.columns]
    if score_cols:
        summary = df.groupby("schedule")[score_cols].agg(["mean", "std"]).round(4)
        print("\n--- Summary ---")
        print(summary.to_string())

    # Bar chart: coherence
    if "coherence" in df.columns:
        means = df.groupby("schedule")["coherence"].mean().reindex(schedules)
        stds  = df.groupby("schedule")["coherence"].std().reindex(schedules)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(means.index, means.values, yerr=stds.values, capsize=4,
               color="steelblue", alpha=0.8)
        ax.set_title("Mean Log-Likelihood Coherence by Schedule")
        ax.set_xlabel("Schedule")
        ax.set_ylabel("Avg token log-prob")
        plt.tight_layout()
        out = os.path.join(config.RESULTS_DIR, "coherence_bar.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[analyze] Saved → {out}")

    # Box plot: coherence
    if "coherence" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        data = [df[df["schedule"] == s]["coherence"].dropna().values for s in schedules]
        ax.boxplot(data, labels=schedules, patch_artist=True)
        ax.set_title("Coherence Distribution by Schedule")
        ax.set_xlabel("Schedule")
        ax.set_ylabel("Avg token log-prob")
        plt.tight_layout()
        out = os.path.join(config.RESULTS_DIR, "coherence_box.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[analyze] Saved → {out}")

    # LLM judge scores
    if "creativity" in df.columns and df["creativity"].notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, col, title in zip(
            axes,
            ["creativity", "coherence_llm"],
            ["Creativity (LLM judge)", "Coherence (LLM judge)"],
        ):
            means = df.groupby("schedule")[col].mean().reindex(schedules)
            stds  = df.groupby("schedule")[col].std().reindex(schedules)
            ax.bar(means.index, means.values, yerr=stds.values, capsize=4,
                   color="coral", alpha=0.8)
            ax.set_title(title)
            ax.set_ylim(1, 5)
            ax.set_xlabel("Schedule")
            ax.set_ylabel("Score (1–5)")
        plt.tight_layout()
        out = os.path.join(config.RESULTS_DIR, "llm_judge_bars.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[analyze] Saved → {out}")

    # Diversity bar
    div_path = os.path.join(config.RESULTS_DIR, "diversity.json")
    if os.path.exists(div_path):
        with open(div_path) as f:
            diversity = json.load(f)
        scheds = [s for s in schedules if s in diversity]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(scheds, [diversity[s] for s in scheds], color="seagreen", alpha=0.8)
        ax.set_title("Self-BLEU Diversity by Schedule (lower = more diverse)")
        ax.set_xlabel("Schedule")
        ax.set_ylabel("Self-BLEU")
        plt.tight_layout()
        out = os.path.join(config.RESULTS_DIR, "diversity_bar.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[analyze] Saved → {out}")
        print("Diversity scores:", diversity)

    print("\n[analyze] Done.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic temperature story generation pipeline")
    parser.add_argument(
        "stage",
        choices=["generate", "evaluate", "analyze", "all"],
        help="Which pipeline stage(s) to run",
    )
    parser.add_argument("--model",            type=str,   default=None, help=f"HuggingFace model name (default: {config.MODEL_NAME})")
    parser.add_argument("--n_prompts",        type=int,   default=None, help=f"Number of prompts (default: {config.N_PROMPTS})")
    parser.add_argument("--schedules",        nargs="+",  default=None, help=f"Schedules to run (default: all five)")
    parser.add_argument("--n_chunks",         type=int,   default=None, help=f"Number of chunks per story (default: {config.N_CHUNKS})")
    parser.add_argument("--tokens_per_chunk", type=int,   default=None, help=f"Tokens per chunk (default: {config.TOKENS_PER_CHUNK})")
    parser.add_argument("--eval_mode",          type=str,   default=None, choices=["metrics", "local_llm", "all", "gemini"], help=f"Evaluation mode (default: {config.EVAL_MODE})")
    parser.add_argument("--prompt_format",      type=str,   default=None, choices=["base", "instruct"], help=f"Prompt framing format (default: {config.PROMPT_FORMAT})")
    parser.add_argument("--repetition_penalty", type=float, default=None, help=f"Repetition penalty for generation (default: {config.REPETITION_PENALTY})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.stage in ("generate", "all"):
        run_generate(args)
    if args.stage in ("evaluate", "all"):
        run_evaluate(args)
    if args.stage in ("analyze", "all"):
        run_analyze(args)
