"""
run_opt_coherence_final.py — score final-experiment stories with OPT coherence.

Coherence metric
----------------
Follows the implementation in compute_coherence.py (Su & Collier 2022):

    coherence(story | prompt) = mean_{i} log p_M(story_i | [BOS + prompt] ⊕ story_{<i})

where M is a *separate* evaluator model (not the generator), loaded as
OPTForCausalLM with GPT2Tokenizer.  Using a different model avoids the trivial
self-coherence bias.

Two model sizes are run sequentially to provide complementary signals:
    facebook/opt-125m   — lightweight cross-check
    facebook/opt-1.3b   — stronger coherence proxy

Each model is loaded, all schedules are scored, then freed from GPU before the
next model loads.  Peak VRAM stays ≤ ~4 GB (opt-1.3b fp16 ≈ 2.6 GB).

Input
-----
Reads  results_final/<safe_gen_model>_<schedule>/stories.jsonl
        (written by run_final_experiment.py)

Output
------
Writes results_final/<safe_gen_model>_<schedule>/stories_opt_scored.jsonl
Each line is the original story record plus two new keys per model:
    coherence_opt_125m    (float)  — mean token log-prob from OPT-125M
    coherence_opt_1_3b    (float)  — mean token log-prob from OPT-1.3B

(Key names use underscores because dots in JSON keys cause issues in pandas.)

If the scored file already exists from a first model pass, the second model
reads from it, so both scores end up in one file.

Usage
-----
    # Score all 11 schedules with both models (default)
    python scripts/run_opt_coherence_final.py

    # Score a subset of schedules
    python scripts/run_opt_coherence_final.py --schedules increasing decreasing

    # Run only one model size
    python scripts/run_opt_coherence_final.py --opt_models facebook/opt-125m

    # Override generator model name (must match the folder names in results_final/)
    python scripts/run_opt_coherence_final.py --gen_model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Defaults  (must match run_final_experiment.py)
# ---------------------------------------------------------------------------
GEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OPT_MODEL_CONFIGS = [
    {"name": "facebook/opt-125m", "key": "coherence_opt_125m"},
    {"name": "facebook/opt-1.3b",  "key": "coherence_opt_1_3b"},
]
RESULTS_DIR = os.path.join(ROOT, "results_final")
N_CHUNKS    = 7


# ---------------------------------------------------------------------------
# Coherence evaluator
# ---------------------------------------------------------------------------

class CoherenceEvaluator(torch.nn.Module):
    """
    Wraps OPTForCausalLM to compute conditional coherence of a story given a prompt.

    Architecture follows compute_coherence.py (Su & Collier 2022):
      1. Tokenise [BOS + prompt] as context and [story] as prediction.
      2. Concatenate and run a single forward pass to get per-step probabilities.
      3. Extract the probability of each story token given all preceding tokens.
      4. Return mean(log(p)) across story tokens.
    """

    def __init__(self, model_name: str):
        super().__init__()
        from transformers import GPT2Tokenizer, OPTForCausalLM

        print(f"[CoherenceEvaluator] Loading {model_name} ...")
        self.model     = OPTForCausalLM.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.vocab_size   = self.model.config.vocab_size
        self.bos_token_id = self.tokenizer.bos_token_id
        print(f"[CoherenceEvaluator] Loaded. vocab_size={self.vocab_size}")

    @torch.no_grad()
    def _forward(self, input_ids):
        """Forward pass. Returns (last_hidden_states, softmax_probabilities)."""
        outputs     = self.model(input_ids=input_ids, output_hidden_states=True)
        logits      = outputs.logits                          # [bsz, seq, vocab]
        probability = F.softmax(logits, dim=-1)
        last_hidden = outputs.hidden_states[-1]
        return last_hidden, probability

    def _extract_probabilities(self, probabilities, labels):
        """
        Extract the probability assigned to each label token.

        probabilities : [1, seq_len, vocab_size]
        labels        : [1, seq_len]
        Returns       : List[float] of length seq_len
        """
        _, seqlen, _ = probabilities.size()
        p_list = torch.unbind(probabilities, dim=1)   # seq_len tensors of [1, vocab]
        l_list = torch.unbind(labels,        dim=1)   # seq_len tensors of [1]
        result = []
        for p, l in zip(p_list, l_list):
            result.append(p[:, l].view(-1).item())
        return result

    def compute_coherence(self, context_id, prediction_id):
        """
        context_id    : [1, context_len]
        prediction_id : [1, prediction_len]
        Returns       : float — mean log-probability over prediction tokens
        """
        _, context_len    = context_id.size()
        _, prediction_len = prediction_id.size()

        concat = torch.cat([context_id, prediction_id], dim=1)
        _, probabilities = self._forward(concat)

        # The probability of prediction token i is at position (context_len + i - 1)
        # because the model predicts token t+1 from position t.
        label_probs = probabilities[:, :-1, :][:, -prediction_len:, :]
        assert label_probs.size(1) == prediction_len

        pred_probs = self._extract_probabilities(label_probs, prediction_id)
        log_probs  = [np.log(p + 1e-10) for p in pred_probs]
        return float(np.mean(log_probs))

    def evaluate_coherence(self, prefix_text: str, prediction_text: str, device) -> float:
        """Tokenise and score one (prefix, prediction) pair."""
        context_tokens = self.tokenizer.tokenize(prefix_text)
        context_ids    = [self.bos_token_id] + self.tokenizer.convert_tokens_to_ids(context_tokens)
        context_ids    = torch.LongTensor(context_ids).view(1, -1).to(device)

        pred_tokens = self.tokenizer.tokenize(prediction_text)
        pred_ids    = self.tokenizer.convert_tokens_to_ids(pred_tokens)
        if len(pred_ids) == 0:
            return float("nan")
        pred_ids = torch.LongTensor(pred_ids).view(1, -1).to(device)

        return self.compute_coherence(context_ids, pred_ids)


# ---------------------------------------------------------------------------
# Per-schedule scoring
# ---------------------------------------------------------------------------

def score_schedule(evaluator, sched_name, gen_model_name, output_key, device):
    """Score all stories in one schedule directory and write to stories_opt_scored.jsonl."""
    from tqdm.auto import tqdm
    from src.generation import load_stories

    base_dir = os.path.join(
        RESULTS_DIR,
        f"{gen_model_name.replace('/', '_')}_{sched_name}",
    )
    in_path  = os.path.join(base_dir, "stories.jsonl")
    out_path = os.path.join(base_dir, "stories_opt_scored.jsonl")

    if not os.path.exists(in_path):
        print(f"[coherence] WARNING: {in_path} not found — skipping.")
        return

    # If a scored file already exists (from a first model pass), continue from it
    # so both models' keys accumulate in the same output file.
    read_path = out_path if os.path.exists(out_path) else in_path
    stories   = load_stories(read_path)
    print(f"[coherence] {sched_name}: scoring {len(stories)} stories → {out_path}")

    evaluator.eval()
    with open(out_path, "w") as f:
        for s in tqdm(stories, desc=sched_name):
            score = evaluator.evaluate_coherence(s["prompt"], s["story"], device)
            s[output_key] = score
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            f.flush()

    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    from src.schedules import get_all_final_schedules

    gen_model_name = args.gen_model or GEN_MODEL_NAME
    all_sched_names = list(get_all_final_schedules(N_CHUNKS).keys())
    sched_names = args.schedules if args.schedules else all_sched_names

    # Which evaluator models to run
    opt_cfgs = OPT_MODEL_CONFIGS
    if args.opt_models:
        opt_cfgs = [c for c in OPT_MODEL_CONFIGS if c["name"] in args.opt_models]
        if not opt_cfgs:
            raise ValueError(
                f"None of {args.opt_models} matched known configs: "
                f"{[c['name'] for c in OPT_MODEL_CONFIGS]}"
            )

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda") if cuda_available else torch.device("cpu")

    print(f"[coherence] Device       : {device}")
    print(f"[coherence] Gen model    : {gen_model_name}")
    print(f"[coherence] Evaluators   : {[c['name'] for c in opt_cfgs]}")
    print(f"[coherence] Schedules    : {sched_names}")
    print(f"[coherence] Results dir  : {RESULTS_DIR}")

    for cfg in opt_cfgs:
        evaluator = CoherenceEvaluator(cfg["name"])
        evaluator.eval()
        if cuda_available:
            evaluator = evaluator.to(device)

        for sched_name in sched_names:
            score_schedule(evaluator, sched_name, gen_model_name, cfg["key"], device)

        # Free GPU memory before loading the next model
        del evaluator
        gc.collect()
        if cuda_available:
            torch.cuda.empty_cache()
        print(f"[coherence] {cfg['name']} freed from memory.\n")

    print("[coherence] All done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score final-experiment stories with OPT-125M and OPT-1.3B coherence"
    )
    parser.add_argument(
        "--schedules", nargs="+", default=None,
        help="Schedules to score (default: all 11)",
    )
    parser.add_argument(
        "--opt_models", nargs="+", default=None,
        help="Subset of evaluator models — choices: "
             "facebook/opt-125m, facebook/opt-1.3b (default: both)",
    )
    parser.add_argument(
        "--gen_model", type=str, default=None,
        help=f"Generator model name as used during generation "
             f"(default: {GEN_MODEL_NAME}). "
             "Must match folder names under results_final/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
