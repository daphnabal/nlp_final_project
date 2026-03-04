"""
Non-LLM evaluation metrics.

- Coherence: average token log-likelihood of the story given its prefix.
- Diversity: self-BLEU (lower = more diverse) across stories for a schedule.
"""

import math
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import PreTrainedModel, PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------

def compute_coherence(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    story: str,
    device: str = None,
) -> float:
    """
    Compute average token log-likelihood of `story` as a coherence proxy.

    Higher (less negative) → more likely / more coherent given the LM.
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer.encode(story, return_tensors="pt").to(device)
    if input_ids.shape[1] < 2:
        return float("nan")

    with torch.no_grad():
        logits = model(input_ids).logits  # (1, seq_len, vocab)

    # Shift: predict token t from tokens 0..t-1
    shift_logits = logits[0, :-1, :]           # (seq-1, vocab)
    shift_labels = input_ids[0, 1:]            # (seq-1,)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs[range(len(shift_labels)), shift_labels]
    return token_log_probs.mean().item()


def score_coherence_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    stories: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Add 'coherence' key to each story dict and return the list."""
    device = next(model.parameters()).device
    for s in stories:
        s["coherence"] = compute_coherence(model, tokenizer, s["story"], device)
    return stories


# ---------------------------------------------------------------------------
# Diversity (self-BLEU)
# ---------------------------------------------------------------------------

def _tokenize_simple(text: str) -> List[str]:
    return text.lower().split()


def self_bleu(stories: List[str], n: int = 4) -> float:
    """
    Compute self-BLEU for a list of stories.

    Each story is the hypothesis; all others are references.
    Returns the mean BLEU score — lower = more diverse.
    """
    if len(stories) < 2:
        return float("nan")

    tokenized = [_tokenize_simple(s) for s in stories]
    smoothing = SmoothingFunction().method1
    weights = tuple(1.0 / n for _ in range(n))

    scores = []
    for i, hyp in enumerate(tokenized):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        score = sentence_bleu(refs, hyp, weights=weights, smoothing_function=smoothing)
        scores.append(score)

    return sum(scores) / len(scores)


def score_diversity(stories_by_schedule: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    """
    Compute self-BLEU for each schedule.

    Args:
        stories_by_schedule: {schedule_name: [story_dict, ...]}

    Returns:
        {schedule_name: self_bleu_score}
    """
    return {
        sched: self_bleu([s["story"] for s in stories])
        for sched, stories in stories_by_schedule.items()
    }
