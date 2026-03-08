"""
Prometheus-as-judge evaluation (local, no API key required).

Uses the Prometheus 2 model (prometheus-eval/prometheus-7b-v2.0) loaded via
transformers. Scores each (prompt, story) pair on creativity and coherence
(1–5 each) by calling the model twice — once per rubric dimension.

Only used when config.EVAL_MODE == "prometheus".

Reference: https://arxiv.org/pdf/2504.04953
Model:     https://huggingface.co/prometheus-eval/prometheus-7b-v2.0
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


# ---------------------------------------------------------------------------
# Prometheus absolute-grading prompt template (Prometheus 2 format)
# ---------------------------------------------------------------------------

_ABS_TEMPLATE = """\
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, \
and a score rubric representing an evaluation criterion are given.
1. Write a detailed feedback that assesses the quality of the response \
strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. \
You should refer to the score rubric.
3. The output format should look as follows: \
"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback: """


_CREATIVITY_RUBRIC = """\
[Is the story highly creative and original?]
Score 1: The story is very formulaic, predictable, and unimaginative — relies entirely on clichés.
Score 2: The story shows minimal creativity with mostly familiar or expected elements.
Score 3: The story shows some creativity but relies on common tropes without a fresh twist.
Score 4: The story is quite creative with interesting, original elements that stand out.
Score 5: The story is highly original and imaginative — surprising, inventive, and memorable."""

_COHERENCE_RUBRIC = """\
[Is the story coherent and well-structured?]
Score 1: The story is completely incoherent or impossible to follow.
Score 2: The story has significant coherence issues that impede understanding.
Score 3: The story is somewhat coherent but has noticeable gaps or structural problems.
Score 4: The story is mostly coherent with clear structure and logical flow.
Score 5: The story is perfectly coherent, well-structured, and easy to follow throughout."""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_prometheus_model(model_id: str = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load and return (model, tokenizer) for the Prometheus judge."""
    model_id = model_id or config.PROMETHEUS_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[prometheus_judge] Loading: {model_id} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Single-dimension scoring
# ---------------------------------------------------------------------------

def _score_one_rubric(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instruction: str,
    response: str,
    rubric: str,
    max_new_tokens: int = 200,
    retries: int = 2,
) -> Tuple[Optional[int], str]:
    """
    Run the Prometheus absolute-grading prompt for a single rubric dimension.

    Returns (score, feedback_text). Score is None on parse failure.
    """
    prompt = _ABS_TEMPLATE.format(
        instruction=instruction,
        response=response,
        rubric=rubric,
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    raw = ""
    for attempt in range(retries):
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = output_ids[0, input_ids.shape[1]:]
            raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            # Prometheus format: "Feedback: ... [RESULT] <int>"
            match = re.search(r"\[RESULT\]\s*([1-5])", raw)
            if match:
                return int(match.group(1)), raw
        except Exception as e:
            if attempt == retries - 1:
                return None, str(e)

    return None, f"parse failed: {raw!r}"


# ---------------------------------------------------------------------------
# Public API — mirrors local_judge.py interface
# ---------------------------------------------------------------------------

def score_story(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    story: str,
) -> Dict[str, Any]:
    """
    Score a single (prompt, story) pair on creativity and coherence.

    Returns dict with keys: creativity, coherence, reasoning.
    """
    creativity, creativity_fb = _score_one_rubric(
        model, tokenizer, prompt, story, _CREATIVITY_RUBRIC
    )
    coherence, coherence_fb = _score_one_rubric(
        model, tokenizer, prompt, story, _COHERENCE_RUBRIC
    )
    reasoning = f"Creativity: {creativity_fb} | Coherence: {coherence_fb}"
    return {"creativity": creativity, "coherence": coherence, "reasoning": reasoning}


def score_stories_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    stories: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Score a list of story dicts with Prometheus.

    Adds 'creativity', 'coherence_llm', and 'judge_reasoning' keys to each dict.
    """
    from tqdm.auto import tqdm

    for s in tqdm(stories, desc="Prometheus judge"):
        scores = score_story(model, tokenizer, s["prompt"], s["story"])
        s["creativity"] = scores.get("creativity")
        s["coherence_llm"] = scores.get("coherence")
        s["judge_reasoning"] = scores.get("reasoning")
    return stories
