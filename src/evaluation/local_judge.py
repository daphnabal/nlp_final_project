"""
Local LLM-as-judge evaluation (on-prem, no API key required).

Scores each (prompt, story) pair on creativity and coherence (1–5 each).
Uses a local HuggingFace instruction-tuned model (see config.LOCAL_JUDGE_MODEL).
Only used when config.EVAL_MODE == "local_llm" or "all".
"""

import re
import json
from typing import Dict, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


RUBRIC_PROMPT = """\
You are a literary critic evaluating a short story written in response to a creative writing prompt.

Writing Prompt:
{prompt}

Story:
{story}

Score the story on two dimensions, each from 1 to 5:

1. **Creativity** (1=very formulaic/predictable, 5=highly original and imaginative)
2. **Coherence** (1=incoherent/confusing, 5=well-structured and easy to follow)

Respond ONLY with a JSON object like:
{{"creativity": <int>, "coherence": <int>, "reasoning": "<one sentence>"}}
"""

SYSTEM_MSG = (
    "You are a literary critic. Respond ONLY with a valid JSON object. "
    "Do not include any explanation or markdown."
)


def load_judge_model(model_name: str = None):
    """Load and return (model, tokenizer) for the local judge."""
    model_name = model_name or config.LOCAL_JUDGE_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[local_judge] Loading judge model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def score_story(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    story: str,
    retries: int = 3,
) -> Dict[str, Any]:
    """
    Use the local judge model to score a single (prompt, story) pair.

    Returns dict with keys: creativity, coherence, reasoning.
    On parse failure returns {creativity: None, coherence: None, reasoning: <raw>}.
    """
    filled = RUBRIC_PROMPT.format(prompt=prompt, story=story)
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": filled},
    ]

    # Apply chat template — works for Qwen, TinyLlama, Mistral, Phi-3, etc.
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    for attempt in range(retries):
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=150,
                    do_sample=False,       # greedy — deterministic, cheaper
                    pad_token_id=tokenizer.eos_token_id,
                )
            # Decode only the newly generated tokens
            new_ids = output_ids[0, input_ids.shape[1]:]
            raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                # Validate expected keys
                if "creativity" in parsed and "coherence" in parsed:
                    return parsed
        except json.JSONDecodeError:
            pass  # retry
        except Exception as e:
            if attempt == retries - 1:
                return {"creativity": None, "coherence": None, "reasoning": str(e)}

    return {"creativity": None, "coherence": None, "reasoning": f"parse failed after {retries} retries: {raw!r}"}


def score_stories_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    stories: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Score a list of story dicts with the local judge model.

    Adds 'creativity', 'coherence_llm', and 'judge_reasoning' keys to each dict.
    """
    from tqdm.auto import tqdm

    for s in tqdm(stories, desc="Local LLM judge"):
        scores = score_story(model, tokenizer, s["prompt"], s["story"])
        s["creativity"] = scores.get("creativity")
        s["coherence_llm"] = scores.get("coherence")   # avoid clash with metrics key
        s["judge_reasoning"] = scores.get("reasoning")
    return stories
