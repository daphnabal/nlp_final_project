"""
Gemini-as-judge evaluation.

Scores each (prompt, story) pair on creativity and coherence (1–5 each).
Only used when config.EVAL_MODE == "gemini".
"""

import os
import re
import json
import time
from typing import Dict, Any, List

import google.generativeai as genai

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


def _init_gemini() -> genai.GenerativeModel:
    api_key = os.environ.get(config.GEMINI_API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            f"Set the {config.GEMINI_API_KEY_ENV} environment variable to use Gemini judge."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(config.GEMINI_MODEL)


def score_story(
    model: genai.GenerativeModel,
    prompt: str,
    story: str,
    retries: int = 3,
    delay: float = 2.0,
) -> Dict[str, Any]:
    """
    Call Gemini to score a single (prompt, story) pair.

    Returns dict with keys: creativity, coherence, reasoning.
    On parse failure returns {creativity: None, coherence: None, reasoning: <raw>}.
    """
    filled = RUBRIC_PROMPT.format(prompt=prompt, story=story)
    for attempt in range(retries):
        try:
            response = model.generate_content(filled)
            raw = response.text.strip()
            # Extract JSON — Gemini sometimes wraps in ```json ... ```
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                return {"creativity": None, "coherence": None, "reasoning": str(e)}
    return {"creativity": None, "coherence": None, "reasoning": "max retries exceeded"}


def score_stories_batch(
    stories: List[Dict[str, Any]],
    rate_limit_delay: float = 1.5,
) -> List[Dict[str, Any]]:
    """
    Score a list of story dicts with Gemini.

    Adds 'creativity' and 'coherence' keys to each dict.
    """
    from tqdm.auto import tqdm

    gemini_model = _init_gemini()
    for s in tqdm(stories, desc="Gemini judge"):
        scores = score_story(gemini_model, s["prompt"], s["story"])
        s["creativity"] = scores.get("creativity")
        s["coherence_llm"] = scores.get("coherence")   # avoid clash with metrics key
        s["judge_reasoning"] = scores.get("reasoning")
        time.sleep(rate_limit_delay)
    return stories
