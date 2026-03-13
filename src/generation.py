"""
Chunked story generation with dynamic temperature schedules.
"""

import json
import os
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _entropy_from_logits(scores_tuple) -> List[float]:
    """
    Compute per-token entropy from a tuple of per-step logits.

    Args:
        scores_tuple: Tuple of tensors (one per generated token), each shape [1, vocab_size].
                      These are the post-processed logits (temperature + repetition penalty
                      already applied) as returned by model.generate(output_scores=True).

    Returns:
        List of scalar entropy values (nats), one per token.
    """
    entropies = []
    for step_logits in scores_tuple:
        probs = F.softmax(step_logits[0], dim=-1)          # [vocab_size]
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum().item()
        entropies.append(entropy)
    return entropies


def _nucleus_size_from_logits(scores_tuple, p: float = 0.9) -> List[int]:
    """
    Compute the nucleus size (number of tokens covering `p` probability mass)
    for each generated token position.

    Args:
        scores_tuple: Same tuple as in _entropy_from_logits.
        p: Probability mass threshold (default 0.9 → 90% nucleus).

    Returns:
        List of integers (nucleus sizes), one per token.
    """
    sizes = []
    for step_logits in scores_tuple:
        probs = F.softmax(step_logits[0], dim=-1)          # [vocab_size]
        sorted_probs, _ = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        # Smallest k such that cumsum[k-1] >= p
        k = int((cumsum < p).sum().item()) + 1
        sizes.append(k)
    return sizes


_BASE_PREFIX = (
    "Write a short creative story based on the following prompt. "
    "Write only the story, no questions or commentary.\n\n"
    "Prompt: {prompt}\n\n"
    "Story:"
)


def _build_input_ids(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    prompt_format: str,
    device,
):
    """Encode the framed prompt into input_ids according to prompt_format."""
    if prompt_format == "instruct":
        messages = [
            {
                "role": "system",
                "content": "You are a creative fiction writer. Write engaging, imaginative short stories.",
            },
            {
                "role": "user",
                "content": (
                    "Write a short creative story based on this prompt. "
                    "Write only the story text, no titles or commentary.\n\n"
                    f"Prompt: {prompt}"
                ),
            },
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = _BASE_PREFIX.format(prompt=prompt)

    return tokenizer.encode(text, return_tensors="pt").to(device)


def generate_story(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    schedule: List[float],
    tokens_per_chunk: int = 120,
    prompt_format: str = "base",
    repetition_penalty: float = 1.3,
) -> Dict[str, Any]:
    """
    Generate a story in chunks, each chunk using the corresponding temperature.

    Args:
        model: HuggingFace causal LM (already on the right device).
        tokenizer: Matching tokenizer.
        prompt: The writing prompt string.
        schedule: List of temperatures, one per chunk.
        tokens_per_chunk: Max new tokens generated per chunk.
        prompt_format: "base" (raw text prefix) or "instruct" (chat template).
        repetition_penalty: Penalise repeated tokens (>1.0 reduces loops).

    Returns:
        Dict with keys: story, prompt, schedule, tokens_per_chunk, chunk_texts,
        chunk_entropies (List[List[float]] — per-token entropy in nats, one list per chunk),
        chunk_nucleus_sizes (List[List[int]] — 90% nucleus size per token, one list per chunk).
    """
    device = next(model.parameters()).device
    input_ids = _build_input_ids(tokenizer, prompt, prompt_format, device)

    chunk_texts = []
    chunk_texts_alt = []
    chunk_entropies: List[List[float]] = []
    chunk_nucleus_sizes: List[List[int]] = []

    for tau in schedule:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=tokens_per_chunk,
                do_sample=True,
                temperature=tau,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            output_alt = model.generate(
                input_ids,
                max_new_tokens=tokens_per_chunk,
                do_sample=True,
                temperature=tau,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_ids = output.sequences
        scores = output.scores   # tuple of [1, vocab_size] tensors, one per new token

        # Only the newly generated tokens
        new_ids = output_ids[0, input_ids.shape[1]:]
        chunk_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        chunk_texts.append(chunk_text)

        new_ids_alt = output_alt[0, input_ids.shape[1]:]
        chunk_texts_alt.append(tokenizer.decode(new_ids_alt, skip_special_tokens=True))

        # Per-token metrics for this chunk
        chunk_entropies.append(_entropy_from_logits(scores))
        chunk_nucleus_sizes.append(_nucleus_size_from_logits(scores, p=0.9))

        # Feed PRIMARY output as context for next chunk (shadow is discarded for continuation)
        input_ids = output_ids

    story = " ".join(chunk_texts)
    return {
        "prompt": prompt,
        "schedule": schedule,
        "tokens_per_chunk": tokens_per_chunk,
        "chunk_texts": chunk_texts,
        "chunk_texts_alt": chunk_texts_alt,
        "story": story,
        "chunk_entropies": chunk_entropies,
        "chunk_nucleus_sizes": chunk_nucleus_sizes,
    }


def run_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[Dict[str, Any]],
    schedule_name: str,
    schedule: List[float],
    tokens_per_chunk: int,
    model_name: str,
    output_dir: str,
    prompt_format: str = "base",
    repetition_penalty: float = 1.3,
) -> str:
    """
    Run generation for all prompts under one schedule and save to JSONL.

    Args:
        prompts: List of dicts with at least keys 'prompt_id' and 'prompt'.
        schedule_name: String label for this schedule.
        schedule: List of temperatures.
        tokens_per_chunk: Tokens per chunk.
        model_name: String identifier saved in output.
        output_dir: Directory to write stories.jsonl.

    Returns:
        Path to the written stories.jsonl file.
    """
    from tqdm.auto import tqdm

    out_path = os.path.join(output_dir, "stories.jsonl")
    with open(out_path, "w") as f:
        for item in tqdm(prompts, desc=f"schedule={schedule_name}"):
            result = generate_story(
                model, tokenizer, item["prompt"], schedule, tokens_per_chunk,
                prompt_format=prompt_format,
                repetition_penalty=repetition_penalty,
            )
            record = {
                "prompt_id": item["prompt_id"],
                "model": model_name,
                "temperatures": schedule,
                **result,
                "schedule": schedule_name,   # override result's "schedule" (list) with the name
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Saved {len(prompts)} stories → {out_path}")
    return out_path


def load_stories(path: str) -> List[Dict[str, Any]]:
    """Load a stories.jsonl file, skipping any truncated or malformed lines."""
    stories = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                stories.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[load_stories] WARNING: skipping malformed line {i} in {path} ({e})")
    return stories
