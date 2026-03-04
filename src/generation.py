"""
Chunked story generation with dynamic temperature schedules.
"""

import json
import os
from typing import List, Dict, Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def generate_story(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    schedule: List[float],
    tokens_per_chunk: int = 120,
) -> Dict[str, Any]:
    """
    Generate a story in chunks, each chunk using the corresponding temperature.

    Args:
        model: HuggingFace causal LM (already on the right device).
        tokenizer: Matching tokenizer.
        prompt: The writing prompt string.
        schedule: List of temperatures, one per chunk.
        tokens_per_chunk: Max new tokens generated per chunk.

    Returns:
        Dict with keys: story, prompt, schedule, tokens_per_chunk, chunk_texts.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    chunk_texts = []
    for tau in schedule:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=tokens_per_chunk,
                do_sample=True,
                temperature=tau,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Only the newly generated tokens
        new_ids = output_ids[0, input_ids.shape[1]:]
        chunk_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        chunk_texts.append(chunk_text)
        # Feed full output as context for next chunk
        input_ids = output_ids

    story = " ".join(chunk_texts)
    return {
        "prompt": prompt,
        "schedule": schedule,
        "tokens_per_chunk": tokens_per_chunk,
        "chunk_texts": chunk_texts,
        "story": story,
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
                model, tokenizer, item["prompt"], schedule, tokens_per_chunk
            )
            record = {
                "prompt_id": item["prompt_id"],
                "model": model_name,
                "schedule": schedule_name,
                "temperatures": schedule,
                **result,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(prompts)} stories → {out_path}")
    return out_path


def load_stories(path: str) -> List[Dict[str, Any]]:
    """Load a stories.jsonl file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
