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
    Compute the nucleus size (NUMBER of tokens covering `p` probability mass)
    for each generated token position.
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


def generate_story_final(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    schedule: List[float],
    tokens_per_chunk: int = 70,
    prompt_format: str = "instruct",
    repetition_penalty: float = 1.3,
) -> Dict[str, Any]:
    """
    Generate one story in N chunks, each chunk using its scheduled temperature.
    """
    device = next(model.parameters()).device
    input_ids = _build_input_ids(tokenizer, prompt, prompt_format, device)

    chunk_texts: List[List[float]] = []
    chunk_entropies: List[List[float]] = []
    chunk_nucleus_sizes: List[List[int]] = []

    for tau in schedule:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=tokens_per_chunk,
                do_sample=True,
                temperature=tau,
                top_p=1.0,          # disable nucleus sampling — temperature is the sole variable
                top_k=0,            # disable top-k sampling for the same reason
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        output_ids = output.sequences
        scores = output.scores          # tuple of [1, vocab_size], one per new token

        new_ids = output_ids[0, input_ids.shape[1]:]
        chunk_texts.append(tokenizer.decode(new_ids, skip_special_tokens=True))

        # We ended up not using these two lines, however we kept them for completeness
        chunk_entropies.append(_entropy_from_logits(scores))
        chunk_nucleus_sizes.append(_nucleus_size_from_logits(scores, p=0.9))

        # The full output (prompt + generated so far) becomes the context for the next chunk.
        input_ids = output_ids

    return {
        "prompt":             prompt,
        "schedule":           schedule,
        "tokens_per_chunk":   tokens_per_chunk,
        "chunk_texts":        chunk_texts,
        "story":              " ".join(chunk_texts),
        "chunk_entropies":    chunk_entropies,
        "chunk_nucleus_sizes": chunk_nucleus_sizes,
    }


def run_generation_final(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[Dict[str, Any]],
    schedule_name: str,
    schedule: List[float],
    tokens_per_chunk: int,
    model_name: str,
    output_dir: str,
    prompt_format: str = "instruct",
    repetition_penalty: float = 1.3,
    n_copies: int = 30,
) -> str:
    """
    Generate n_copies independent stories per prompt under one temperature schedule.

    Output format (one JSON line per story in stories.jsonl):
        {
          "prompt_id":    int,
          "shadow_id":    int,        # 0 … n_copies-1
          "model":        str,
          "schedule":     str,        # schedule name
          "temperatures": List[float],
          "prompt":       str,
          "tokens_per_chunk": int,
          "chunk_texts":  List[str],
          "story":        str,
          "chunk_entropies":    List[List[float]],
          "chunk_nucleus_sizes": List[List[int]]
        }
    """
    from tqdm.auto import tqdm

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "stories.jsonl")

    # "Resume" ckpt
    # Read any already-completed (prompt_id, shadow_id) pairs so we can skip
    # them on a re-run after a premature kill.
    done: set = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["prompt_id"], r["shadow_id"]))
            except (json.JSONDecodeError, KeyError):
                pass
        if done:
            print(f"[resume] {schedule_name}: skipping {len(done)} already-completed stories.")

    total = len(prompts) * n_copies
    remaining = total - len(done)

    # Append to existing file so completed stories are preserved.
    with open(out_path, "a") as f:
        pbar = tqdm(total=remaining, desc=f"schedule={schedule_name}")
        for item in prompts:
            for copy_id in range(n_copies):
                if (item["prompt_id"], copy_id) in done:
                    continue
                result = generate_story_final(
                    model, tokenizer, item["prompt"], schedule,
                    tokens_per_chunk=tokens_per_chunk,
                    prompt_format=prompt_format,
                    repetition_penalty=repetition_penalty,
                )
                # Build record, replacing the temperature-list "schedule" key
                # from result with the human-readable schedule name.
                # We initially called the "copies" "shadows", hence the "shadow_id" key. While CR, we decided that "copy_id" was a more sufficient term.
                record = {
                    "prompt_id":    item["prompt_id"],
                    "shadow_id":    copy_id,
                    "model":        model_name,
                    "schedule":     schedule_name,
                    "temperatures": schedule,
                }
                for k, v in result.items():
                    if k != "schedule":      # already set above as string name
                        record[k] = v
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                pbar.update(1)
        pbar.close()

    print(f"Saved {remaining} new stories (total {total}) → {out_path}")
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
