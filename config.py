# Central configuration for the dynamic temperature story generation project.
# Swap model or eval mode here — no changes needed elsewhere.

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

EVAL_MODE = "metrics"   # "gemini" | "metrics"

N_CHUNKS = 3
STORY_TARGET_TOKENS = 360          # total tokens; divided evenly across chunks
TOKENS_PER_CHUNK = STORY_TARGET_TOKENS // N_CHUNKS   # 120

N_PROMPTS = 50                     # number of WritingPrompts examples to use

# Schedule names to run (must match keys in src/schedules.py::get_all_schedules)
SCHEDULES = ["fixed", "increasing", "decreasing", "valley", "peak"]

# Gemini settings (only used when EVAL_MODE == "gemini")
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"   # env var name

# Paths
import os

ROOT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT_DIR, "data", "prompts.jsonl")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def results_dir(model_name: str, schedule: str) -> str:
    """Return (and create) the results directory for a given (model, schedule) pair."""
    safe_model = model_name.replace("/", "_")
    path = os.path.join(RESULTS_DIR, f"{safe_model}_{schedule}")
    os.makedirs(path, exist_ok=True)
    return path
