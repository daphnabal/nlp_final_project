# Central configuration for the dynamic temperature story generation project.
# Swap model or eval mode here — no changes needed elsewhere.

MODEL_NAME = "Qwen/Qwen2.5-0.5B"          # base model (prompt_format="base")
MODEL_NAME_2 = "Qwen/Qwen2.5-1.5B-Instruct"  # instruct model (prompt_format="instruct")

EVAL_MODE = "all"   # "metrics" | "local_llm" | "all" | "gemini"

# Local on-prem judge model (used when EVAL_MODE == "local_llm" or "all")
LOCAL_JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Generation settings
PROMPT_FORMAT = "base"       # "base" (raw text prefix) | "instruct" (chat template)
REPETITION_PENALTY = 1.3     # >1.0 discourages repetition loops

# Temperature sweep baselines
SWEEP_TEMPS = [0.5, 0.7, 1.0, 1.2, 1.5]
SWEEP_SCHEDULES = [f"sweep_{t}" for t in SWEEP_TEMPS]

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
# DATA_PATH = os.path.join(ROOT_DIR, "data", "prompts.jsonl")
# RESULTS_DIR = os.path.join(ROOT_DIR, "results", "results_eval")
DATA_PATH = "/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/prompts.jsonl"
RESULTS_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/results"

def results_dir(model_name: str, schedule: str) -> str:
    """Return (and create) the results directory for a given (model, schedule) pair."""
    safe_model = model_name.replace("/", "_")
    path = os.path.join(RESULTS_DIR, f"{safe_model}_{schedule}")
    os.makedirs(path, exist_ok=True)
    return path
