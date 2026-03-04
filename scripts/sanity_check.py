"""
sanity_check.py — Quick, lightweight checks that require no GPU/model download.

Run from the project root:
    python scripts/sanity_check.py

Each check prints PASS or FAIL. No external dependencies beyond the project itself.
"""

import os
import sys
import json
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(name: str, fn):
    try:
        fn()
        print(f"  [{PASS}] {name}")
        return True
    except Exception as e:
        print(f"  [{FAIL}] {name}")
        print(f"          {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# 1. Schedules
# ---------------------------------------------------------------------------

def test_schedules_shapes():
    from src.schedules import get_all_schedules, get_schedule
    for n in [1, 3, 5]:
        sched = get_all_schedules(n)
        for name, temps in sched.items():
            assert len(temps) == n, f"{name} has {len(temps)} temps, expected {n}"
        # get_schedule round-trip
        assert get_schedule("fixed", n) == sched["fixed"]


def test_schedules_values():
    from src.schedules import get_all_schedules
    sched = get_all_schedules(3)
    # fixed: all same
    assert len(set(sched["fixed"])) == 1
    # increasing: strictly ascending
    inc = sched["increasing"]
    assert all(inc[i] < inc[i + 1] for i in range(len(inc) - 1))
    # decreasing: strictly descending
    dec = sched["decreasing"]
    assert all(dec[i] > dec[i + 1] for i in range(len(dec) - 1))
    # peak: middle is highest
    peak = sched["peak"]
    assert peak[1] >= peak[0] and peak[1] >= peak[2]
    # valley: middle is lowest
    val = sched["valley"]
    assert val[1] <= val[0] and val[1] <= val[2]


def test_unknown_schedule_raises():
    from src.schedules import get_schedule
    try:
        get_schedule("nonexistent")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 2. Config
# ---------------------------------------------------------------------------

def test_config_paths():
    import config
    assert isinstance(config.MODEL_NAME, str) and len(config.MODEL_NAME) > 0
    assert config.EVAL_MODE in ("metrics", "gemini")
    assert config.N_CHUNKS > 0
    assert config.TOKENS_PER_CHUNK > 0
    assert config.N_PROMPTS > 0
    assert len(config.SCHEDULES) > 0


def test_results_dir_creation():
    import config
    with tempfile.TemporaryDirectory() as tmp:
        # Temporarily redirect RESULTS_DIR to tmp
        original = config.RESULTS_DIR
        config.RESULTS_DIR = tmp
        path = config.results_dir("test/model", "fixed")
        config.RESULTS_DIR = original
        assert os.path.isdir(path)
        assert "test_model" in path
        assert "fixed" in path


# ---------------------------------------------------------------------------
# 3. load_stories round-trip
# ---------------------------------------------------------------------------

def test_load_stories_roundtrip():
    from src.generation import load_stories

    sample = [
        {"prompt_id": 0, "story": "Once upon a time.", "schedule": "fixed", "coherence": -2.1},
        {"prompt_id": 1, "story": "The dragon roared.", "schedule": "fixed", "coherence": -1.9},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for rec in sample:
            f.write(json.dumps(rec) + "\n")
        path = f.name

    try:
        loaded = load_stories(path)
        assert loaded == sample, f"Loaded: {loaded}"
    finally:
        os.unlink(path)


def test_load_stories_empty_lines():
    from src.generation import load_stories

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": 1}\n\n\n{"id": 2}\n')
        path = f.name
    try:
        loaded = load_stories(path)
        assert len(loaded) == 2
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 4. Metrics (no model needed — only self-BLEU diversity)
# ---------------------------------------------------------------------------

def test_self_bleu_ranking():
    """More varied stories should have lower self-BLEU (more diverse)."""
    from src.evaluation.metrics import score_diversity

    identical = [{"story": "The cat sat on the mat."} for _ in range(5)]
    varied = [
        {"story": "A dragon flew over mountains."},
        {"story": "Scientists discovered a new planet."},
        {"story": "The detective solved the mystery at dawn."},
        {"story": "Robots took over the factory floor."},
        {"story": "She painted the sunset in bold strokes."},
    ]
    scores = score_diversity({"identical": identical, "varied": varied})
    assert scores["identical"] > scores["varied"], (
        f"Identical={scores['identical']:.4f} should be > Varied={scores['varied']:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Data file
# ---------------------------------------------------------------------------

def test_prompts_file_exists_and_parseable():
    import config
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(
            f"{config.DATA_PATH} not found. Run: python data/prepare_prompts.py"
        )
    with open(config.DATA_PATH) as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) > 0, "prompts.jsonl is empty"
    first = json.loads(lines[0])
    assert "prompt" in first, f"Expected 'prompt' key, got: {list(first.keys())}"
    assert "prompt_id" in first, f"Expected 'prompt_id' key, got: {list(first.keys())}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = []

    print("\n=== Schedule checks ===")
    results.append(check("Schedule shapes (n=1,3,5)", test_schedules_shapes))
    results.append(check("Schedule values (fixed/inc/dec/peak/valley)", test_schedules_values))
    results.append(check("Unknown schedule raises ValueError", test_unknown_schedule_raises))

    print("\n=== Config checks ===")
    results.append(check("Config fields valid", test_config_paths))
    results.append(check("results_dir() creates directory", test_results_dir_creation))

    print("\n=== I/O checks ===")
    results.append(check("load_stories round-trip", test_load_stories_roundtrip))
    results.append(check("load_stories skips empty lines", test_load_stories_empty_lines))

    print("\n=== Metrics checks ===")
    results.append(check("Self-BLEU: varied < identical", test_self_bleu_ranking))

    print("\n=== Data checks ===")
    results.append(check("prompts.jsonl exists and parseable", test_prompts_file_exists_and_parseable))

    total = len(results)
    passed = sum(results)
    print(f"\n{'='*40}")
    print(f"  {passed}/{total} checks passed")
    if passed < total:
        sys.exit(1)
